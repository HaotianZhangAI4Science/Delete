import os
import shutil
import argparse
from tqdm.auto import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader

from models.delete import Delete
from utils.datasets import *
from utils.transforms import *
from utils.misc import *
from utils.train import *
from time import time
from utils.train import get_model_loss
from utils.datasets.pl import SurfLigandPairDataset

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/train_frag_moad.yml')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--logdir', type=str, default='./logs')

args = parser.parse_args()

config = load_config(args.config)
config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
seed_all(config.train.seed)

log_dir = get_new_log_dir(args.logdir, prefix=config_name)
ckpt_dir = os.path.join(log_dir, 'checkpoints')
os.makedirs(ckpt_dir, exist_ok=True)
logger = get_logger('train', log_dir)
logger.info(args)
logger.info(config)
shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
shutil.copytree('./models', os.path.join(log_dir, 'models'))

protein_featurizer = FeaturizeProteinAtom()
ligand_featurizer = FeaturizeLigandAtom()                   
masking = get_mask(config.train.transform.mask)
composer = AtomComposer(protein_featurizer.feature_dim, ligand_featurizer.feature_dim, config.model.encoder.knn)

edge_sampler = EdgeSample(config.train.transform.edgesampler)
cfg_ctr = config.train.transform.contrastive
contrastive_sampler = ContrastiveSample(cfg_ctr.num_real, cfg_ctr.num_fake, cfg_ctr.pos_real_std, cfg_ctr.pos_fake_std, config.model.field.knn)
transform = Compose([
    RefineData(),
    LigandCountNeighbors(),
    protein_featurizer,
    ligand_featurizer,
    masking,
    composer,

    FocalBuilder(),
    edge_sampler,
    contrastive_sampler,
])

def get_dataset(config, *args, **kwargs):
    name = config.name
    root = config.path
    if name == 'pl':
        dataset = SurfLigandPairDataset(root, *args, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)
    
    if 'split' in config:
        split_by_name = torch.load(config.split)
        split = {
            k: [dataset.name2id[n] for n in names if n in dataset.name2id]
            for k, names in split_by_name.items()
        }
        subsets = {k:Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset



dataset, subsets = get_dataset(
    config = config.dataset,
    transform = transform,
)

train_set, val_set = subsets['train'], subsets['test']
follow_batch = []
collate_exclude_keys = ['ligand_nbh_list']
val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False, follow_batch=follow_batch, exclude_keys = collate_exclude_keys,)
train_loader = DataLoader(train_set, config.train.batch_size, shuffle=False, exclude_keys = collate_exclude_keys)

model = Delete(
    config.model, 
    num_classes = contrastive_sampler.num_elements,
    num_bond_types = edge_sampler.num_bond_types,
    protein_atom_feature_dim = protein_featurizer.feature_dim,
    ligand_atom_feature_dim = ligand_featurizer.feature_dim,
).to(args.device)
print('Num of parameters is {0:.4}M'.format(np.sum([p.numel() for p in model.parameters()]) /100000 ))
optimizer = get_optimizer(config.train.optimizer, model)
scheduler = get_scheduler(config.train.scheduler, optimizer)

def update_losses(eval_loss, loss, loss_frontier, loss_pos, loss_cls, loss_edge, loss_real, loss_fake, loss_surf):
    eval_loss['total'].append(loss)
    eval_loss['frontier'].append(loss_frontier)
    eval_loss['pos'].append(loss_pos)
    eval_loss['cls'].append(loss_cls)
    eval_loss['edge'].append(loss_edge)
    eval_loss['real'].append(loss_real)
    eval_loss['fake'].append(loss_fake)
    eval_loss['surf'].append(loss_surf)
    return eval_loss

def evaluate(epoch, verbose=1):
    model.eval()
    eval_start = time()
    #eval_losses = {'total':[], 'frontier':[], 'pos':[], 'cls':[], 'edge':[], 'real':[], 'fake':[], 'surf':[] }
    eval_losses = []
    for batch in val_loader:
        batch = batch.to(args.device)  
        loss, loss_frontier, loss_pos, loss_cls, loss_edge, loss_real, loss_fake, loss_surf = get_model_loss(model, batch, config )
        eval_losses.append(loss.item())    
    average_loss = sum(eval_losses) / len(eval_losses)
    if verbose:
        logger.info('Evaluate Epoch %d | Average_Loss %.5f | Single Batch Loss %.6f | Loss(Fron) %.6f | Loss(Pos) %.6f | Loss(Cls) %.6f | Loss(Edge) %.6f | Loss(Real) %.6f | Loss(Fake) %.6f | Loss(Surf) %.6f  ' % (
                epoch, average_loss,  loss.item(), loss_frontier.item(), loss_pos.item(), loss_cls.item(), loss_edge.item(), loss_real.item(), loss_fake.item(), loss_surf.item()
                ))
    return average_loss

def load(config, model, optimizer=False, scheduler=False):
    '''
    Load model, optimizer, scheduler
    '''
    ckpt_name = config.train.ckpt_name
    resume_epoch = int(config.train.start_epoch)
    ckpt = torch.load(os.path.join(config.train.checkpoint_path,ckpt_name))
    best_loss = float(ckpt['best_loss'])
    model.load_state_dict(ckpt["model"])
    if scheduler:
        scheduler.load_state_dict(ckpt["scheduler"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer"])
        if args.device == 'cuda':
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(args.device)

    return model, best_loss, resume_epoch


def train(model, verbose=1, num_epoches=300):
    train_start = time()
    train_losses = []
    val_losses = []
    start_epoch = 0
    best_loss = 1000
    if config.train.resume_train:
        ckpt_name = config.train.ckpt_name
        model, best_loss, start_epoch = load(config, model, optimizer, scheduler)
        logger.info('load pretrained model from '.format(ckpt_name))
    logger.info('start training...')

    for epoch in range(num_epoches):
            
        model.train()
        epoch_start = time()
        batch_losses = []
        batch_cnt = 0

        for batch in train_loader:
            batch_cnt+=1
            batch = batch.to(args.device)
            loss, loss_frontier, loss_pos, loss_cls, loss_edge, loss_real, loss_fake, loss_surf = get_model_loss(model, batch, config )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            if (epoch==0 and batch_cnt <= 10):
                logger.info('Training Epoch %d | Step %d | Loss %.6f | Loss(Fron) %.6f | Loss(Pos) %.6f | Loss(Cls) %.6f | Loss(Edge) %.6f | Loss(Real) %.6f | Loss(Fake) %.6f | Loss(Surf) %.6f  ' % (
                        epoch+start_epoch, batch_cnt, loss.item(), loss_frontier.item(), loss_pos.item(), loss_cls.item(), loss_edge.item(), loss_real.item(), loss_fake.item(), loss_surf.item()
                        ))
        average_loss = sum(batch_losses) / (len(batch_losses)+1)
        train_losses.append(average_loss)
        if verbose:
            logger.info('Training Epoch %d | Average_Loss %.5f | Loss %.6f | Loss(Fron) %.6f | Loss(Pos) %.6f | Loss(Cls) %.6f | Loss(Edge) %.6f | Loss(Real) %.6f | Loss(Fake) %.6f | Loss(Surf) %.6f  ' % (
                    epoch+start_epoch, average_loss , loss.item(), loss_frontier.item(), loss_pos.item(), loss_cls.item(), loss_edge.item(), loss_real.item(), loss_fake.item(), loss_surf.item()
                    ))
        average_eval_loss = evaluate(epoch+start_epoch, verbose=1)
        val_losses.append(average_eval_loss)

        if config.train.scheduler.type=="plateau":
            scheduler.step(average_eval_loss)
        else:
            scheduler.step()
        if val_losses[-1] < best_loss:
            best_loss = val_losses[-1]
            if config.train.save:
                ckpt_path = os.path.join(ckpt_dir, 'val_%d.pt' % int(epoch+start_epoch))
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': start_epoch + epoch,
                    'best_loss': best_loss
                }, ckpt_path)
        else:
            if len(train_losses) > 20:
                if (train_losses[-1]<train_losses[-2]):
                    if config.train.save:
                        ckpt_path = os.path.join(ckpt_dir, 'train_%d.pt' % int(epoch+start_epoch))
                        torch.save({
                            'config': config,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'epoch': start_epoch + epoch,
                            'best_loss': best_loss
                        }, ckpt_path)                      
        torch.cuda.empty_cache()

train(model)