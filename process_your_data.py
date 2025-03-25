import os
import pickle
import lmdb
import torch
from tqdm.auto import tqdm
import os.path as osp
from utils.chem import read_pkl
from utils.transforms import *
from utils.misc import *
from utils.surface import read_ply
from utils.protein_ligand import parse_sdf_file, parse_rdmol
from utils.data import ProteinLigandData, torchify_dict
from utils.surface import geodesic_matrix, dst2knnedge, parse_face, gds_edge_process
from rdkit import RDLogger
import argparse
from utils.datasets.pl import from utils.datasets.pl import SurfLigandPairDataset

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train_linker.yml')
    parser.add_argument('--surf_path', type=str, default='/home/haotian/Molecule_Generation/SurfGen/data/crossdock2020_surface_8',
                        help='the path storing surface files')
    parser.add_argument('--lig_path', type=str, default='/home/haotian/Molecule_Generation/SurfGen/data/crossdocked_pocket10',
                        help='the path storing ligand files')
    parser.add_argument('--index_path', type=str, default='./data/crossdock_data/index.pkl',
                        help='a list storing each pair information, including surf_file, lig_file, protein_file')
    parser.add_argument('--processed_path', type=str, default='./data/your_data.lmdb',
                        help='the path to store the processed data')
    parser.add_argument('--name2id_path', type=str, default='./data/your_data_name2id.pt',
                        help='the path to store the name2id dict, which is used to split the dataset according to the split_name.pt')
    args = parser.parse_args()
    config = load_config(args.config)

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


    index = read_pkl(parser.index_path)

    db = lmdb.open(
        args.processed_path,
        map_size=50*(1024*1024*1024),   # 10GB
        create=True,
        subdir=False,
        readonly=False, # Writable
    )
    num_skipped = 0

    import time
    start=time.time()

    with db.begin(write=True, buffers=True) as txn:
        for i, (pocket_nm, ligand_nm, protein_nm,_) in enumerate(tqdm(index)):
            if pocket_nm is None: 
                continue
            try:
                surf_nm = pocket_nm[:-6]+'_8.0_res_1.5.ply'
                sdf_file = osp.join(args.lig_path,ligand_nm)
                ply_file = osp.join(args.surf_path,surf_nm)
                
                pocket_dict = read_ply(ply_file)
                ligand_dict = parse_sdf_file(sdf_file)
                data = ProteinLigandData.from_protein_ligand_dicts(
                    protein_dict=torchify_dict(pocket_dict),
                    ligand_dict=torchify_dict(ligand_dict),
                )
                data.pocket_filename = pocket_nm
                data.protein_filename = protein_nm
                data.ligand_filename = ligand_nm
                data.surface_filename = surf_nm
                data.ligand_mol = parse_rdmol(sdf_file)

                txn.put(
                    key = str(i).encode(),
                    value = pickle.dumps(data)
                )
            except Exception as e:
                print(e)
                num_skipped += 1
                if num_skipped%100 == 0:
                    print('Skipping (%d) %s' % (num_skipped, ligand_nm, ))
                
    db.close()

    print('finished, {}'.format(time.time()-start))

    # create name2id if not exists
    SurfLigandPairDataset(index_path=args.index_path,
                        processed_path=args.processed_path,
                        name2id=args.name2id_path, transform=transform)