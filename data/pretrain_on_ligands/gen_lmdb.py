import os
import re
import sys
import lmdb
import torch
import argparse
import pickle
import os.path as osp
from tqdm import tqdm
from glob import glob

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

from utils.protein_ligand import parse_sdf_file
from utils.data import torchify_dict, ProteinLigandData

def write_lmdb(save_name):
    output_dir = osp.dirname(save_name)
    os.makedirs(output_dir, exist_ok=True)

    try:
        os.remove(save_name)
    except:
        pass
    env_new = lmdb.open(
        save_name,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(30e9),
    )
    txn_write = env_new.begin(write=True)

    return txn_write, env_new

def extract_number(file):
    numbers = re.findall(r'\d+', file)
    return int(numbers[0]) if numbers else None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./data/ligand_pretrain_40w.lmdb')
    parser.add_argument('--cache_dir', type=str, default='./data/cache')
    args = parser.parse_args()
    print(args)
    txn_write, env_new = write_lmdb(args.save_path)
    mol_files = sorted(glob(args.cache_dir+'/*'), key=extract_number)[:1874685]
    raw_data = []
    for f in tqdm(mol_files, desc='Parsing mol files'):
        try:
            raw_data.append(torchify_dict(parse_sdf_file(f)))
        except Exception as e:
            print(e)
    pkt_fake = {'pos': torch.empty(0, 3), 'feature': torch.empty(0, 4)}

    success_id = 0
    for i, data in tqdm(enumerate(raw_data), total=len(raw_data), desc='Processing ligand data'):
        try:
            lig_data = ProteinLigandData.from_protein_ligand_dicts(ligand_dict=data, protein_dict=pkt_fake)
            txn_write.put(f'{success_id}'.encode("ascii"), pickle.dumps(lig_data, protocol=-1))
            success_id+=1
        except Exception as e:
            print(e)

    txn_write.commit()
    env_new.close()