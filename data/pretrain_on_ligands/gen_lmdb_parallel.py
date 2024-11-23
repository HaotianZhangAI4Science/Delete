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
    
    env_new = lmdb.open(
        save_name,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(30e9),  # 10 GB map size
    )
    
    return env_new

def extract_number(file):
    numbers = re.findall(r'\d+', file)
    return int(numbers[0]) if numbers else None

def process_file(file):
    try:
        data = parse_sdf_file(file)
        return torchify_dict(data)
    except Exception as e:
        print(f"Error parsing file {file}: {e}")
        return None

def process_range(mol_files, start_idx, end_idx, output_dir):
    # Filter files to be within the specified range
    mol_files_in_range = [file for file in mol_files if start_idx <= extract_number(file) < end_idx]

    save_name = f"{output_dir}/ligand_data_{start_idx}_{end_idx}.lmdb"
    # Remove existing database if it exists
    if os.path.exists(save_name):
        os.remove(save_name)
    
    env = write_lmdb(save_name)
    txn_write = env.begin(write=True)
    
    pkt_fake = {'pos': torch.empty(0, 3), 'feature': torch.empty(0, 4)}

    for i, file in enumerate(tqdm(mol_files_in_range, desc=f'Processing range {start_idx}-{end_idx}')):
        if os.path.exists(file):
            data = process_file(file)
            if data is not None:
                try:
                    lig_data = ProteinLigandData.from_protein_ligand_dicts(ligand_dict=data, protein_dict=pkt_fake)
                    txn_write.put(f'{i}'.encode("ascii"), pickle.dumps(lig_data, protocol=-1))
                    
                    if (i + 1) % 1000 == 0:
                        txn_write.commit()
                        txn_write = env.begin(write=True)
                except Exception as e:
                    print(f"Error processing data at index {start_idx + i}: {e}")
        else:
            print(f"File {file} does not exist. Skipping.")
    
    txn_write.commit()
    env.close()

def main(args):
    mol_files = glob(args.cache_dir + '/*.sdf')
    
    process_range(mol_files, args.start_idx, args.end_idx, args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', type=str, default='./data/cache')
    parser.add_argument('--output_dir', type=str, default='./data/lmdb_batches')
    parser.add_argument('--start_idx', type=int, required=True)
    parser.add_argument('--end_idx', type=int, required=True)
    args = parser.parse_args()
    print(args)
    
    main(args)
