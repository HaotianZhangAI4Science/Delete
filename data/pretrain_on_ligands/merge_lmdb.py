import lmdb
import os
import argparse
from tqdm import tqdm

def get_total_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def merge_multiple_lmdb(source_dir, target_path):
    total_size = get_total_size(source_dir)
    map_size = total_size * 2  # Double the total size to be safe

    env_target = lmdb.open(
        target_path,
        map_size=map_size,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )
    
    lmdb_files = [f for f in os.listdir(source_dir) if f.endswith('.lmdb')]
    total_keys = 0
    
    try:
        for lmdb_file in tqdm(lmdb_files, desc="Merging LMDB files"):
            source_path = os.path.join(source_dir, lmdb_file)
            env_source = lmdb.open(source_path, readonly=True, lock=False, subdir=False)
            
            try:
                with env_source.begin(write=False) as txn_source:
                    with env_target.begin(write=True) as txn_target:
                        cursor = txn_source.cursor()
                        for key, value in cursor:
                            new_key = f"{lmdb_file[:-5]}_{key.decode()}".encode()  # Prepend filename to avoid conflicts
                            txn_target.put(new_key, value)
                            total_keys += 1
            finally:
                env_source.close()
    
    finally:
        env_target.sync()
        env_target.close()
    
    return total_keys, map_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default='./data_test/lmdb_batches', help='Directory containing processed LMDB files')
    parser.add_argument('--target_path', type=str, default='./data_test/ligand_pretrain_100_parallel.lmdb', help='Path for the final merged LMDB file')
    args = parser.parse_args()
    print(args)
    
    # Ensure the target directory exists
    os.makedirs(os.path.dirname(args.target_path), exist_ok=True)
    
    # Merge LMDB files
    total_keys, map_size = merge_multiple_lmdb(args.source_dir, args.target_path)
    
    print(f"LMDB merge completed. Total keys: {total_keys}")
    print(f"Final LMDB map size: {map_size / 2 / (1024**3):.2f} GB")
    
    # Verify the merge
    env_verify = lmdb.open(args.target_path, readonly=True, lock=False, subdir=False)
    with env_verify.begin() as txn:
        actual_keys = txn.stat()['entries']
    env_verify.close()
    
    print(f"Actual keys in merged LMDB: {actual_keys}")
    assert actual_keys == total_keys, "Key count mismatch!"