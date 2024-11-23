#!/bin/zsh
python gen_lmdb.py \
--save_path ./../data/ligand_pretrain_187w.lmdb \
--cache_dir ./../data/cache

: << 'COMMENT'
:save_path: The path to the output LMDB file
:cache_dir: The directory containing the sdf files
COMMENT