#!/bin/zsh
python merge_lmdb.py \
--source_dir ./../data/lmdb_batches \
--target_path ./../data/ligand_pretrain_187w.lmdb

: << 'COMMENT'
:source_dir: The directory containing the LMDB files to merge
:target_path: The path to the output LMDB file
COMMENT
