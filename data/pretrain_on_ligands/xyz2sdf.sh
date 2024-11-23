#!/bin/zsh
python xyz2sdf.py \
--csv_path /home/megagatlingpea/workdir/Dataset/nabladft.csv \
--xyz_dir_path /home/megagatlingpea/workdir/Dataset/nabladft-conformer \
--sdf_dir_path /home/megagatlingpea/workdir/Delete-pretrain/data_test/cache \
--range 1-2000000

: << 'COMMENT'
csv_path: The path to the CSV file containing the SMILES strings.
xyz_dir_path: The directory containing the XYZ files.
sdf_dir_path: The directory to write the SDF files to.
range: The range of indexes to process according to csv/dataset. Default is 1-1000000.
COMMENT