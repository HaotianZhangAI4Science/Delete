
# Script Folder

This folder contains scripts for processing and storing conformer datasets provided by nablaDFT. Below is a description of each script and usage instructions.

## Data Processing Workflow

1. **Download Conformer Dataset**
   Download the conformer dataset from the nablaDFT website. The dataset files are in xyz format.

2. **Convert xyz Files to sdf Files**
   Run the `xyz2sdf.sh` script to convert xyz files to sdf files:
   ```sh
   ./xyz2sdf.sh
   ```

3. **Store sdf Files in lmdb Format**
   Given the large size of the dataset, it is recommended to use parallel scripts for processing:

   - **Parallel Processing**
     Use the `gen_lmdb_parallel.sh` script to process sdf files in parallel and store them in lmdb format:
     ```sh
     ./gen_lmdb_parallel.sh
     ```
     This script will generate batch files. Next, run `merge_lmdb.sh` to merge the batch files into a single lmdb file:
     ```sh
     ./merge_lmdb.sh
     ```

   - **Non-Parallel Processing**
     If not using parallel processing, you can directly run the `gen_lmdb.sh` script:
     ```sh
     ./gen_lmdb.sh
     ```
     This step can skip the batch merging step.

4. **Pretrain the model**
    you may use the preprocessed data to train the model.
    ```sh
    ./pretrain.sh
    ```
  
## Script Descriptions

- **xyz2sdf.sh**
  Script to convert xyz files to sdf files.
  ```sh
  python xyz2sdf.py 
  --csv_path /path/to/your/csv_file 
  --xyz_dir_path /path/to/your/xyz_directory 
  --sdf_dir_path /path/to/your/sdf_directory # cache
  --range 1-2000000 # index range of xyz files to convert
  ```

- **gen_lmdb.sh**
  Script for non-parallel processing to store sdf files in lmdb format.
  ```sh
  python gen_lmdb.py 
  --save_path /path/to/your/lmdb_file
  --cache_dir /path/to/your/sdf_directory
  ```
- **gen_lmdb_parallel.sh**
  Script for parallel processing to store sdf files in lmdb format.Detailed information can be seen in the comments of the shell script file.

- **merge_lmdb.sh**
  Script to merge batch files generated from parallel processing into a single lmdb file.
  ```sh
  python merge_lmdb.py
  --source_dir /path/to/your/lmdb_batches
  --target_path /path/to/your/combined/lmdb_file
  ```
- **pretrain.sh**
  Script to pretrain the model. More information can be found in yml file.
  ```sh
  python delete_pretrain.py 
  --config ./../configs/pretrain_ligand.yml 
  --logdir ./../logs
  ```

- **xyz2sdf.py**
  Python script to convert xyz files to sdf files, called by `xyz2sdf.sh`.

- **gen_lmdb.py**
  Python script for non-parallel processing, called by `gen_lmdb.sh`.

- **gen_lmdb_parallel.py**
  Python script for parallel processing, called by `gen_lmdb_parallel.sh`.

- **merge_lmdb.py**
  Python script to merge batch files generated from parallel processing, called by `merge_lmdb.sh`.

- **delete_pretrain.py**
  Python script to pretrain the model, called by `pretrain.sh`.
## Folder Structure
```sh
  - Delete/
      - other files/directories
      - configs/
          - *.yml
      - logs/
      - scripts/
          - *.sh
          - *.py
          - README.md # You Are Here
      - data/ # Our Recommended Folder to Store Data
          - cache(About 7.7G in size)
            - *.sdf(Produced by xyz2sdf.sh)
          - lmdb_batches(About 20G in size)
            - *.lmdb(Produced by gen_lmdb_parallel.sh)
          - *.lmdb(Produced by merge_lmdb.sh or gen_lmdb.sh)
          - ...
``` 

## Notes

- Ensure all necessary dependencies are installed before running the scripts.
- Read the script contents before running to ensure paths and parameters are configured correctly.


## Data Sources

To download the conformer dataset and summary CSV file from nablaDFT, use the following links:

- nablaDFT GitHub: [AIRI-Institute/nablaDFT](https://github.com/AIRI-Institute/nablaDFT)
- Conformers Archive (xyz files): [Download Link](https://a002dlils-kadurin-nabladft.obs.ru-moscow-1.hc.sbercloud.ru/data/nablaDFTv2/conformers_archive_v2.tar)
- Summary CSV File: [Download Link](https://a002dlils-kadurin-nabladft.obs.ru-moscow-1.hc.sbercloud.ru/data/nablaDFTv2/summary.csv.gz)

## Acknowledgments

We would like to express our gratitude to the nablaDFT team for providing the conformer dataset. This dataset is an invaluable resource for our research and development efforts.
