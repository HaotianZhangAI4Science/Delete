import subprocess
from glob import glob 
import os.path as osp
import time

data_base = './example/cb1'

ply_filename = 'Gi_protein_pocket_8.0.ply'
# frag_filenames = ['Gi_indole_del.sdf', 'Gi_sidechain_del.sdf']
frag_filenames = ['Gi_benofindole_del.sdf', 'Gi_ester_del.sdf', 'Gi_free_del.sdf']
out_dir = './outputs/Gi_protein'

# ply_filename = 'arr2_protein_pocket_8.0.ply'
# frag_filenames = ['arr2_sidechian_del.sdf', 'arr2_indole_del.sdf']
# frag_filenames = ['arr2_benofindole_del.sdf', 'arr2_ester_del.sdf', 'arr2_free_del.sdf']
# out_dir = './outputs/arr2_protein'

checkpoints = ['./checkpoint/ckpt/val_53.pt','./checkpoint/ckpt/val_119.pt','./checkpoint/ckpt/val_235.pt',\
    './checkpoint/ckpt_frag/crossdock_val_159.pt','./checkpoint/ckpt_frag/moad_val_226.pt',\
        './checkpoint/ckpt_linker/crossdock_val_192.pt','./checkpoint/ckpt_linker/crossdock_val_281.pt','./checkpoint/ckpt_linker/moad_train_262.pt','./checkpoint/ckpt_linker/moad_val_81.pt']

for frag_filename in frag_filenames:

    start_time = time.time()
    frag_file = osp.join(data_base,frag_filename)
    ply_file = osp.join(data_base,ply_filename)

    for ckpt in checkpoints:
        suboutdir = frag_filename[:-4]+'_'+ckpt.split('/')[-1][:-3]
        command = f'python -u delete.py --surf_path {ply_file} --frag_path {frag_file} --check_point {ckpt} \
            --outdir {out_dir} --suboutdir {suboutdir} '
        print(command)
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print('executed successfully.')
            print('Output:')
            print(result.stdout)
            print('consumed time: ',time.time()-start_time)
        else:
            print('execution failed.')
            print('Error:')
            print(result.stderr)
