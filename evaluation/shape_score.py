
import os
from glob import glob
import os.path as osp
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from easydict import EasyDict
import logging
import traceback

# set logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_shape(mol_a, mol_b):
    try:
        return 1 - AllChem.ShapeTanimotoDist(mol_a, mol_b)
    except Exception as e:
        logger.error(f"Error calculating similarity: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def shape_based_result(dock_mols, ori_mol):
    results = []
    for i, dock_mol in enumerate(dock_mols):
        try:
            affinity = float(dock_mol.GetProp('REMARK').splitlines()[0].split()[2:][0])
            shape_score = calculate_shape(ori_mol, dock_mol)
            if shape_score is None:
                logger.warning(f"Skipping {i} molecule, as shape score cannot be calculated")
                continue
            results.append(EasyDict({
                'rdmol': dock_mol,
                'affinity': affinity,
                'shape_score': shape_score
            }))
        except Exception as e:
            logger.error(f"Error dealing with {i} molecule: {str(e)}")
            logger.debug(traceback.format_exc())
    
    try:
        results = sorted(results, key=lambda x: x['shape_score'], reverse=True)
    except Exception as e:
        logger.error(f"Error sorting results: {str(e)}")
        logger.debug(traceback.format_exc())
    
    return results

def read_sdf(sdf_file):
    try:
        supp = Chem.SDMolSupplier(sdf_file,sanitize=False)
        mols_list = [i for i in supp if i is not None]
        if not mols_list:
            logger.warning(f"SDF file {sdf_file} does not contain any molecules")
        return mols_list
    except Exception as e:
        logger.error(f"Error reading SDF file {sdf_file}: {str(e)}")
        logger.debug(traceback.format_exc())
        return []

def write_sdf(mol_list, file):
    try:
        writer = Chem.SDWriter(file)
        for i in mol_list:
            writer.write(i)
        writer.close()
        logger.info(f"SDF {file} saved")
    except Exception as e:
        logger.error(f"Error writing SDF file {file}: {str(e)}")
        logger.debug(traceback.format_exc())

def read_pkl(file):
    try:
        with open(file, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        logger.error(f"Error reading PKL file {file}: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def write_pkl(list_data, file):
    try:
        with open(file, 'wb') as f:
            pickle.dump(list_data, f)
        logger.info(f"PKL file {file} saved")
    except Exception as e:
        logger.error(f"Error writing PKL file {file}: {str(e)}")
        logger.debug(traceback.format_exc())
        raise e  # Re-raise exception to handle in main loop

# shape_based scoring 

base_path = '/home/megagatlingpea/workdir/delete_shape'
task = 'nabla510_finetune_293'
targets = glob(osp.join(base_path, task, "*"))

if not targets:
    logger.warning(f"No targets found in {osp.join(base_path, task)}")

failed = []  # List to keep track of failed targets

for target in targets:

    target_failed = False  # Flag to indicate if current target has any errors
    results = []
    SDF_dir = osp.join(target, 'SDF')
    
    if not osp.exists(SDF_dir):
        logger.warning(f"SDF directory does not exist: {SDF_dir}")
        failed.append(target)
        continue  # Skip to next target
    
    ori_sdf = sorted(glob(osp.join(target, '*.sdf')), key=len)
    if not ori_sdf:
        logger.warning(f"No original sdf file found in {target}")
        failed.append(target)
        continue  # Skip to next target
    ori_sdf = ori_sdf[0]
    ori_mols = read_sdf(ori_sdf)
    
    if not ori_mols:
        logger.warning(f"Error reading original sdf file {ori_sdf}")
        failed.append(target)
        continue  # Skip to next target
    ori_mol = ori_mols[0]
    
    docked_sdfs = sorted(glob(osp.join(SDF_dir, '*_out.sdf')), key=lambda x: osp.basename(x).split('_')[0])
    
    if not docked_sdfs:
        logger.warning(f"No docking sdf files found in {SDF_dir}")
        failed.append(target)
        continue  # Skip to next target
    
    for docked_sdf in docked_sdfs:
        docked_mols = read_sdf(docked_sdf)
        if docked_mols:
            result = shape_based_result(docked_mols, ori_mol)
            if result is None:
                logger.warning(f"Shape-based result is None for {docked_sdf}")
                target_failed = True
                break  # Exit processing current target
            results.append(result)
        else:
            logger.warning(f"Skipping empty docking sdf file: {docked_sdf}")
            target_failed = True
            break  # Exit processing current target
    
    if target_failed:
        failed.append(target)
        continue  # Do not write pkl for this target
    
    if results:
        try:
            write_pkl(results, osp.join(target, 'shape_based_docking_results.pkl'))
        except Exception:
            failed.append(target)
            continue  # Skip to next target
    else:
        logger.warning(f"No results generated for target {target}")
        failed.append(target)

logger.info(f"Processing completed. Failed targets: {failed}")
