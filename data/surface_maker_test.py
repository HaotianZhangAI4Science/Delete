# python surface_maker_test.py --check_software yes
import argparse
import os
import numpy as np
import subprocess
import pymesh
import tempfile, shutil
#import Bio.PDB
from Bio.PDB import PDBParser, PDBIO, Select 
from Bio.PDB import NeighborSearch, Selection
from rdkit import Chem
from scipy.spatial import distance, KDTree
from IPython.utils import io
from joblib import Parallel, delayed
import sys
sys.path.append('../utils/masif')
from compute_normal import compute_normal
from computeAPBS import computeAPBS
from computeCharges import computeCharges, assignChargesToNewMesh
from computeHydrophobicity import computeHydrophobicity
from computeMSMS import computeMSMS
from fixmesh import fix_mesh
from save_ply import save_ply


parser = argparse.ArgumentParser()
parser.add_argument(
    '--pdb_file', action='store',required=False,type=str,default='./3cl.pdb',
    help='protein file'
)

parser.add_argument(
    '--lig_file', action='store',required=False,type=str,default='./3cl_ligand.sdf',
    help='protein file'
)

parser.add_argument(
    '--check_software', action='store',required=True,type=str,default='no',
    help='have you ever installed the dependent softwares and manually specify their locations?'
)


args = parser.parse_args()

#os.environ["LD_LIBRARY_PATH"] = '/home/haotian/Molecule_Generation/APBS-3.0.0.Linux/lib:/home/haotian/software/miniconda3/envs/deepdock/lib'
msms_bin="/home/haotian/Molecule_Generation/SurfBP/dataset/install_software/APBS-3.0.0.Linux/bin/msms"
apbs_bin = '/home/haotian/Molecule_Generation/SurfBP/dataset/install_software/APBS-3.0.0.Linux/bin/apbs'
pdb2pqr_bin="/home/haotian/Molecule_Generation/SurfBP/dataset/install_software/pdb2pqr-linux-bin64-2.1.1/pdb2pqr"
multivalue_bin="/home/haotian/Molecule_Generation/SurfBP/dataset/install_software/APBS-3.0.0.Linux/share/apbs/tools/bin/multivalue"

prot_path = './1z6e_protein.pdb'
lig_path = './1z6e_ligand.mol2'
outdir='.'
dist_threshold=8.0
use_hbond=True
use_hphob=True
use_apbs=True
mesh_res=1.0
epsilon=1.0e-6
feature_interpolation=True
out_name= None

workdir = './workdir'
os.makedirs(workdir, exist_ok=True)
protname = os.path.basename(prot_path).replace(".pdb","")
# Get atom coordinates
suffix = lig_path.split('.')[-1]
if suffix == 'mol':
    mol = Chem.MolFromMolFile(lig_path)
elif suffix == 'mol2':
    mol = Chem.MolFromMol2File(lig_path)
elif suffix == 'sdf':
    suppl = Chem.SDMolSupplier(lig_path, sanitize=False)
    mols = [mol for mol in suppl if mol]	
    mol = mols[0]
    # we just use the first mol of .sdf file by default
atomCoords = mol.GetConformers()[0].GetPositions()
# Read protein and select aminino acids in the binding pocket
parser = PDBParser(QUIET=True) # QUIET=True avoids comments on errors in the pdb.

structures = parser.get_structure('target', prot_path)
structure = structures[0] # 'structures' may contain several proteins in this case only one.

atoms  = Selection.unfold_entities(structure, 'A')
ns = NeighborSearch(atoms)

close_residues= []
for a in atomCoords:  
    close_residues.extend(ns.search(a, dist_threshold+5, level='R'))
close_residues = Selection.uniqueify(close_residues)

class SelectNeighbors(Select):
    def accept_residue(self, residue):
        if residue in close_residues:
            if all(a in [i.get_name() for i in residue.get_unpacked_list()] for a in ['N', 'CA', 'C', 'O']) or residue.resname=='HOH':
                return True
            else:
                return False
        else:
            return False
    
pdbio = PDBIO()
pdbio.set_structure(structure)
pdbio.save("%s/%s_pocket_%s.pdb"%(workdir, protname, dist_threshold+5), SelectNeighbors())

# Identify closes atom to the ligand
structures = parser.get_structure('target', "%s/%s_pocket_%s.pdb"%(workdir, protname, dist_threshold+5))
structure = structures[0] # 'structures' may contain several proteins in this case only one.
atoms = Selection.unfold_entities(structure, 'A')

dist = [distance.euclidean(atomCoords.mean(axis=0), a.get_coord()) for a in atoms]
atom_idx = np.argmin(dist)
vertices1, faces1, normals1, names1, areas1 = computeMSMS("%s/%s_pocket_%s.pdb"%(workdir, protname, dist_threshold+5),  
                                                            protonate=True, 
                                                            one_cavity=atom_idx, 
                                                            msms_bin=msms_bin,
                                                            workdir=workdir)

kdt = KDTree(atomCoords)
d, r = kdt.query(vertices1)
assert(len(d) == len(vertices1))
iface_v = np.where(d <= dist_threshold)[0]
faces_to_keep = [idx for idx, face in enumerate(faces1) if all(v in iface_v  for v in face)] 
if use_hbond:
    vertex_hbond = computeCharges(prot_path.replace(".pdb",""), vertices1, names1)    
if use_hphob:
    vertex_hphobicity = computeHydrophobicity(names1) 

vertices2 = vertices1
faces2 = faces1
# Fix the mesh.
mesh = pymesh.form_mesh(vertices2, faces2)
mesh = pymesh.submesh(mesh, faces_to_keep, 0)
with io.capture_output() as captured:
    regular_mesh = fix_mesh(mesh, mesh_res)

regular_mesh, info = pymesh.remove_degenerated_triangles(regular_mesh)
vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces, eps=epsilon)

if use_hbond:
    vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hbond, feature_interpolation)

if use_hphob:
    vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hphobicity, feature_interpolation)

if use_apbs:
    vertex_charges = computeAPBS(regular_mesh.vertices, "%s/%s_pocket_%s.pdb"%(workdir, protname, dist_threshold+5), 
                                apbs_bin, pdb2pqr_bin, multivalue_bin, workdir)
regular_mesh.add_attribute("vertex_mean_curvature")
H = regular_mesh.get_attribute("vertex_mean_curvature")
regular_mesh.add_attribute("vertex_gaussian_curvature")
K = regular_mesh.get_attribute("vertex_gaussian_curvature")
elem = np.square(H) - K
# In some cases this equation is less than zero, likely due to the method that computes the mean and gaussian curvature.
# set to an epsilon.
elem[elem<0] = 1e-8
k1 = H + np.sqrt(elem)
k2 = H - np.sqrt(elem)
# Compute the shape index 
si = (k1+k2)/(k1-k2)
si = np.arctan(si)*(2/np.pi)

# Convert to ply and save.
save_ply("%s/%s_pocket_%s.ply"%(outdir, protname, dist_threshold), regular_mesh.vertices,
        regular_mesh.faces, normals=vertex_normal, charges=vertex_charges,
        normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity,
        si=si)
        
# shutil.rmtree(workdir)
