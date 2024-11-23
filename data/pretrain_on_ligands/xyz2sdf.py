import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import argparse
from tqdm import tqdm

# Read the first conformation's atoms and coordinates from an XYZ file
def read_xyz(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    num_atoms = int(lines[0].strip())
    atoms = []
    coordinates = []
    for line in lines[2:2 + num_atoms]:
        parts = line.split()
        atoms.append(parts[0])
        coordinates.append([float(x) for x in parts[1:4]])

    return atoms, np.array(coordinates)

# Convert XYZ files to SDF files
def xyz2sdf(csv_path, xyz_dir_path, sdf_dir_path, molecule_range, conformer_id=0):
    # Read the CSV file
    data = pd.read_csv(csv_path)

    # Ensure the SDF directory exists
    if not os.path.exists(sdf_dir_path):
        os.makedirs(sdf_dir_path)

    # Process XYZ files within the specified range
    start, end = map(int, molecule_range.split('-'))
    xyz_files = [f for f in os.listdir(xyz_dir_path) if f.endswith('.xyz')]
    xyz_files_to_process = [f for f in xyz_files if start <= int(f.split('_')[0]) < end]

    allowed_elements = {'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl'}
    skipped_molecules_count = 0

    for xyz_file in tqdm(xyz_files_to_process, desc="Processing XYZ files"):
        xyz_path = os.path.join(xyz_dir_path, xyz_file)

        # Extract MOSES id and CONFORMER id
        moses_id = int(xyz_file.split('_')[0])
        sdf_path = os.path.join(sdf_dir_path, f'{moses_id}_{conformer_id}.sdf')

        # Extract the corresponding SMILES from the DataFrame
        smiles = data[(data['MOSES id'] == moses_id) & (data['CONFORMER id'] == conformer_id)]['SMILES'].values[0]

        # Create the reference molecule (ref_mol) from the SMILES string
        ref_mol = Chem.MolFromSmiles(smiles)
        ref_mol = Chem.AddHs(ref_mol)
        AllChem.EmbedMolecule(ref_mol)

        # Read the first conformation from the XYZ file
        atoms, coordinates = read_xyz(xyz_path)

        # Check for unsupported elements
        if not all(atom in allowed_elements for atom in atoms):
            print(f"Skipping {xyz_file} due to unsupported elements.")
            skipped_molecules_count += 1
            continue

        # Check if the number of atoms in the SMILES-generated molecule matches the number in the XYZ file
        if len(atoms) != ref_mol.GetNumAtoms():
            raise ValueError("The number of atoms in the SMILES does not match the number of atoms in the XYZ file.")

        # Create a new conformer and assign XYZ coordinates to it
        conformer = Chem.Conformer(ref_mol.GetNumAtoms())
        for i, (x, y, z) in enumerate(coordinates):
            conformer.SetAtomPosition(i, Chem.rdGeometry.Point3D(x, y, z))

        # Assign the new conformer to the reference molecule
        ref_mol.RemoveAllConformers()
        ref_mol.AddConformer(conformer)

        # Save the molecule object as an SDF file
        w = Chem.SDWriter(sdf_path)
        w.write(ref_mol)
        w.close()

    print(f"Number of molecules skipped due to unsupported elements: {skipped_molecules_count}")

if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Convert XYZ files to SDF files using SMILES from a CSV file.')
    parser.add_argument('--csv_path', type=str, default='/home/megagatlingpea/workdir/Dataset/nabladft.csv', help='Path to the CSV file containing MOSES id, CONFORMER id, and SMILES.')
    parser.add_argument('--xyz_dir_path', type=str, default='/home/megagatlingpea/workdir/Dataset/nabladft-conformer-mini', help='Path to the directory containing XYZ files to be converted.')
    parser.add_argument('--sdf_dir_path', type=str, default='/home/megagatlingpea/workdir/Delete-pretrain/data/cache', help='Path to the directory where output SDF files will be saved.')
    parser.add_argument('--range', type=str, default='1-10000', help='Range of molecules to be processed (e.g., 1-10000).')
    parser.add_argument('--conformer_id', type=int, default=0, help='Conformer ID to be used (default: 0)')

    args = parser.parse_args()

    # Call the function
    xyz2sdf(args.csv_path, args.xyz_dir_path, args.sdf_dir_path, args.range, args.conformer_id)
