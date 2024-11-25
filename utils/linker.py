# This is designed for the UniMG protocol, let's finish it
import os
import random
import re
import math
import csv
import sys
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdMMPA
from rdkit import DataStructs
from rdkit.Chem import Descriptors, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdMolAlign
from rdkit.Chem import MolStandardize

import matplotlib.pyplot as plt

#import sascorer
from itertools import chain, product

from joblib import Parallel, delayed

#import calc_SC_RDKit

def fragment_mol(smi, cid, pattern="[#6+0;!$(*=,#[!#6])]!@!=!#[*]", design_task="linker"):
    '''
    smi: is the smiles you should pass in
    pattern: is the cutted pattern 
    cid could be ignored 
    '''
    mol = Chem.MolFromSmiles(smi)

    #different cuts can give the same fragments
    #to use outlines to remove them
    outlines = set()

    if (mol == None):
        sys.stderr.write("Can't generate mol for: %s\n" % (smi))
    else:
        if design_task == "linker":
	        frags = rdMMPA.FragmentMol(mol, minCuts=2, maxCuts=2, maxCutBonds=100, pattern=pattern, resultsAsMols=False)
        elif design_task == "elaboration":
	        frags = rdMMPA.FragmentMol(mol, minCuts=1, maxCuts=1, maxCutBonds=100, pattern=pattern, resultsAsMols=False)
        else:
            print("Invalid choice for design_task. Must be 'linker' or 'elaboration'.")
        for core, chains in frags:
            if design_task == "linker":
                output = '%s,%s,%s,%s' % (smi, cid, core, chains)
            elif design_task == "elaboration":
                output = '%s,%s,%s' % (smi, cid, chains)
            if (not (output in outlines)):
                outlines.add(output)
        if not outlines:
            # for molecules with no cuts, output the parent molecule itself
            outlines.add('%s,%s,,' % (smi,cid))

    return outlines

