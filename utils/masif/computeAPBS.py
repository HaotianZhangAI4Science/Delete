import os
import numpy as np
import subprocess
import pymesh
import tempfile

#from deepdock.default_config.global_vars import apbs_bin, pdb2pqr_bin, multivalue_bin
import random

""" 
Modified from:
computeAPBS.py: Wrapper function to compute the Poisson Boltzmann electrostatics for a surface using APBS.
Pablo Gainza - LPDI STI EPFL 2019
"""

def computeAPBS(vertices, pdb_file, apbs_bin, pdb2pqr_bin, multivalue_bin, workdir="."):
    """
        Calls APBS, pdb2pqr, and multivalue and returns the charges per vertex
    """    
    cmd = pdb2pqr_bin + " --ff=PARSE --whitespace --noopt --apbs-input %s temp1"%(pdb_file)
    p = subprocess.Popen([cmd], shell=True, cwd=workdir)
    p.wait()
    
    cmd = apbs_bin + " temp1.in"
    p = subprocess.Popen([cmd], shell=True, cwd=workdir)
    p.wait()
    
    vertfile = open("%s/temp1.csv"%workdir, "w")
    for vert in vertices:
        vertfile.write("{},{},{}\n".format(vert[0], vert[1], vert[2]))
    vertfile.close()
    
    cmd = multivalue_bin + " temp1.csv temp1.dx temp1_out.csv"
    p = subprocess.Popen([cmd], shell=True, cwd=workdir)
    p.wait()
    
    # Read the charge file
    chargefile = open("%s/temp1_out.csv"%workdir)
    charges = np.array([0.0] * len(vertices))
    for ix, line in enumerate(chargefile.readlines()):
        charges[ix] = float(line.split(",")[3])
    #os.system("rm " + tmp_file_base + "*")
    #os.system("rm io.mc")
    
    return charges



"""  ORIGINAL FUNCTION
'''
computeAPBS.py: Wrapper function to compute the Poisson Boltzmann electrostatics for a surface using APBS.
Pablo Gainza - LPDI STI EPFL 2019
This file is part of MaSIF.
Released under an Apache License 2.0
'''

def computeAPBS(vertices, pdb_file, tmp_file_base = tempfile.mktemp()):
    
        #Calls APBS, pdb2pqr, and multivalue and returns the charges per vertex
    
    #fields = tmp_file_base.split("/")[0:-1]
    #directory = "/".join(fields) + "/"
    fields = tmp_file_base
    directory = str(fields) + "/"
    filename_base = tmp_file_base.split("/")[-1]
    pdbname = pdb_file.split("/")[-1]
    args = [
        pdb2pqr_bin,
        "--ff=parse",
        "--whitespace",
        "--noopt",
        "--apbs-input",
        pdbname,
        filename_base,
    ]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE, cwd=directory)
    stdout, stderr = p2.communicate()

    args = [apbs_bin, filename_base + ".in"]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE, cwd=directory)
    stdout, stderr = p2.communicate()

    vertfile = open(directory + "/" + filename_base + ".csv", "w")
    for vert in vertices:
        vertfile.write("{},{},{}\n".format(vert[0], vert[1], vert[2]))
    vertfile.close()

    args = [
        multivalue_bin,
        filename_base + ".csv",
        filename_base + ".dx",
        filename_base + "_out.csv",
    ]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE, cwd=directory)
    stdout, stderr = p2.communicate()

    # Read the charge file
    chargefile = open(tmp_file_base + "_out.csv")
    charges = numpy.array([0.0] * len(vertices))
    for ix, line in enumerate(chargefile.readlines()):
        charges[ix] = float(line.split(",")[3])

    return charges
"""
