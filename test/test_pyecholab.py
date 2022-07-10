# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 09:53:22 2022

@author: Tim21
"""

import os
import glob
import pandas as pd

#os.chdir('G:/git/pelAcousticsAI')
os.chdir('C:/git/pelacousticsAI')

from pyEcholab.echolab2.instruments import EK80, EK60
from pyEcholab.echolab2.plotting.matplotlib import echogram

# Detect FileType
def ek_detect(fname):
    with open(fname, 'rb') as f:
        file_header = f.read(8)
        file_magic = file_header[-4:]
        if file_magic.startswith(b'XML'):
            return "EK80"
        elif file_magic.startswith(b'CON'):
            return "EK60"
        else:
            return None

def ek_read(fname):
    ftype = ek_detect(fname)
    if ftype == "EK80":
        ek80_obj = EK80.EK80()
        
        ek80_obj.read_raw(fname)
        return ek80_obj
    elif ftype == "EK60":
        ek60_obj = EK60.EK60()
        ek60_obj.read_raw(fname)
        return ek60_obj

dat = pd.read_csv('CRIMAC-preprocessing/test/testset.csv')
root = 'G:/.shortcut-targets-by-id/1ZwauWhe7_s6mqxc-GcgzauBn-p9OGp1-/CRIMAC/test_data/'

for idxDataSet in [*range(0,6,1), *range(8,len(dat),1)]: # file 7 is an ME70 file
    raw_dir = root+dat.testdataset[idxDataSet]+dat.RAW_files[idxDataSet]
    print(dat.testdataset[idxDataSet])
    listFiles = glob.glob(raw_dir + '/*.raw')
    for idxFile in range(len(listFiles)):
        fname = listFiles[idxFile]
        ftype = ek_detect(fname)
        print(ftype+'-'+dat.RAW_type[idxDataSet])
        raw_obj = ek_read(fname)

