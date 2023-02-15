
from echolab2.instruments import EK80, EK60

import sys
import subprocess
import re
import dask
import scipy.ndimage
import numpy as np
import xarray as xr
import zarr as zr
import os.path
import shutil
import glob
import ntpath
import datetime
import gc
import netCDF4

from scipy import interpolate
from psutil import virtual_memory


from annotationtools import readers



import pyarrow as pa
import pyarrow.parquet as pq


import math
from numcodecs import Blosc

import os
import sys


rawdir = ""
workdir = ""
pq_filepath = ""

if("-rawdir" in  sys.argv):
    rawdir = str(sys.argv[sys.argv.index("-rawdir") + 1])
if("-workdir" in  sys.argv):
    workdir = sys.argv[sys.argv.index("-workdir") + 1]
if("-savefile" in  sys.argv):
    pq_filepath = sys.argv[sys.argv.index("-savefile") + 1]

#python parseworkfiles.py  -rawdir /localscratch_hdd/tomasz/cruise/2019/EK60_RAWDATA/  -workdir /localscratch_hdd/tomasz/cruise/2019/WORK/ -savefile /localscratch_hdd/tomasz/cruise/scripts/S2019847_labels.parquet 


def append_to_parquet(df, pq_filepath, pq_obj=None):
    # Must set the schema to avoid mismatched schema errors
    fields = [
        pa.field('ping_index', pa.int64()),
        pa.field('ping_time', pa.timestamp('ns')),
        pa.field('mask_depth_upper', pa.float64()),
        pa.field('mask_depth_lower', pa.float64()),
        pa.field('priority', pa.int64()),
        pa.field('acoustic_category', pa.string()),
        pa.field('proportion', pa.float64()),
        pa.field('object_id', pa.string()),
        pa.field('channel_id', pa.string())
    ]
    df_schema = pa.schema(fields)
    pa_tbl = pa.Table.from_pandas(df, schema=df_schema, preserve_index=False)
    if pq_obj == None:
        pq_obj = pq.ParquetWriter(pq_filepath, pa_tbl.schema)
    pq_obj.write_table(table=pa_tbl)
    return pq_obj





pq_writer = None 
counter=0

parsefile=1
 
filenames = os.listdir(rawdir)
 
sorted_filenames = sorted(filenames)

for filename in sorted_filenames:
    raw_fname=os.path.join(rawdir, filename)
    if raw_fname.endswith(".raw"):
        if(parsefile==1):
            
            work_fname =os.path.join(workdir, filename.replace("raw","work"))
            
            exists_work = os.path.isfile(work_fname)
            if exists_work:
                idx_fname =   raw_fname.replace("raw","idx")
                exists_idx = os.path.isfile(idx_fname)
                if exists_idx:
                    ann_obj = None
                    try:
                        work = readers.work_reader( work_fname )
                        ann_obj = readers.work_to_annotation(work, idx_fname)
                    except Exception as e:
                        exception_type, exception_object, exception_traceback = sys.exc_info()
                        filename = exception_traceback.tb_frame.f_code.co_filename
                        line_number = exception_traceback.tb_lineno
                        #print("Exception type: ", exception_type)
                        #print("File name: ", filename)
                        #print("Line number: ", line_number)
                        #print("ERROR3: - Something went wrong when reading the WORK file: " + str(work_fname) + " (" + str( e) + ")")
                        
                    if ann_obj is not None and ann_obj.df_ is not None:
                        df = ann_obj.df_
                        pq_writer = append_to_parquet(df, pq_filepath, pq_writer)
                    print (raw_fname)
                    print (counter)
                    counter=counter+1

pq_writer.close()
 
# Read the Parquet file
table = pq.read_table(pq_filepath)
df= table.to_pandas()
print (df)


