import time
from timeit import timeit
from scipy import interpolate
import zarr
import os
import pyarrow as pa
import pandas as pd
import xarray as xr
import math
import numpy as np
from numcodecs import Blosc
import pyarrow.parquet as pq
import dask.array as dask
import datetime
import csv
import sys
import subprocess
import os.path
from os import path
import pyarrow.dataset as ds
 
readfile=""
if("-readfile" in  sys.argv):
    readfile = str(sys.argv[sys.argv.index("-readfile") + 1])

def fill_nan(A):
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    #print(inds)
    #print(good)
    f = interpolate.interp1d(inds[good], A[good], bounds_error=False, fill_value='extrapolate')
    B = np.where(np.isfinite(A), A, f(inds))
    return B
def isNaN(num):
    return num != num

def correct_parquet(parquetfile):
    table1 = pq.read_table( parquetfile)
    t1 = table1.to_pandas()['distance']
    t2 = table1.to_pandas()['raw_file']
    t3 = table1.to_pandas()['ping_time2']

    distdif=0.0010
    ferror = open(str(parquetfile) + '_error.csv', 'w' )
    writererror = csv.writer(ferror)
    dataerror = []
    dataerror.append("index")
    dataerror.append("ping1")
    dataerror.append("ping2")
    dataerror.append("dist1")
    dataerror.append("dist2")
    dataerror.append("file")
    dataerror.append("type")
    writererror.writerow(dataerror)
    print(parquetfile)
    lastping=0
    lastdist=0
    lastfile=""
    index=0
    distcount=0
    pingcount=0
    distnancount=0
    pingnancount=0
    cc=0
    printping=0
    printlastdist=0
    jumpval=0
    #t1.info()
    arr = []
    pings = []
    add10k=False
    for row in t1:
        newdist=row
        newping=t3[index]
        epoch =newping.value
        newping = epoch
        saveping = epoch
        row=row+jumpval
        newdist = row
        if add10k:
            row=row+10000
            newdist = row
        if newdist < (lastdist-9000):
            add10k = True
            dataerror = []
            dataerror.append(index)
            dataerror.append(pd.to_datetime(lastping, origin='unix'))
            dataerror.append(pd.to_datetime(newping, origin='unix'))
            dataerror.append(lastdist)
            dataerror.append(newdist)
            dataerror.append(t2[index])
            dataerror.append("dist-reset")
            writererror.writerow(dataerror)
            row = row + 10000
            newdist = row

        if newdist > lastdist+49000:
            row = float("NaN")
            newdist=lastdist
            dataerror = []
            dataerror.append(index)
            dataerror.append(pd.to_datetime(lastping, origin='unix'))
            dataerror.append(pd.to_datetime(newping, origin='unix'))
            dataerror.append(lastdist)
            dataerror.append(newdist)
            dataerror.append(t2[index])
            dataerror.append("dist-reset-peak")
            writererror.writerow(dataerror)

        if newdist == lastdist:
            #row = float("NaN")
            row=row +distdif
            newdist = row
            jumpval=jumpval+distdif
            dataerror = []
            dataerror.append(index)
            dataerror.append(pd.to_datetime(lastping, origin='unix'))
            dataerror.append(pd.to_datetime(newping, origin='unix'))
            dataerror.append(lastdist)
            dataerror.append(newdist)
            dataerror.append(t2[index])
            dataerror.append("d_0")
            writererror.writerow(dataerror)
            distcount = distcount + 1
        if newdist < lastdist:
            #row = float("NaN")
            jumpadd= lastdist - newdist
            row=row +distdif +jumpadd
            newdist = row
            jumpval=jumpval+distdif+jumpadd
            dataerror = []
            dataerror.append(index)
            dataerror.append(pd.to_datetime(lastping, origin='unix'))
            dataerror.append(pd.to_datetime(newping, origin='unix'))
            dataerror.append(lastdist)
            dataerror.append(newdist)
            dataerror.append(t2[index])
            dataerror.append("d_neg")
            writererror.writerow(dataerror)
            distcount = distcount + 1

        if isNaN(newdist):
            dataerror = []
            dataerror.append(index)
            dataerror.append(pd.to_datetime(lastping, origin='unix'))
            dataerror.append(pd.to_datetime(newping, origin='unix'))
            dataerror.append(lastdist)
            dataerror.append(newdist)
            dataerror.append(t2[index])
            dataerror.append("dn")
            writererror.writerow(dataerror)
            distnancount = distnancount + 1
        else:
            if newdist > lastdist:
                printlastdist=lastdist
                lastdist = newdist
                lastfile=t2[index]
        arr.insert(len(arr), row)

        if index>0:
            if newping < lastping:
                saveping = float("NaN")

                dataerror = []
                dataerror.append(index)
                dataerror.append(pd.to_datetime(lastping, origin='unix'))
                dataerror.append(pd.to_datetime(newping, origin='unix'))
                dataerror.append(printlastdist)
                dataerror.append(newdist)
                dataerror.append(t2[index])
                dataerror.append("p")
                writererror.writerow(dataerror)
                pingcount = pingcount + 1
        if isNaN(newping):
            pingnancount = pingnancount + 1
            dataerror = []
            dataerror.append(index)
            dataerror.append(pd.to_datetime(lastping, origin='unix'))
            dataerror.append(pd.to_datetime(newping, origin='unix'))
            dataerror.append(printlastdist)
            dataerror.append(newdist)
            dataerror.append(t2[index])
            dataerror.append("pn")
            writererror.writerow(dataerror)

        else:
            if index>0:
                if newping > lastping:
                    lastping = newping
            else:
                lastping = newping
        pings.insert(len(pings), saveping)

        index=index+1
    print("dist "+str(len(arr))+" : "+str(distcount)+" "+ str(distnancount) )
    print("ping "+str(len(pings)) + " : " + str(pingcount) + " " + str(pingnancount))

    distcount = 0
    pingcount = 0
    distnancount = 0
    pingnancount = 0
    lastdist = 0
    index = 0
    arr2 = np.array(arr)

    for x in arr2:
        newdist=x
        if newdist <= lastdist:
            distcount = distcount + 1
            #print(str(lastdist)+" "+str(newdist))
        if isNaN(newdist):
            distnancount = distnancount + 1
        else :
            if newdist > lastdist:
                lastdist = newdist
        index = index + 1
    print("dist "+str(len(arr)) + " : " + str(distcount) + " " + str(distnancount) )
    arr2 = fill_nan(arr2)

    index = 0
    pings2 = np.array(pings)
    for x in pings2:
        newping=x
        if index > 0:
            if newping <= lastping:
                pingcount = pingcount + 1
        if isNaN(newping):
            pingnancount = pingnancount + 1
        else :
            if index > 0:
                if newping > lastping:
                    lastping = newping
            else:
                lastping = newping
        index = index + 1
    print("ping "+str(len(pings2)) + " : " + str(pingcount) + " " + str(pingnancount) )
    pings2 = fill_nan(pings2)

    index = 0
    distcount = 0
    pingcount = 0
    distnancount = 0
    pingnancount = 0
    lastdist = 0
    for x in arr2:
        newdist = x
        if newdist <= lastdist:
            distcount = distcount + 1
        if isNaN(newdist):
            distnancount = distnancount + 1
        else:
            if newdist > lastdist:
                lastdist = newdist
        index = index + 1
    print("dist "+str(len(arr)) + " : " + str(distcount) + " " + str(distnancount))
    index = 0

    for x in pings2:
        newping = pd.to_datetime(x, origin='unix')
        if index > 0:
            if newping <= lastping:
                pingcount = pingcount + 1
        if isNaN(newping):
            pingnancount = pingnancount + 1
        else:
            if index > 0:
                if newping > lastping:
                    lastping = newping
            else:
                lastping = newping
        index = index + 1
    print("ping "+str(len(pings2)) + " : " + str(pingcount) + " " + str(pingnancount))
    print("----------")


    #savedir = parquetfile.replace("_pingdist.parquet", "correction/")
    #os.system("mkdir " + savedir)
    arr3= []
    pings3 = []
    cc = 0
    lastfile=""
    savefile=parquetfile.replace(".parquet", "corrected.parquet")
    fields = [ 
                pa.field('raw_file', pa.string()),
                pa.field('distance', pa.float64()),
                pa.field('ping_time', pa.timestamp('ns')),
                ]
    df_schema = pa.schema(fields)
    #parquet_format = ds.ParquetFileFormat()
    #file_options = parquet_format.make_write_options(coerce_timestamps='us', allow_truncated_timestamps=True)
    pq_obj = None
    for x in arr2:
        p=pd.to_datetime(pings2[cc], origin='unix')
        #print(p)
        #p=p.round('ms')
        p=p.round('ms')
        #print(p.round('ns'))
        #print(type(p))
        if t2[cc]!=lastfile:

            if len(arr3)>0:
                rawname = np.full(len(arr3), lastfile)
                distcorrected = np.array(arr3)
                arr3 = []
                pingcorrected = np.array(pings3)
                pings3= []
                #arrow_array1 = pa.array(rawname)
                #arrow_array2 = pa.array(distcorrected)
                #arrow_array3 = pa.array(pingcorrected)     
                
                df = pd.DataFrame({'raw_file':rawname,'distance':distcorrected, 'ping_time': pingcorrected})
                
                #print(df)
                pa_tbl = pa.Table.from_pandas(df, schema=df_schema, preserve_index=False)
                if pq_obj == None:
                    pq_obj = pq.ParquetWriter(savefile, pa_tbl.schema,coerce_timestamps='us', allow_truncated_timestamps=True)
                pq_obj.write_table(table=pa_tbl )
        
                #print(savefile)
                
            lastfile=t2[cc]
        arr3.insert(len(arr3), x)
        pings3.insert(len(pings3), p)
        cc = cc+1
        	
    rawname = np.full(len(arr3), lastfile)
    distcorrected = np.array(arr3)
    arr3 = []
    pingcorrected = np.array(pings3)
    pings3= []
    df = pd.DataFrame({'raw_file':rawname,'distance':distcorrected, 'ping_time': pingcorrected})
    #print(df)
    pa_tbl = pa.Table.from_pandas(df, schema=df_schema, preserve_index=False)
    if pq_obj == None:
        pq_obj = pq.ParquetWriter(savefile, pa_tbl.schema)
    pq_obj.write_table(table=pa_tbl)
    print(savefile)
    ferror.close()


correct_parquet(readfile)
