import xarray as xr
import numpy as np
import pandas as pd
import s3fs
import time
import traceback

# Reads files directly from S3
host = 'https://s3.hi.no'
access_key = 'crimac'
secret_key = '9!%L*h7Q'
bucketname = 'crimac-scratch'
region = 'us-east-1'

fs = s3fs.S3FileSystem(
    key=access_key,
    secret=secret_key,
    client_kwargs={
          'endpoint_url': host,
          'region_name': region})

fs.ls(bucketname)

SURVEY = ['S2019847_0511', 'S2007205', 'S2008205',
          'S2009107', 'S2010205', 'S2011206',
          'S2012837','S2013842','S2014807',
          'S2015837','S2016837','S2017843',
          'S2018823','S2019847']#,'S2020821',
          #'S2021847']

YEAR = ['2019', '2007', '2008',
        '2009', '2010', '2011',
        '2012','2013','2014',
        '2015','2016','2017',
        '2018', '2019'] #, '2020',
        #'2021']

for i, _SURVEY in enumerate(SURVEY):
    _YEAR = YEAR[i]
    #d = fs.ls('crimac-scratch/gpfs0-crimac-scratch/'+_YEAR+
    #          '/'+_SURVEY+'/ACOUSTIC/GRIDDED')
    uri = 's3://crimac-scratch/gpfs0-crimac-scratch/'+_YEAR+'/'+_SURVEY+'/ACOUSTIC/GRIDDED/'+_SURVEY+'_sv.zarr'
    print('-------------------------------------------')
    print(_SURVEY)
    print(uri)
        
    try: 
        # Access zarr file
        
        sv_file = s3fs.S3Map(uri, s3=fs)
        time.sleep(2)
        grid = xr.open_zarr(sv_file)
        time.sleep(2)
        t = pd.Series(grid.ping_time)
        dif = t.diff().astype(int) < 0
        dif[0] = False
        ind = np.where(dif)
        print('Successfully read data.')
        print('Number of negative diff time variable: '+str(len(ind)))
        print('Number of NaNs in time variable:       '+str(t.isnull().sum()))
        print('Number of NaNs in distance variable:   '+
              str(pd.Series(grid.distance).isnull().sum()))
    except:
        print('Failed to read data. Traceback:')
        traceback.print_exc()        
    print(' ')
        






