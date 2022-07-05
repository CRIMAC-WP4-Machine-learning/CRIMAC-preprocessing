import pandas as pd
import os
import docker
import xarray as xr
import zarr

os.chdir('/home/nilsolav/repos/CRIMAC-preprocessing/test/')
dat = pd.read_csv('testset.csv')
datain0 = '/mnt/c/DATAscratch/crimac-scratch/test_data/'
datawork0 = '/mnt/c/DATAscratch/crimac-scratch/test_data/'
dataout0 = '/mnt/c/DATAscratch/crimac-scratch/test_data/'
PREPROCESSOR_MAIN_FREQ = 38000
PREPROCESSOR_MAX_RANGE_SRC = 500 

# Loop over test set
for i, file in enumerate(dat['testdataset']):
    print(file)

    # Prepare docker
    datain = datain0+file+dat['RAW_files'][i]
    datawork = datain0+file+dat['WORK_files'][i]
    dataout = dataout0+file+dat['GRIDDED_files'][i]
    
    mount_list = {
        datain: {'bind': '/datain', 'mode': 'ro'},
        datawork: {'bind': '/workin', 'mode': 'ro'},
        dataout: {'bind': '/dataout', 'mode': 'rw'}
    }
    
    environments = [
        "OUTPUT_NAME=" + file,
        "OUTPUT_TYPE=zarr",
        "MAIN_FREQ=" + str(PREPROCESSOR_MAIN_FREQ),
        "MAX_RANGE_SRC=" + str(PREPROCESSOR_MAX_RANGE_SRC),
        "WRITE_PNG=1"
    ]
    
    image_tag = 'crimac/preprocessor'

    # Run the docker image
    try:
        client = docker.from_env()
        container = client.containers.run(image_tag,
                                          auto_remove = True,
                                          volumes=mount_list,
                                          environment=environments)

        print(file+' :OK')
        # Get "checksum"
        sv = xr.open_zarr(dataout+file+'.zarr')
        dat['sv_sum'][i] = str(sv.sv.sum(skipna=True).values)
        dat['status'][i] = 'OK'
    except:
        print(file+' :Failed')
        print(file+' :Failed')
        dat['sv_sum'][i] = 'NaN'
        dat['status'][i] = 'Fail'

       
dat.to_csv(path_or_buf=None, sep=',')

