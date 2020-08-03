# -*- coding: utf-8 -*-

# This script reads the matlabfiles and convert them to python pickle files
#
# Dependencies:
# Import libraries
# import os
# import numpy as np
# import scipy.io as spio
# import platform
# import pickle
# import scipy.ndimage
#
# Required input data files (example for the sand eel case):
# /dataout/


# /D:\DATA\LSSS-label-versioning\S2017838\ACOUSTIC\memmap
# Output data files:
# <scratchfolder>/data/DataOverview_North Sea NOR Sandeel cruise in Apr_May/memmap/<rawfilename>/
# Each folder contains these files
# Memmap files:
# data_for_freq_18.dat 
# data_for_freq_200.dat
# data_for_freq_333.dat
# data_for_freq_38.dat
# Pickle files:
# labels.dat             Label file
# data_dtype.pkl 
# frequencies.pkl        List of requencies
# label_dtype.pkl        List of labels
# objects.pkl
# range_vector.pkl
# shape.pkl
# time_vector.pkl

# (C) COGMAR. Anders, Nils Olav and Olav

#%% Import libraries
import os
import numpy as np
import scipy.io as spio
import pickle
import scipy.ndimage
import json
import ntpath
import pdb

##Parameters
overwrite = False
data_dtype = 'float32'
label_dtype = 'int16' #  Max ac category is less than 10000. Int16 covers \pm 32767

#%% Set local environment variables
if os.path.isfile('setpyenv.json'):
    # Read the environment from the json file
    json_data=open('setpyenv.json').read()    
    env = json.loads(json_data)
else:
    # Write the file at first run
    env = {}
    env["scratch"] = "/nr/project/bild/Cogmar/data/akustikk_all/data" #"/nr/project/bild/Cogmar/data/akustikk_all/data/DataOverview_North Sea NOR Sandeel cruise in Apr_May/"
    with open('setenv.json', 'w') as outfile:
        json.dump(env, outfile)

print(env)

#%%Function for saving memmory maps, meta data
def save_memmap(data, path, dtype, overwrite = False):
    path = (path + '.dat').replace('.dat.dat','.dat')
    head, tail = ntpath.split(path)
    if os.path.isfile(path) and not overwrite:
        print(' - File already exist', path.strip(path_to_mat_files))
    else:
        print(' - Saving', path.strip(path_to_mat_files))
        fp = np.memmap(path, dtype=dtype, mode='w+', shape=data.shape)
        fp[:] = data.astype(dtype)
        del fp

def save_pickle(data, name, out_folder):
    with open(os.path.join(out_folder,name+'.pkl'), 'wb') as f:
        pickle.dump(data, f)

#%% The sand eel case (will be expanded to other surveys later)
path_to_mat_files = env["scratch"] + 'North Sea NOR Sandeel cruise in Apr_May'

## Make memmap folder for this survey
if not os.path.isdir(os.path.join(path_to_mat_files, 'memmap')):
    os.makedirs(os.path.join(path_to_mat_files, 'memmap'))

##Loop through matlab files in the different years and save to numpy memory maps
year_dirs = [x for x in os.listdir(path_to_mat_files) if x not in {'Thumbs.db','memmap'}]

for year_dir in year_dirs:
    filedir = os.path.join(path_to_mat_files,year_dir)
    for file in os.listdir(filedir):
        if file.endswith(".mat") and  'datastatus' not in file:
            #Make file names
            filename, file_extension = os.path.splitext(file)
            out_folder = os.path.join(path_to_mat_files, 'memmap', filename)
            in_file = os.path.join(filedir, filename) + '.mat'
            print(filename)
            #Make folder
            if not os.path.isdir(out_folder):
                os.makedirs(out_folder)
            #Load file
            try:
                mat = spio.loadmat(in_file)
                '''pdb.set_trace()
                nils = mat['sv'].flatten()
                plt.hist(10*np.log10(nils))
                plt.show()'''
            except:
                print('Could not open', in_file)
                continue
            #Save output
            for i,f in enumerate(mat['F'].squeeze()):
                save_memmap( mat['sv'][:,:,i], os.path.join(out_folder, 'data_for_freq_' + str(int(f))), data_dtype, overwrite)

            save_memmap( mat['I'], os.path.join(out_folder, 'labels'), label_dtype, overwrite)
            #save_memmap( mat['depths'], os.path.join(out_folder, 'depths'), label_dtype, overwrite)
            #Save meta data
            save_pickle(mat['F'], 'frequencies', out_folder)
            save_pickle(mat['range'], 'range_vector', out_folder)
            save_pickle(mat['t'], 'time_vector', out_folder)
            save_pickle(data_dtype, 'data_dtype', out_folder)
            save_pickle(label_dtype, 'label_dtype', out_folder)
            save_pickle(mat['sv'][:, :, 0].shape, 'shape', out_folder)
            save_pickle(mat['depths'], 'depths', out_folder)
            save_pickle(mat['heave'], 'heave', out_folder)
            # Make list of objects
            objects = []

            indexes = np.indices(mat['I'].shape).transpose([1,2,0])
            for fish_type_ind in np.unique(mat['I']):
                if fish_type_ind != 0:

                    #Do connected components analysis
                    labeled_img, n_components = scipy.ndimage.label(mat['I']==fish_type_ind)

                    #Loop through components
                    for i in range(1,n_components+1):
                        object = {}

                        #Collect indexes for component
                        indexes_for_components = indexes[labeled_img==i]

                        #Collect data + metadata
                        object['fish_type_index'] = fish_type_ind
                        object['indexes'] = indexes_for_components
                        object['n_pixels'] = indexes_for_components.shape[0]
                        object['bounding_box'] = [np.min(indexes_for_components[:,0]), np.max(indexes_for_components[:,0]), np.min(indexes_for_components[:,1]), np.max(indexes_for_components[:,1]) ]
                        area_of_bounding_box = (object['bounding_box'][1]-object['bounding_box'][0]+1) * (object['bounding_box'][3]-object['bounding_box'][2]+1)
                        object['labeled_as_segmentation'] = area_of_bounding_box != object['n_pixels']

                        objects.append(object)

            save_pickle(objects, 'objects', out_folder)
            #save_pickle(len(objects), 'n_objects', out_folder)

            print(' -', str(len(objects)), 'objects found')


