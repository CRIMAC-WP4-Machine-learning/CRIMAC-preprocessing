import pdb
import sys
import os
import numpy as np
from matplotlib.pyplot import figure, show, subplots_adjust, get_cmap

# Temporary:
# sys.path.append('/mnt/d/repos/Github/pyEcholab/')
sys.path.append('d:/repos/Github/pyEcholab/')
from echolab2.processing.batch_utils import FileAggregator as fa
# from echolab2.processing.align_pings import AlignPings
from echolab2.instruments.EK60 import EK60
from echolab2.instruments.EK80 import EK80
from echolab2.plotting.matplotlib.echogram import Echogram

# This script reads the ek60 and ek80 (CW) files and generates the netcdf, memmap og zarr files.

# The script interpolates the data into a common grid using the master frequency as the starting point.


# Dependencies:
# pyEcholab

main_frequency = 38000
plt = False

# Which range vector to use when interpolating into the common grid
# par['range_frequency'] = str(main_frequency)

# Survey data directory per year
# TEMPORARY for testing purposes
if sys.platform == 'linux':
    dd_data_in = '/datain/'
    dd_data_work = '/datawork/'
    dd_data_out = '/dataout/'
else:  # For testing purposes on PC
    dd_data_in = \
        'D:\DATA\LSSS-label-versioning\S2017838\ACOUSTIC\EK60\EK60_RAWDATA'
    dd_data_work = 'D:\DATA\LSSS-label-versioning\S2017838\ACOUSTIC\LSSS\WORK'
    dd_data_out = 'D:\DATA\LSSS-label-versioning\S2017838\ACOUSTIC\memmap'
print(dd_data_in)

# Get the file list
# raw0 = dir(fullfile(dd_data_in, '*.raw'))

file_bins = fa(dd_data_in, 10).file_bins
raw_files = file_bins[0]


# Process one file
def read_raw_file(raw_files):
    """
    This part parses one raw file
    """

    # Generate the raw object
    try:
        # Create an instance of the EK60 instrument.
        ek = EK60()
        type = 'EK60'
        # Create an instance of the EK80 instrument.
    except:
        ek = EK80()
        type = 'EK80'
        
    # Use the read_raw method to read in a data file.
    ek.read_raw(raw_files)
    # Get a dictionary of channels
    channel_main = ek.get_channel_data(frequencies=main_frequency)
    # Get the range vectors for the main_channel
                           
    # raw_dat = raw_main.get_raw_data()
    
    # Get a dictionary of all channels
    channels = ek.get_channel_data()
    # Loop over channels
    for key in channels:
        print(key)
        print(channels[key])
        # Assumes one echogram per channel:
        ch = channels[key][0]
        sv = ch.get_sv()

        # align pings to master frequency
        # see match_ping_example

        # regrid based on regrid.py

        # Use iambaim code to get it together
        
        # Sample thickness should not change within a file
        # NB: Test a file with variable number of pings. We do not want any interpolation!
        range = sv.range
        time = sv.ping_time
        data = sv.data
                
