# -*- coding: utf-8 -*-
"""
Reads raw file and convert it into a grid-ed format.
Right now it can handle different range sizes between channels.
Has the ability to save the resulting grid into netcdf or zarr.
"""


from echolab2.instruments import EK80, EK60
from echolab2.plotting.matplotlib import echogram

import numpy as np
import xarray as xr
import os.path

from matplotlib import pyplot as plt, colors
from matplotlib.colors import LinearSegmentedColormap, Colormap
from numcodecs import Blosc


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
    if ek_detect(fname) == "EK80":
        ek80_obj = EK80.EK80()
        ek80_obj.read_raw(fname)
        return ek80_obj
    elif ek_detect(fname) == "EK60":
        ek60_obj = EK60.EK60()
        ek60_obj.read_raw(fname)
        return ek60_obj

# Simple plot function
def plot_sv(da, minmax, axes, idx):
    # Prepare simrad cmap
    simrad_color_table = [(1, 1, 1),
                                        (0.6235, 0.6235, 0.6235),
                                        (0.3725, 0.3725, 0.3725),
                                        (0, 0, 1),
                                        (0, 0, 0.5),
                                        (0, 0.7490, 0),
                                        (0, 0.5, 0),
                                        (1, 1, 0),
                                        (1, 0.5, 0),
                                        (1, 0, 0.7490),
                                        (1, 0, 0),
                                        (0.6509, 0.3255, 0.2353),
                                        (0.4705, 0.2353, 0.1568)]
    simrad_cmap = (LinearSegmentedColormap.from_list
                                ('Simrad', simrad_color_table))
    simrad_cmap.set_bad(color='grey')

    # Use 0.5 bins
    x_bins = np.arange(da.range.min().round(), da.range.max().round(), 0.5)
    # Plot using lower resolution
    da_1 = da.groupby_bins("range", x_bins).mean()
    da_1.plot(norm=colors.LogNorm(vmin = minmax[0], vmax = minmax[1]), cmap=simrad_cmap, ax=axes[idx])

def process_data_to_xr(raw_data):
    # Get calibration object
    cal_obj = raw_data.get_calibration()
    # Get sv values
    sv_obj = raw_data.get_sv(calibration = cal_obj)
    # Get sv as depth
    sv_obj_as_depth = raw_data.get_sv(calibration = cal_obj,
        return_depth=True)

    # Get frequency label
    freq = sv_obj.frequency

    # Expand sv values into a 3d object
    data3d = np.expand_dims(sv_obj.data, axis=0)

    # This is the sv data in 3d    
    sv = xr.DataArray(name="sv", data=data3d, dims=['frequency', 'ping_time', 'range'],
                           coords={ 'frequency': [freq],
                                    'ping_time': sv_obj.ping_time,
                                    'range': sv_obj.range,
                                   })
    # This is the depth data
    depth = xr.DataArray(name="depth", data=np.expand_dims(sv_obj_as_depth.depth, axis=0), dims=['frequency', 'range'],
                           coords={ 'frequency': [freq],
                                    'range': sv_obj.range,
                                   })
    return [sv, depth]

def resampleWeight(r_t, r_s):
    """
    The regridding is a linear combination of the inputs based
    on the fraction of the source bins to the range bins.
    See the different cases below
    """

    # Create target bins from target range
    bin_r_t = np.append(r_t[0]-(r_t[1] - r_t[0])/2, (r_t[0:-1] + r_t[1:])/2)
    bin_r_t = np.append(bin_r_t, r_t[-1]+(r_t[-1] - r_t[-2])/2)

    # Create source bins from source range
    bin_r_s = np.append(r_s[0]-(r_s[1] - r_s[0])/2, (r_s[0:-1] + r_s[1:])/2)
    bin_r_s = np.append(bin_r_s, r_s[-1]+(r_s[-1] - r_s[-2])/2)

    # Initialize W matrix (sparse)
    W = np.zeros([len(r_t), len(r_s)+1])
    # NB: + 1 length for space to NaNs in edge case

    # Loop over the target bins
    for i, rt in enumerate(r_t):
    
        # Check that this is not an edge case
        if bin_r_t[i] > bin_r_s[0] and bin_r_t[i+1] < bin_r_s[-1]:
            # The size of the target bin
            # example target bin:  --[---[---[---[-
            drt = bin_r_t[i+1] - bin_r_t[i]  # From example: drt = 4
        
            # find the indices in source
            j0 = np.searchsorted(bin_r_s, bin_r_t[i], side='right')-1
            j1 = np.searchsorted(bin_r_s, bin_r_t[i+1], side='right')
        
            # CASE 1: Target higher resolution, overlapping 1 source bin
            # target idx     i    i+1
            # target    -----[-----[-----
            # source    --[-----------[--
            # source idx  j0          j1
        
            if j1-j0 == 1:
                W[i, j0] = 1
                
            # CASE 2: Target higher resolution, overlapping 1 source bin
            # target idx      i   i+1
            # target    --[---[---[---[-
            # source    -[------[------[-
            # source idx j0            j1
        
            elif j1-j0 == 2:
                W[i, j0] = (bin_r_s[j0+1]-bin_r_t[i])/drt
                W[i, j1-1] = (bin_r_t[i+1]-bin_r_s[j1-1])/drt
                
            # CASE 3: Target lower resolution
            # target idx    i       i+1
            # target    ----[-------[----
            # source    --[---[---[---[--
            # source idx  j0          j1
                
            elif j1-j0 > 2:
                for j in range(j0, j1):
                    if j == j0:
                        W[i, j] = (bin_r_s[j+1]-bin_r_t[i])/drt
                    elif j == j1-1:
                        W[i, j] = (bin_r_t[i+1]-bin_r_s[j])/drt
                    else:
                        W[i, j] = (bin_r_s[j+1]-bin_r_s[j])/drt
                    
        #  Edge case 1
        # target idx    i       i+1
        # target    ----[-------[----
        # source        #end# [---[---[
        # source idx          j0  j1
                
        #  Edge case 2
        # target idx    i       i+1
        # target    ----[-------[----
        # source    --[---[ #end#
        # source idx  j0  j1
        else:
            # Edge case (NaN must be in W, not in sv_s.
            # Or else np.dot failed)
            W[i, -1] = np.nan
    return W

def regrid(sv_s, W, n_pings):
    """
    Use the weights to regrid the sv data
    """
    # Add a row of at the bottom to be used in edge cases
    sv_s_mod = np.vstack((sv_s, np.zeros(n_pings)))
    # Do the dot product
    return np.dot(W, sv_s_mod)

def raw_to_grid_single(raw_fname, main_frequency = 38000, write_output = False, out_fname = "", output_type = "zarr", overwrite = False):

    # Prepare out_fname
    if out_fname == "":
        out_fname = raw_fname

    # Read input raw
    raw_obj = ek_read(raw_fname)
    print(raw_obj)

    # Get all channels
    all_channels = list(raw_obj.raw_data.keys())

    main_channel = all_channels.copy()

    # Get real frequency channel (for EK80 - FM)
    main_raw_data = raw_obj.get_channel_data(main_frequency)[main_frequency][0]

    # Placeholder for all frequrncy
    all_frequency = []

    # Get the other channels
    other_channels = []
    for chan in all_channels:
        # Getting raw data for a frequency
        raw_data = raw_obj.raw_data[chan][0]
        tmp = raw_data.get_frequency(unique = True)
        all_frequency.append(*tmp)
        if(main_raw_data.get_frequency(unique = True) != tmp):
            other_channels.append(chan)
            main_channel.remove(chan)

    # Handle similar frequency below
    other_channels = other_channels + main_channel[1:]
    main_channel = [main_channel[0]]

    print("Main frequency: " + str(main_frequency))
    print("Main channel: " + str(main_channel))
    print("Other channels: " + str(other_channels))

    # TODO : Handle not found frequency

    # Getting raw data for  frequency
    raw_data_main = raw_obj.raw_data[main_channel[0]][0]
    [sv_main, depth_main] = process_data_to_xr(raw_data_main)
    da_sv = sv_main
    da_depth = depth_main

    # Process channels with same frequency (TODO)
    #for chan in all_frequency:
        
    # Process channels with different ping times (TODO)
    for chan in other_channels:
        # Getting raw data for a frequency
        raw_data = raw_obj.raw_data[chan][0]
        # Check if the time scales is the same
        if(np.array_equal(raw_data.ping_time, raw_data_main.ping_time) == False):
            print("Time mismatch with the main channel, use ping_match()")

    # Process Sv
    for chan in other_channels:
        # Getting raw data for a frequency
        raw_data = raw_obj.raw_data[chan][0]
        # Process it into xarray
        [sv, depth] = process_data_to_xr(raw_data)

        if(sv_main.range.equals(sv.range) == False):
            print(str(chan) + " range mismatch with main channel: Main range size: " + str(sv_main.range.size) + " Sub range size: " + str(sv.range.size))
            # Re-grid sv
            sv_obj = sv[0,]
            W = resampleWeight(sv_main.range.values, sv_obj.range)
            sv_tmp = regrid(sv_obj.data.transpose(), W, sv_obj.ping_time.size).transpose()
            # Create new xarray with this frequency
            sv = xr.DataArray(name="sv", data=np.expand_dims(sv_tmp, axis = 0), dims=['frequency', 'ping_time', 'range'],
                            coords={ 'frequency': [sv_obj.frequency],
                                    'ping_time': sv_main.ping_time,
                                    'range': sv_main.range,
                                    })
            # The depth data is simply using the main channel
            depth = xr.DataArray(name="depth", data=depth_main, dims=['frequency', 'range'],
                            coords={ 'frequency': [sv_obj.frequency],
                                    'range': sv_main.range,
                                    })
        # Combine different frequencies
        da_sv = xr.concat([da_sv, sv], dim='frequency')
        da_depth = xr.concat([da_depth, depth], dim='frequency')

    # Getting motion data
    obj_heave = raw_obj.motion_data.heave
    obj_pitch = raw_obj.motion_data.pitch
    obj_roll = raw_obj.motion_data.roll
    obj_heading = raw_obj.motion_data.heading

    # Crate a dataset
    ds = xr.Dataset(
        data_vars=dict(
            sv=(["frequency", "ping_time", "range"], da_sv),
            depth = (["frequency", "range"], da_depth),
            heave=(["ping_time"], obj_heave),
            pitch=(["ping_time"], obj_pitch),
            roll=(["ping_time"], obj_roll),
            heading=(["ping_time"], obj_heading),
            ),
        coords=dict(
            frequency = da_sv.frequency,
            ping_time = da_sv.ping_time,
            range = da_sv.range,
            ),
        attrs=dict(description="Multi-frequency sv values from EK."),
    )
    print("Created dataset:")
    print(ds)

    if write_output == True:
        if output_type == "netcdf4":
            # Save into netcdf
            target_fname = out_fname + ".nc"
            is_exists = (os.path.isfile(target_fname) or os.path.isdir(target_fname))
            if (is_exists == True and overwrite == True) or is_exists == False:
                comp = dict(zlib=True, complevel=5)
                encoding = {var: comp for var in ds.data_vars}
                ds.to_netcdf(target_fname, mode="w", encoding=encoding)
            else:
                print("Output data exists. Not overwriting.")
        elif output_type == "zarr":
            # Save into zarr
            target_fname = out_fname + ".zarr"
            is_exists = os.path.isfile(target_fname) or os.path.isdir(target_fname)
            if (is_exists == True and overwrite == True) or is_exists == False:
                compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
                encoding = {var: {"compressor" : compressor} for var in ds.data_vars}
                ds.to_zarr(target_fname, mode="w", encoding=encoding)
            else:
                print("Output data exists. Not overwriting.")
        else:
            print("Output type is not supported")
    
    return ds




# EXAMPLE TO USE BELOW

# File collections
dir = os.path.expanduser("~/IMR/echo-test/hackathon/")

files = [   'CRIMAC_2020_EK80_CW_DemoFile_GOSars.raw',
            'CRIMAC_2020_EK80_FM_DemoFile_GOSars.raw',
            '2017843-D20170426-T115044.raw',
            'N1-D20201106-T015512.raw']

# Select a file
raw_fname = dir + files[0]
#raw_fname = dir + files[1]
#raw_fname = dir + files[2]
#raw_fname = dir + files[3]

ds = raw_to_grid_single(raw_fname, write_output = True, overwrite = True)

# Plot all channels
fig, axes = plt.subplots(nrows = ds.sv.frequency.size, constrained_layout = True)

for idx, fr in enumerate(ds.sv.frequency):
    plot_sv(ds.sv.sel(frequency = fr), [ds.sv.min(), ds.sv.max()], axes, idx)

# Show the plot
plt.show()