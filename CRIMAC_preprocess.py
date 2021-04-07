# -*- coding: utf-8 -*-
"""
CRIMAC Master Preprocessing Script

Reads EK60/EK8 raw files and convert it into a grid-ed format. Currently it
can handle different range sizes between channels data. This script
also has the ability to save the resulting grid into NetCDF or ZARR formatted
files.

Copyright (C) 2020, Ibrahim Umar, Nils Olav Handegard, Alba Ordonez, Rune
Øyerhamn, and The Institute of Marine Research, Norway.

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program; if not, write to the Free Software Foundation,
Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""


from echolab2.instruments import EK80, EK60

import dask
import numpy as np
import xarray as xr
import os.path
import shutil
import glob
import ntpath
import datetime
import netCDF4 

from annotationtools import readers

import pyarrow as pa
import pyarrow.parquet as pq

from matplotlib import pyplot as plt, colors
from matplotlib.colors import LinearSegmentedColormap, Colormap
import math
from numcodecs import Blosc

def append_to_parquet(df, pq_filepath, pq_obj=None):
    pa_tbl = pa.Table.from_pandas(df)
    if pq_obj == None:
        pq_obj = pq.ParquetWriter(pq_filepath, pa_tbl.schema)
    pq_obj.write_table(table=pa_tbl)
    return pq_obj

# From https://github.com/pydata/xarray/issues/1672#issuecomment-685222909
def _expand_variable(nc_variable, data, expanding_dim, nc_shape, added_size):
    # For time deltas, we must ensure that we use the same encoding as
    # what was previously stored.
    # We likely need to do this as well for variables that had custom
    # econdings too
    if hasattr(nc_variable, 'calendar'):
        
        data.encoding = {
            'units': nc_variable.units,
            'calendar': nc_variable.calendar,
        }
    data_encoded = xr.conventions.encode_cf_variable(data) # , name=name)
    left_slices = data.dims.index(expanding_dim)
    right_slices = data.ndim - left_slices - 1
    nc_slice   = (slice(None),) * left_slices + (slice(nc_shape, nc_shape + added_size),) + (slice(None),) * (right_slices)
    nc_variable[nc_slice] = data_encoded.data

def append_to_netcdf(filename, ds_to_append, unlimited_dims):
    if isinstance(unlimited_dims, str):
        unlimited_dims = [unlimited_dims]
        
    if len(unlimited_dims) != 1:
        # TODO: change this so it can support multiple expanding dims
        raise ValueError(
            "We only support one unlimited dim for now, "
            f"got {len(unlimited_dims)}.")

    unlimited_dims = list(set(unlimited_dims))
    expanding_dim = unlimited_dims[0]
    
    with netCDF4.Dataset(filename, mode='a') as nc:
        nc_dims = set(nc.dimensions.keys())

        nc_coord = nc[expanding_dim]
        nc_shape = len(nc_coord)
        
        added_size = len(ds_to_append[expanding_dim])
        variables, attrs = xr.conventions.encode_dataset_coordinates(ds_to_append)

        for name, data in variables.items():
            if expanding_dim not in data.dims:
                # Nothing to do, data assumed to the identical
                continue

            nc_variable = nc[name]
            _expand_variable(nc_variable, data, expanding_dim, nc_shape, added_size)

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

# Simple plot function
def plot_all(ds, out_name, range_res = 600, time_res = 800, interpolate = False):
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

    sv = ds.sv

    range_len = len(ds.sv.range)
    time_len = len(ds.sv.ping_time)

    if range_len > range_res or time_len > time_res:
        mult_range = math.floor(range_len/range_res)
        mult_time = math.floor(time_len/time_res)

        if mult_range == 0:
            mult_range = 1

        if mult_time == 0:
            mult_time = 1

        if interpolate == False:
            sv = ds.sv[:, ::mult_time,::mult_range]
        else:
            sv = ds.sv.coarsen(range = mult_range, ping_time = mult_time, boundary="trim").mean(skipna=True)

    vmin = sv.min(skipna=True).compute()
    vmax = sv.max(skipna=True).compute()

    # Handle duplicate frequencies
    if len(sv.frequency.data) == len(np.unique(sv.frequency.data)):
        sv.plot(x="ping_time", y="range", row= "frequency", vmin = vmin, vmax = vmax, norm=colors.LogNorm(), cmap=simrad_cmap)
    else:
        frstr = ["%.2f" % i for i in sv.frequency.data]
        new_coords = []
        for frname in frstr:
            orig = frname
            i = 1
            while frname in new_coords:
                frname = orig + " #" + str(i)
                i += 1
            new_coords.append(frname)
        sv.coords["frequency"] = new_coords
        sv.plot(x="ping_time", y="range", row= "frequency", vmin = vmin, vmax = vmax, norm=colors.LogNorm(), cmap=simrad_cmap)

    plt.gca().invert_yaxis()
    plt.gcf().set_size_inches(8,11)
    plt.savefig(out_name + "." + 'png', bbox_inches = 'tight', pad_inches = 0)

def process_data_to_xr(raw_data, raw_obj=None, get_positions=False):
    # Get calibration object
    cal_obj = raw_data.get_calibration()
    # Get sv values
    sv_obj = raw_data.get_sv(calibration = cal_obj)
    # Get sv as depth
    #sv_obj_as_depth = raw_data.get_sv(calibration = cal_obj,
    #    return_depth=True)

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
    trdraft = xr.DataArray(name="transducer_draft", data=np.expand_dims(sv_obj.transducer_draft, axis=0), dims=['frequency', 'ping_time'],
                           coords={ 'frequency': [freq],
                                    'ping_time': sv_obj.ping_time,
                                   })

    # Additional data
    pulse_length = None
    angle_alongship = None
    angle_athwartship = None

    if hasattr(raw_data, 'pulse_length'):
        pulse_length = np.unique(raw_data.pulse_length)[0]
    elif hasattr(raw_data, 'pulse_duration'):
        pulse_length = np.unique(raw_data.pulse_duration)[0]
    else:
        pulse_length = 0

    # Calculate angles
    ang1, ang2 = raw_data.get_physical_angles(calibration = cal_obj)
    angle_alongship = sv.copy(data = np.expand_dims(ang1.data, axis=0))
    angle_athwartship = sv.copy(data = np.expand_dims(ang2.data, axis=0))

    if get_positions:
        positions = raw_obj.nmea_data.interpolate(sv_obj, 'RMC')
        return [sv, trdraft, pulse_length, angle_alongship, angle_athwartship, positions]
    else:
        return [sv, trdraft, pulse_length, angle_alongship, angle_athwartship]

def _resampleWeight(r_t, r_s):
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

def _regrid(sv_s, W, n_pings):
    """
    Use the weights to regrid the sv data
    """
    # Add a row of at the bottom to be used in edge cases
    sv_s_mod = np.vstack((sv_s, np.zeros(n_pings)))
    # Do the dot product
    return np.dot(W, sv_s_mod)

def regrid_sv(sv, reference_range):
    print("Channel with frequency " + str(sv.frequency.values[0]) + " range mismatch! Reference range size: " + str(reference_range.size) + " != " + str(sv.range.size))
    # Re-grid this channel sv
    sv_obj = sv[0,]
    W = _resampleWeight(reference_range.values, sv_obj.range)
    sv_tmp = _regrid(sv_obj.data.transpose(), W, sv_obj.ping_time.size).transpose()
    # Create new xarray with the same frequency
    sv = xr.DataArray(name="sv", data=np.expand_dims(sv_tmp, axis = 0), dims=['frequency', 'ping_time', 'range'],
                    coords={ 'frequency': sv.frequency,
                            'ping_time': sv.ping_time,
                            'range': reference_range.values,
                    })
    return sv

def expand_range(old_range, target, interval):

    # Create new range data using np.arange with a given interval
    new_range_data = np.arange(old_range[0].values, target, interval)

    # Remove values > target
    new_range_data = new_range_data[new_range_data < target]

    # Construct a new range
    new_range = xr.DataArray(name="range", data=new_range_data, dims=['range'],
                    coords={'range': new_range_data})

    return new_range

def compare_range(ref_range, src_range):
    len_ref = len(ref_range)
    len_src = len(src_range)

    if len_src > len_ref:
        return False
    else:
        if ref_range[:len_src].equals(src_range) == True:
            return True
        else:
            return False

def process_channel(raw_obj, channel, raw_data_main, reference_range):

    # Get the raw data
    raw_data = raw_obj.raw_data[channel][0]

    # Process channels with different ping times and with different frequencies
    if(np.array_equal(raw_data.ping_time, raw_data_main.ping_time) == False
        and raw_data.get_frequency(unique=True) != raw_data_main.get_frequency(unique=True)):
        print("This channel's time mismatched the main channel's, attempting match_pings() within 100th of a second.")
        raw_data.match_pings(raw_data_main)

    # Process it into xarray
    sv_bundle = process_data_to_xr(raw_data)

    # Check if we need to regrid this channel's sv
    if(compare_range(reference_range, sv_bundle[0].range) == False):
        sv_bundle[0] = regrid_sv(sv_bundle[0], reference_range)
        # Regridding means emptying the angles (TODO)
        sv_bundle[3] = sv_bundle[0].copy(data = np.full(sv_bundle[0].shape, np.nan))
        sv_bundle[4] = sv_bundle[0].copy(data = np.full(sv_bundle[0].shape, np.nan))
    else:
        # Ordinary padding (sv and angles)
        if(len(reference_range) != len(sv_bundle[0].range)):
            for it in [0, 3, 4]:
                sv_bundle[it] = sv_bundle[it].pad(range =(0, len(reference_range) - len(sv_bundle[it].range)))
                sv_bundle[it]['range'] = reference_range.values

    return [channel] + sv_bundle

def process_raw_file(raw_fname, main_frequency, reference_range = None):
    # Read input raw
    print("\n\nNow processing file: " + raw_fname)
    raw_obj = ek_read(raw_fname)
    print(raw_obj)

    # Get all channels
    all_channels = list(raw_obj.raw_data.keys())

    main_channel = all_channels.copy()

    # Get real frequency channel. Give an error and exit if not found.
    try:
        main_raw_data = raw_obj.get_channel_data(main_frequency)[main_frequency][0]
    except KeyError as error:
        print("There is no channel with the " + str(main_frequency) + " frequency. Using the first available channel!!!")
        # Fall back into using the first available channel.
        main_raw_data = raw_obj.raw_data[all_channels[0]][0]

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

    # Getting Sv for the main channel
    raw_data_main = raw_obj.raw_data[main_channel[0]][0]
    sv_bundle = process_data_to_xr(raw_data_main, raw_obj, get_positions=True)

    # Get positions
    positions = sv_bundle[5][1]

    # Check whether we need to set a reference range using this file's range or max_range
    if type(reference_range) == type(None):
        reference_range = sv_bundle[0].range
    else:
        # If we need to use the target range
        if isinstance(reference_range, (int, float, complex)) and not isinstance(reference_range, bool):
            range_intervals = list(a[0]-a[1] for a in zip(sv_bundle[0].range[1:].values, sv_bundle[0].range[:-1].values))
            unique_range_intervals = np.unique(range_intervals)
            if len(unique_range_intervals) > 1:
                print("ERROR: Interval is not unique!!!")
            reference_range = expand_range(sv_bundle[0].range, reference_range, unique_range_intervals)

        # Check if we also need to regrid this main channel
        if(compare_range(reference_range, sv_bundle[0].range) == False):
            sv_bundle[0] = regrid_sv(sv_bundle[0], reference_range)
            # Regridding means emptying the angles (TODO)
            sv_bundle[3] = sv_bundle[0].copy(data = np.full(sv_bundle[0].shape, np.nan))
            sv_bundle[4] = sv_bundle[0].copy(data = np.full(sv_bundle[0].shape, np.nan))
        else:
            # Ordinary padding (sv and angles)
            if(len(reference_range) != len(sv_bundle[0].range)):
                for it in [0, 3, 4]:
                    sv_bundle[it] = sv_bundle[it].pad(range =(0, len(reference_range) - len(sv_bundle[it].range)))
                    sv_bundle[it]['range'] = reference_range.values

    # Prepare placeholder for combined data
    sv_list = [sv_bundle[0]]
    trdraft_list = [sv_bundle[1]]
    plength_list = [sv_bundle[2]]
    angles_alongship_list = [sv_bundle[3]]
    angles_athwartship_list = [sv_bundle[4]]
            
    # Process Sv for all other channels in parallel
    worker_data = []
    for chan in other_channels:
        # Getting raw data for a frequency
        result = dask.delayed(process_channel)(raw_obj, chan, raw_data_main, reference_range)
        worker_data.append(result)
    
    ready = dask.delayed(zip)(*worker_data)
    channelID, sv, trdraft, plength, angles_alongship, angles_athwartship = ready.compute(scheduler='threads')

    sv_list.extend(list(sv))
    trdraft_list.extend(list(trdraft))
    plength_list.extend(list(plength))
    angles_alongship_list.extend(list(angles_alongship))
    angles_athwartship_list.extend(list(angles_athwartship))

    # Combine different frequencies
    da_sv = xr.concat(sv_list, dim='frequency')
    da_trdraft = xr.concat(trdraft_list, dim='frequency')
    da_angles_alongship = xr.concat(angles_alongship_list, dim='frequency')
    da_angles_athwartship = xr.concat(angles_athwartship_list, dim='frequency')

    # Getting motion data, apply extra treatment for duplicate frequencies with different times
    if(len(da_sv.ping_time) == len(raw_obj.motion_data.heave)):
        obj_heave = raw_obj.motion_data.heave
        obj_pitch = raw_obj.motion_data.pitch
        obj_roll = raw_obj.motion_data.roll
        obj_heading = raw_obj.motion_data.heading
    else:
        # Find nearest time for motion
        pidx = np.searchsorted(raw_obj.motion_data.times, da_sv.ping_time.data, side='right') - 1
        obj_heave = raw_obj.motion_data.heave[pidx]
        obj_pitch = raw_obj.motion_data.pitch[pidx]
        obj_roll = raw_obj.motion_data.roll[pidx]
        obj_heading = raw_obj.motion_data.heading[pidx]

        # Find nearest time for positions
        pidx = np.searchsorted(positions['ping_time'], da_sv.ping_time.data, side='right') - 1
        positions['latitude'] = positions['latitude'][pidx]
        positions['longitude'] = positions['longitude'][pidx]

    # Crate a dataset
    ds = xr.Dataset(
        data_vars=dict(
            sv=(["frequency", "ping_time", "range"], da_sv),
            angle_alongship = (["frequency", "ping_time", "range"], da_angles_alongship),
            angle_athwartship = (["frequency", "ping_time", "range"], da_angles_athwartship),
            transducer_draft=(["frequency", "ping_time"], da_trdraft),            
            heave=(["ping_time"], obj_heave),
            pitch=(["ping_time"], obj_pitch),
            roll=(["ping_time"], obj_roll),
            heading=(["ping_time"], obj_heading),
            pulse_length=(["frequency"], plength_list)
            ),
        coords=dict(
            frequency = da_sv.frequency,
            ping_time = da_sv.ping_time,
            range = da_sv.range,
            ),
        attrs=dict(description="Multi-frequency sv values from EK."),
    )

    # Add channel ID
    ds.coords["channelID"] = ("frequency", main_channel + list(channelID))

    # Add positions
    ds.coords["latitude"] = ("ping_time", positions['latitude'])
    ds.coords["longitude"] = ("ping_time", positions['longitude'])

    # Add ping_time to file mapping as coordinates
    ds.coords["raw_file"] = ("ping_time", [ntpath.basename(raw_fname)] * len(ds.ping_time))

    return ds

def raw_to_grid_single(raw_fname, main_frequency = 38000, write_output = False, out_fname = "", output_type = "zarr", overwrite = False):

    # Prepare for writing output
    target_fname = ""
    if write_output == True:
        # Construct target_fname
        if out_fname == "":
            out_fname = raw_fname
        if output_type == "netcdf4":
            target_fname = out_fname + ".nc"
        elif output_type == "zarr":
            target_fname = out_fname + ".zarr"
        else:
            print("Output type is not supported")
            return False

        # Check logic to proceed with write
        is_exists = (os.path.isfile(target_fname) or os.path.isdir(target_fname))
        if (is_exists == True and overwrite == True) or is_exists == False:
            do_write = True
        else:
            print("Output data exists. Not overwriting.")
            do_write = False
    else:
        print("Not writing output data.")
        do_write = False

    # Process single file
    ds = process_raw_file(raw_fname, main_frequency)
    
    print("Created dataset:")
    print(ds)

    if do_write == True:
        if output_type == "netcdf4":
            comp = dict(zlib=True, complevel=5)
            encoding = {var: comp for var in ds.data_vars}
            ds.to_netcdf(target_fname, mode="w", encoding=encoding)
        elif output_type == "zarr":
            compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
            encoding = {var: {"compressor" : compressor} for var in ds.data_vars}
            ds.to_zarr(target_fname, mode="w", encoding=encoding)
        else:
            print("Output type is not supported")
    
    return True


def prepare_resume(target_type, target_file, filename_list):

    # Try to open the file
    reference_range = None
    last_timestamp = None
    if target_type == "zarr":
        with xr.open_zarr(target_file) as tmp_src:
            last_timestamp = (tmp_src.ping_time[-1:]).values.astype('datetime64[s]')
            reference_range = tmp_src.range
    elif target_type == "netcdf4":
        with xr.open_dataset(target_file) as tmp_src:
            last_timestamp = (tmp_src.ping_time[-1:]).values.astype('datetime64[s]')
            reference_range = tmp_src.range
    else:
        print("Unsupported format. Can't resume.")

    # Re-select file list based on the last_timestamp recorded on the target flle
    # eg     "2020102-D20200302-T030956.raw" to time
    filename_list_date = [np.datetime64(datetime.datetime.strptime(''.join(fname.split(".")[:-1][0].split("-")[-2:]), 'D%Y%m%dT%H%M%S')) for fname in filename_list]
    filename_list_mask = [(fdate > last_timestamp).tolist()[0] for fdate in filename_list_date]
    new_filename_list = [*(d for d, s in zip(filename_list, filename_list_mask) if s)]

    return new_filename_list, reference_range

def get_max_range_from_files(dir_loc, raw_fname, main_frequency):
    print("Now trying to find the maximum range from the list of raw files...")
    ref_file = ''
    ref_range = 0

    for fn in raw_fname:
        # Read input raw
        raw_obj = ek_read(dir_loc + "/" + fn)
        try:
            main_raw_data = raw_obj.get_channel_data(main_frequency)[main_frequency][0]
        except KeyError as error:
            # Fall back into using the first available channel.
            main_raw_data = raw_obj.raw_data[list(raw_obj.raw_data.keys())[0]][0]
        if main_raw_data.data_type == 'power/angle':
            ref_data = main_raw_data.power
        elif main_raw_data.data_type == 'complex-FM' or main_raw_data.data_type == 'complex-CW':
            ref_data = main_raw_data.complex
        else:
            ref_data = np.zeros((0,0))

        range_len = ref_data.shape[1]
        if range_len > ref_range:
            ref_range = range_len
            ref_file = fn

    # Now get the maximum range
    raw_obj = ek_read(dir_loc + "/" + ref_file)
    try:
        main_raw_data = raw_obj.get_channel_data(main_frequency)[main_frequency][0]
    except KeyError as error:
        # Fall back into using the first available channel.
        main_raw_data = raw_obj.raw_data[list(raw_obj.raw_data.keys())[0]][0]

    cal_obj = main_raw_data.get_calibration()
    sv_obj = main_raw_data.get_sv(calibration = cal_obj)
    # Construct a new range
    new_range = xr.DataArray(name="range", data=sv_obj.range, dims=['range'],
                    coords={'range': sv_obj.range})

    print("Using this range from " + ref_file + ":")
    print(new_range)
    return new_range

def raw_to_grid_multiple(dir_loc, work_dir_loc, main_frequency = 38000, write_output = False, out_fname = "", output_type = "zarr", overwrite = False, resume = False, max_reference_range = None):

    # Misc. conditionals
    write_first_loop = True

    # List files
    raw_fname = [ntpath.basename(a) for a in sorted(glob.glob(dir_loc + "/*.raw"))] 

    # Check reference range info
    if type(max_reference_range) == type(None):
        # Use range from main_frequency channel on the first read file
        reference_range = None
    elif max_reference_range == "auto":
        # Do a pass on all files and use a suitable range
        reference_range = get_max_range_from_files(dir_loc, raw_fname, main_frequency)
    elif isinstance(max_reference_range, (int, float, complex)) and not isinstance(max_reference_range, bool):
        print("Using " + str(max_reference_range) + " as the maximum range.")
        reference_range = max_reference_range
    else:
        print("Invalid max_reference_range! Using the main_frequency channel's range on the first read file.")
        reference_range = None

    # Prepare for writing output
    target_fname = ""
    if write_output == True:
        # Construct target_fname
        if out_fname == "":
            out_fname = "out"
        if output_type == "netcdf4":
            target_fname = out_fname + ".nc"
        elif output_type == "zarr":
            target_fname = out_fname + ".zarr"
        else:
            print("Output type is not supported")
            return None

        # Check logic to proceed with write
        is_exists = (os.path.isfile(target_fname) or os.path.isdir(target_fname))

        # For overwriting
        if is_exists == True: 
            if overwrite == True:
                # Delete existing files
                if os.path.isfile(target_fname):
                    os.remove(target_fname)
                if os.path.isdir(target_fname):
                    shutil.rmtree(target_fname)
                do_write = True
            elif resume == True:
                # Resuming
                write_first_loop = False
                print("Trying to resume batch processing")
                # Updating file list and using the reference range
                raw_fname, reference_range = prepare_resume(output_type, target_fname, raw_fname)
                print("New list of files:")
                print(raw_fname)
                print("Reference range:")
                print(reference_range)
                do_write = True
            else:
                # All failed
                print("Output data exists. Not overwriting nor resuming.")
                do_write = False
        else:
            do_write = True
            
    else:
        do_write = False

    if do_write == False:
        # Nothing to do here
        return None

    # Prepare parquet file path for work file data
    pq_writer = None
    pq_filepath = out_fname + "_work.parquet"

    for fn in raw_fname:
        # Get base name
        base_fname, _ = os.path.splitext(fn)

        # Process single file
        ds = process_raw_file(dir_loc + "/" + fn, main_frequency, reference_range)

        # Process work file (if any)
        work_fname = work_dir_loc + "/" + base_fname + ".work"
        is_exists_work = os.path.isfile(work_fname)
        if is_exists_work:
            idx_fname = dir_loc + "/" + base_fname + ".idx"
            is_exists_idx = os.path.isfile(idx_fname)
            if is_exists_idx:
                # Process work file
                work = readers.work_reader(work_fname)
                ann_obj = readers.work_to_annotation(work, idx_fname)
                if ann_obj.df_ is not None:
                    # Exclude layers for now (only schools and gaps)
                    df = ann_obj.df_[ann_obj.df_.priority != 3]
                    pq_writer = append_to_parquet(df, pq_filepath, pq_writer)

        if do_write == True:
            if output_type == "netcdf4":
                compressor = dict(zlib=True, complevel=5)
                encoding = {var: compressor for var in ds.data_vars}
                if write_first_loop == False:
                        append_to_netcdf(target_fname, ds, unlimited_dims='ping_time')
                else:
                        ds.to_netcdf(target_fname, mode="w", unlimited_dims=['ping_time'], encoding=encoding)
                        # Propagate range to the rest of the files
                        reference_range = ds.range
            elif output_type == "zarr":
                compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
                encoding = {var: {"compressor" : compressor} for var in ds.data_vars}
                if write_first_loop == False:
                        ds.to_zarr(target_fname, append_dim="ping_time")
                else:
                        ds.to_zarr(target_fname, mode="w", encoding=encoding)
                        # Propagate range to the rest of the files
                        reference_range = ds.range
            else:
                print("Output type is not supported")

            write_first_loop = False

    return True

# Default input raw dir
raw_dir = os.path.expanduser("/datain")

# Default input work dir
work_dir = os.path.expanduser("/workin")

# Get the output type
out_type = os.getenv('OUTPUT_TYPE', 'zarr')

# Get the output name
out_name = os.path.expanduser("/dataout") + '/' + os.getenv('OUTPUT_NAME', 'out')

# Get the range determination type (numeric, 'auto', or None)
# A numeric type will force the range steps to be equal to the specified number
# 'auto' will force the range steps to be equal to the maximum range steps of all the processed files
# None will use the first file's main channel's range steps
max_ref_ran = os.getenv('MAX_RANGE_SRC', 'None')
if max_ref_ran != "auto":
    try:
        max_ref_ran = int(max_ref_ran)
    except ValueError as verr:
        max_ref_ran = None
    except Exception as ex:
        max_ref_ran = None

# Get the frequency for the main channel
main_freq = os.getenv('MAIN_FREQ', '38000')
try:
    main_freq = int(main_freq)
except ValueError as verr:
    main_freq = 38000
except Exception as ex:
    main_freq = 38000

# Get whether we should produce an overview image
do_plot = os.getenv('WRITE_PNG', '0')
if do_plot == '1':
    do_plot = True
else:
    do_plot = False

# Do process
status = raw_to_grid_multiple(raw_dir,
                            work_dir_loc = work_dir,
                            main_frequency = main_freq,
                            write_output = True,
                            out_fname = out_name,
                            output_type = out_type,
                            overwrite = False,
                            resume = True,
                            max_reference_range = max_ref_ran)

if status == True and do_plot == True:
    if out_type == "netcdf4":
        ds = xr.open_dataset(out_name + ".nc")
    else:
        ds = xr.open_zarr(out_name + ".zarr", chunks={'ping_time':'auto'})
    plot_all(ds, out_name)
