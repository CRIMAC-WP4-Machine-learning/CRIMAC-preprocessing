# This repository contain the code to preprocess acoustic data for the CRIMAC project

The repository contain code to pre-process a collection of SIMRAD's EK60/EK80 acoustic raw files and LSSS interpretaion masks into an `xarray` datasets using [pyEcholab](https://github.com/CI-CMG/PyEcholab) and the CRIMAC annotationtools (https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-annotationtools) . The dataset is then stored as `zarr` or `netcdf` files on disk.

## Processing steps

The processing is split into three separate steps. The steps needs to be run in order, but later steps can be rerun independent of the first step.

### Step 1: Index files

The first step is to generate an index-time file. The output from this step is a parquet file containing the individual input file names and associated ping and time numbers. In cases where there are discontinouties in the time or distance variable, a new time and distance variable is generated. This new variable is used when generating time and distance in the subsequent steps. The parquet file can be used to look up the original data.

The output of this step are the parquet files: 

<OUTPUT_NAME>_pingdist.parquet     
This file contains the uncorrected ping_time and distance values for the survey

<OUTPUT_NAME>_pingdistcorrected.parquet     
This file contains the corrected ping_time and distance values for the survey

The corrected parquet file contains the 3 following columns "raw_file" ,  "distance" and "ping_time"
This correction file is automatically read in step 2 



### Step 2: Generate gridded sv data

This step reads the .raw files and generate a gridded version of the data such that the dimension is time, range and frequency. If the range resolution is similar between the channels, the data is simply stacked. In cases where the data have different range resolution, the data is regridded onto the grid of the main frequency (MAIN_FREQ).

The output of this step is the Zarr/NetCDF file: `<OUTPUT_NAME>_sv.zarr` or `<OUTPUT_NAME>_sv.nc`.

### Step 3: Label data

This steps first convert Marec LSSS' work files into a parquet file containing the annotations using the CRIMAC-annotationtools. These data are independent of the gridded data in step 2. Next the data is overlayed on the grid from step 2, and a pixel wise annotation that matches the grid in step 2 is generated.

The output of this step is the parquet file: `<OUTPUT_NAME>_labels.parquet` and the Zarr/NetCDF file: `<OUTPUT_NAME>_labels.zarr` or `<OUTPUT_NAME>_labels.nc`.

## Features

1. Automatic range re-gridding (by default it uses the main channel’s range from the first raw file, see `MAX_RANGE_SRC` option below).
2. Sv processing and re-gridding the channels are done in parallel (using `Dask`’s delayed).
3. Automatic resuming from the last `ping_time` if the output file exists.
4. Batch processing is done by appending directly to the output file, should be memory efficient.
5. The image of this repository is available at Docker Hub (https://hub.docker.com/r/crimac/preprocessor).
6. Processing annotations from `.work` files into a `pandas` dataframe object (using: https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-annotationtools).

## Options to run

### Using local Python installation


#### Example



### Using Docker
1. Two directories need to be mounted:

    1. `/datain` should be mounted to the data directory where the `.raw` files are located.
    2. `/dataout` should be mounted to the directory where the output is written.
    3. `/workin` should be mounted to the directory where the `.work` files are located (_optional_).

2. Choose the frequency of the main channel: 

    ```bash
    --env MAIN_FREQ=38000
    ```

3. Choose the range determination type: 

    ```bash
    # Set the maximum range as 500,
    --env MAX_RANGE_SRC=500

    # or use the the main channel's maximum range from all the files (for historical data),
    --env MAX_RANGE_SRC=auto
    
    # or use the the main channel's maximum range from the first processed file (for historical data)
    --env MAX_RANGE_SRC=None
    ```

4. Select output type, `zarr` and `NetCDF4` are supported:

    ```bash
    --env OUTPUT_TYPE=zarr

    --env OUTPUT_TYPE=netcdf4
    ```

5. Select file name output (optional,  default to `out.<zarr/nc>`)

    ```bash
    --env OUTPUT_NAME=S2020842
    ```

6. Set if we want a visual overview of the Sv data (in a PNG format image)

    ```bash
    --env WRITE_PNG=1 # enable or 0 to disable
    ```

7. Optional attribute to process only one selected file when there are many raw files in the raw folder

    ```bash
    --env RAW_FILE=2019847-D20190509-T014326.raw
    ```

8. Optional attribute for logging LOGGING=1 (on) LOGGING=0 (off). Standard is with logging on when the attribute is not set
    
    ```bash
    --env LOGGING=1 # enable or 0 to disable
    ```

9. Optional attribute for debug (detailed stderr output) DEBUG=1 (on) DEBUG=0 (off). Standard is with debug off when the attribute is not set. DEBUG=1 will often exit on errors
    
    ```bash
    --env DEBUG=1 # enable or 0 to disable
    ```


#### Example

```bash

docker run -it \
-v /data/cruise_data/2020/S2020842_PHELMERHANSSEN_1173/ACOUSTIC/EK60/EK60_RAWDATA:/datain \
-v /data/cruise_data/2020/S2020842_PHELMERHANSSEN_1173/ACOUSTIC/LSSS/WORK:/workin \
-v /localscratch/ibrahim-echo/out:/dataout \
--security-opt label=disable \
--env OUTPUT_TYPE=zarr \
--env MAIN_FREQ=38000 \
--env MAX_RANGE_SRC=500 \
--env OUTPUT_NAME=S2020842 \
--env WRITE_PNG=0 \
crimac/preprocessor

```
#### manual docker build
```bash
git clone https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-preprocessing.git
cd CRIMAC-preprocessing/
docker build --build-arg=commit_sha=$(git rev-parse HEAD) --no-cache --tag crimac-preprocessor20230208 .
```
