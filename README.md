# This repository contain the code to preprocess acoustic data for the CRIMAC project

## Features

1. Automatic range re-gridding (by default it uses the main channel’s range from the first raw file).
2. Sv processing and re-gridding the channels are done in parallel (using `Dask`’s delayed).
3. Automatic resuming from the last `ping_time` if the output file exists.
4. Batch processing is done by appending directly to the output file, should be memory efficient.
5. The image of this repository is available at Docker Hub (https://hub.docker.com/r/crimac/test-preprocessor).

## Options to run

1. Two directories needs to be linked:

    1. `/datain` should be mapped to the data directory where the .raw, .bot and .idx files are located.
    2. `/dataout` should be mapped to the directory where the memmap files are located.

2. Select output type, zarrand NetCDF4 are supported:

    ```bash
    --env OUTPUT_TYPE=zarr

    --env OUTPUT_TYPE=netcdf4
    ```

3. Select file name output (optional,  default to `out.<zarr / nc>`)

    ```bash
    --env OUTPUT_NAME=S2020842
    ```

## Example

```bash

docker run -it --name pyechopreprocess \
-v /data/cruise_data/2020/S2020842_PHELMERHANSSEN_1173/ACOUSTIC/EK60/EK60_RAWDATA:/datain \
-v /localscratch/ibrahim-echo/out:/dataout  \
--security-opt label=disable \
--env OUTPUT_TYPE=zarr \
--env OUTPUT_NAME=S2020842 \
crimac/test-preprocessor

```
