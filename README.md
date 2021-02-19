# This repository contain the code to preprocess acoustic data for the CRIMAC project

## Features

1. Automatic range re-gridding (by default it uses the main channel’s range from the first raw file).
2. Sv processing and re-gridding the channels are done in parallel (using `Dask`’s delayed).
3. Automatic resuming from the last `ping_time` if the output file exists.
4. Batch processing is done by appending directly to the output file, should be memory efficient.
5. The image of this repository is available at Docker Hub (https://hub.docker.com/r/crimac/test-preprocessor).

## Options to run

1. Two directories need to be mounted:

    1. `/datain` should be mounted to the data directory where the `.raw` files are located.
    2. `/dataout` should be mounted to the directory where the output is written.

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

## Example

```bash

docker run -it --name pyechopreprocess \
-v /data/cruise_data/2020/S2020842_PHELMERHANSSEN_1173/ACOUSTIC/EK60/EK60_RAWDATA:/datain \
-v /localscratch/ibrahim-echo/out:/dataout  \
--security-opt label=disable \
--env OUTPUT_TYPE=zarr \
--env MAIN_FREQ=38000 \
--env MAX_RANGE_SRC=500 \
--env OUTPUT_NAME=S2020842 \
--env WRITE_PNG=0 \
crimac/preprocessor


```
