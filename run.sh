#!/bin/bash

var=${MODE:-script}
master=${MASTER_HOST:localhost}

# Either run master, worker, or the script
echo "Mode is $var"
echo "Scheduler address is $master"

if [[ $var == "master" ]]; then
    dask-scheduler
fi
if [[ $var == "worker" ]]; then
    dask-worker $master:8786
fi
if [[ $var == "script" ]]; then
  time python CRIMAC_preprocess.py
fi