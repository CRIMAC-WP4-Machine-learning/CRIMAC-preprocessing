#!/bin/bash

var=${MODE:-script}

# Either run master, worker, or the script
echo "Mode is $var"

if [[ $var == "master" ]]; then
    dask-scheduler
fi
if [[ $var == "worker" ]]; then
    dask-worker crimac-master:8786
fi
if [[ $var == "script" ]]; then
  time python CRIMAC_preprocess.py
fi