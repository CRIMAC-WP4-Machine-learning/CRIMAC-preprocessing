#!/bin/sh
/CRIMAC-preprocessing/run_magicsquare.sh /opt/mcr/v98 5
/CRIMAC/run_CRIMAC_preprocess.sh /opt/mcr/v98
python3 /CRIMAC-preprocessing/CRIMAC_preprocess_generate_memmap_files.py

