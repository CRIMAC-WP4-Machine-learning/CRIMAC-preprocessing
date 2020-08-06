#!/bin/sh
chmod 755 /CRIMAC-preprocessing/run_CRIMAC_preprocess.sh
/CRIMAC-preprocessing/run_CRIMAC_preprocess.sh /opt/mcr/v98
python3 /CRIMAC-preprocessing/CRIMAC_preprocess_generate_memmap_files.py

