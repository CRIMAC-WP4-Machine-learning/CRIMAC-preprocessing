#!/bin/sh
chmod 755 /CRIMAC-preprocessing/run_magicsquare.sh
chmod 755 /CRIMAC-preprocessing/run_CRIMAC_preprocess.sh
/CRIMAC-preprocessing/run_magicsquare.sh /opt/mcr/v98 5
/CRIMAC-preprocessing/run_CRIMAC_preprocess.sh /opt/mcr/v98
python3 /CRIMAC-preprocessing/CRIMAC_preprocess_generate_memmap_files.py

