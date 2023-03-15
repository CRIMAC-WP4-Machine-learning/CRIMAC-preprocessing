import os
import sys
import CRIMAC_preprocess
import gc


# this is just used as a wrapper to run and test the preprocessor outside of docker from the commandline
#
# to run the preprocessor at HI sewrvers run the 2 following commands to activate the crimac conda environment
#
# source /software/anaconda-bio/anaconda3/bin/activate 
# conda activate crimac


raw = sys.argv[1]
work = sys.argv[2]
savename = sys.argv[3]
savedir = sys.argv[4]
runtype= sys.argv[5]



CRIMAC_preprocess.parsedata(    rawdir=raw,    workdir=work,    outdir=savedir,    OUTPUT_TYPE=runtype,    OUTPUT_NAME=savename,    MAX_RANGE_SRC="500",    MAIN_FREQ="38000")
