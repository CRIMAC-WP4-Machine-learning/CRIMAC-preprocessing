import os
import sys
import CRIMAC_preprocess
import gc

#walk_dir = sys.argv[1]

walk_dir ="/Users/tf/work/crimacannot/test_data2/"
#print('walk_dir = ' + walk_dir)

savedir ="/Users/tf/work/crimacannot/result3/"
os.system("mkdir "+savedir)


for root, subdirs, files in os.walk(walk_dir):

    for filename in files:
        file_path = os.path.join(root, filename)
        if filename.endswith("raw"):
            #print('  %s  : %s' % (root , filename))
            raw = root

            pos=root.index("ACOUSTIC")
            work = root[:pos]+"ACOUSTIC/LSSS/WORK"
            #work = root

            pos=filename.index("raw")-1
            savename= filename[:pos]
            savefolder= savedir+filename[:pos]
            os.system("mkdir " +  savefolder)

            try:
                CRIMAC_preprocess.parsedata(
                    rawdir=raw,
                    workdir=work,
                    outdir=savefolder,
                    OUTPUT_TYPE="zarr",
                    OUTPUT_NAME=savename,
                    MAX_RANGE_SRC="500",
                    MAIN_FREQ="38000",
                    RAW_FILE=filename,
                    WRITE_PNG="1",
                    LOGGING="1",
                    DEBUGMODE="1")
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print("ERROR: Something went wrong when reading the WORK file: " + str(filename) +"  "+ str(fname) + " (" + str(
                    e) + ")")

            print("gc.collect memory")
            print(gc.get_count())
            print(gc.collect())
            print(gc.get_count())
