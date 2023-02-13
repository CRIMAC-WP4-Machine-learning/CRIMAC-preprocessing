import pyarrow.parquet as pq
import pandas as pd
import timeit
import random
 

 
# Read the Parquet file
table = pq.read_table("/localscratch_hdd/tomasz/cruise/2019/GRIDDED/S2019847_pingdistcorrected.parquet")
df= table.to_pandas()
print (df)

 
 
starttime = timeit.default_timer()
print("The start time is :",starttime)


filter_column = "raw_file"

filter_value = "2019847-D20190511-T005652.raw"
filtered_df = df.loc[df[filter_column] == filter_value]
column_name = "distance"
column = filtered_df[column_name]
array = column.to_numpy()
print (len(array  ))
print (type(array  ))
print (array)
print("The time difference is :", timeit.default_timer() - starttime)


starttime = timeit.default_timer()
print("The start time is :",starttime)
filter_column = "raw_file"
filter_value = "2019847-D20190512-T030936.raw"
filtered_df = df.loc[df[filter_column] == filter_value]
print(filtered_df )
column_name = "distance"
column = filtered_df[column_name]
array = column.to_numpy()
print (len(array  ))
print (type(array  ))
print (array)
print("The time difference is :", timeit.default_timer() - starttime)


starttime = timeit.default_timer()
print("The start time is :",starttime)
filter_column = "raw_file"
filter_value = "2019847-D20190512-T030936.raw"
filtered_df = df.loc[df[filter_column] == filter_value]
print(filtered_df )
column_name = "ping_time"
column = filtered_df[column_name]
array = column.to_numpy()
print (len(array  ))
print (type(array  ))
print (array)
print(type(array[0]))
print("The time difference is :", timeit.default_timer() - starttime)
