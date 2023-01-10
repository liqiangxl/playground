import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

oneGB = 1e9
srow= 10
needs_remove_duplication = False
hostname = "nvdl-a112-d001"
TITLE=hostname + ", cpp benckmark"

#baseFile = "/hhome/debug/layer_norm/" + hostname + "_NvFuserScheduler_LayerNorm_fp16_debug.csv" #"/hhome/debug/layer_norm/" + hostname + "_lm16devel.csv"
#baseName = "pb12-32"
fileDir = "/workspace/playground/debug/all_benchmarks/"
baseFile = fileDir + "all10ms_old.csv"
baseCase = "before_this_PR"
ListOfDataFiles=[]
ListOfDataCases=[]

ListOfDataFiles.append(fileDir + "all10ms.csv")
ListOfDataCases.append("after_this_PR")


def getInfo(raw_data):
    case_name             = raw_data['name']
    bytes_per_second      = raw_data['bytes_per_second'] / oneGB
    df_raw = pd.DataFrame({'name':case_name, 'bytes_per_second': bytes_per_second })
    if needs_remove_duplication:
        df = df_raw.groupby(['out_dim','inner_dim']).mean() # average duplicated runs
        df = df.reset_index(level=(0,1))
        return df
    else:
        return df_raw

def read_process(dataFileName, srow=srow):  
    raw_data = pd.read_csv(dataFileName, skiprows=srow)
    df_new = getInfo(raw_data)
    return df_new['bytes_per_second']

#============================= main ##############
raw_data = pd.read_csv(baseFile, skiprows=srow)
new_data = getInfo(raw_data)
df = pd.DataFrame({'name':new_data['name']})
df[baseCase] = new_data['bytes_per_second']
nfiles = len(ListOfDataFiles)
yplot = [baseCase]
for i in range(nfiles):
    if i>=0:
        new_column = read_process(ListOfDataFiles[i])
    else:
        new_column = read_process(ListOfDataFiles[i])
    df[ListOfDataCases[i]] = new_column
    yplot.append(ListOfDataCases[i]);

# s-3 show data
# relative speed
df['Bandwidth new/old'] = df[ListOfDataCases[0]] / df[baseCase]
gain_df = df[df['Bandwidth new/old'] > 1.05]
drop_df = df[df['Bandwidth new/old'] < 0.95]

axarr = gain_df.hist(column='Bandwidth new/old', bins=20)
for ax in axarr.flatten():
    ax.set_xlabel("Bandwidth new/old")
    ax.set_ylabel("count")
    ax.set_title(TITLE)
gain_df.to_csv(fileDir+"benckmark_perf_gains.csv")

axarr2 = drop_df.hist(column='Bandwidth new/old', bins=20)
for ax in axarr2.flatten():
    ax.set_xlabel("Bandwidth new/old")
    ax.set_ylabel("count")
    ax.set_title(TITLE)
drop_df.to_csv(fileDir+"benckmark_perf_drop.csv")