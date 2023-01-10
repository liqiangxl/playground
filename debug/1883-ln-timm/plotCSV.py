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
fileDir = "/workspace/playground/debug/1883-ln-timm/"
baseFile = fileDir + "Baseline_TIMM_LayerNorm_fp16.csv"
baseName = "eager"
ListOfDataFiles=[]
ListOfDataCases=[]

ListOfDataFiles.append(fileDir + "NvFuserScheduler_TIMM_LayerNorm_fp16_welford.csv")
ListOfDataCases.append("fuser (welford)")

ListOfDataFiles.append(fileDir + "NvFuserScheduler_TIMM_LayerNorm_fp16_twopass.csv")
ListOfDataCases.append("fuser (two-pass)")

def getInfo(raw_data):
    #NCHW
    h = raw_data['name'].apply(lambda s : int(s.split("/")[-2]))
    w = h
    c = raw_data['name'].apply(lambda s : int(s.split("/")[-3]))    
    n = raw_data['name'].apply(lambda s : int(s.split("/")[-4]))
    out_dim = n*h*w
    inner_dim = c
    bytes_per_second      = raw_data['bytes_per_second'] / oneGB
    df_raw = pd.DataFrame({ 'out_dim':out_dim, 'inner_dim': inner_dim, 'bytes_per_second': bytes_per_second })
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
df = pd.DataFrame({ 'out_dim':new_data['out_dim'], 'inner_dim': new_data['inner_dim'], baseName:new_data['bytes_per_second']})
nfiles = len(ListOfDataFiles)
yplot = [baseName]
for i in range(nfiles):
    if i>=0:
        new_column = read_process(ListOfDataFiles[i])
    else:
        new_column = read_process(ListOfDataFiles[i])
    df[ListOfDataCases[i]] = new_column
    yplot.append(ListOfDataCases[i]);

# s-3 show data

# bandwidth
#df = df.sort_values(by=['inner_dim'])

df.plot(x="inner_dim", y=yplot, kind="bar", ylabel="Bandwidth (GB/s)", title=TITLE)
print(df)

# relative speed
df['Bandwidth fuser(two-pass)/eager'] = df[ListOfDataCases[1]] / df[baseName]
axarr = df.hist(column='Bandwidth fuser(two-pass)/eager', bins=20)
for ax in axarr.flatten():
    ax.set_xlabel("Bandwidth fuser/eager")
    ax.set_ylabel("count")
    ax.set_title(title=TITLE)

# relative speed
df['Bandwidth fuser(two-pass)/ fuser(welford)'] = df[ListOfDataCases[1]] / df[ListOfDataCases[0]]
axarr2 = df.hist(column='Bandwidth fuser(two-pass)/ fuser(welford)', bins=20)
for ax in axarr2.flatten():
    ax.set_xlabel("Bandwidth fuser(two-pass)/ fuser(welford)")
    ax.set_ylabel("count")
    ax.set_title(title=TITLE)

## save data
df.to_csv(fileDir+"timm_benckmark.csv")