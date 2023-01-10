import pandas as pd


METHOD="fp16_a100_regmax"
srow= 10
needs_remove_duplication = False

mycsvOld = "/hhome/debug/layer_norm/" + "nvdl-a112-d001_lm16baseline.csv"
mycsvNew = "/hhome/debug/layer_norm/" + "nvdl-a112-d001_lm16pr.csv"

# s-1, read data
bm16Old = pd.read_csv(mycsvOld, skiprows=srow)
bm16New = pd.read_csv(mycsvNew, skiprows=srow)

# s-2 process data, average over duplicated runs
def getInfo(bm16):
    out_dim = bm16['name'].apply(lambda s : int(s.split("/")[-3]))
    inner_dim = bm16['name'].apply(lambda s : int(s.split("/")[-2]))
    bytes_per_second      = bm16['bytes_per_second']
    df_raw = pd.DataFrame({ 'out_dim':out_dim, 'inner_dim': inner_dim, 'bytes_per_second': bytes_per_second })
    if needs_remove_duplication:
        df = df_raw.groupby(['out_dim','inner_dim']).mean() # average duplicated runs
        df = df.reset_index(level=(0,1))
        return df
    else:
        return df_raw

df_old = getInfo(bm16Old)
df_new = getInfo(bm16New)
speedup_mem = df_new['bytes_per_second'] / df_old['bytes_per_second']
speedup_mem_cut = speedup_mem.clip(0.5,1.5)
df = pd.DataFrame({ 'out_dim':df_old['out_dim'], 'inner_dim': df_old['inner_dim'],\
     'ratio of memory throughput (higher is better)': speedup_mem_cut , 'speedup_unclip' : speedup_mem})            

# s-3 show data
df.plot(kind='scatter', x='inner_dim', xlabel='inner_dim', y='out_dim', ylabel="out_dim", \
    title=METHOD, loglog=False, c='ratio of memory throughput (higher is better)', colormap='rainbow')
axarr = df.hist(column='ratio of memory throughput (higher is better)', bins=20)


for ax in axarr.flatten():
    ax.set_xlabel("speedup")
    ax.set_ylabel("count")
    ax.set_title("{512,512*64} x {10240, 10240*5} and {512,512*64} x {512,512*64}")

print(speedup_mem.describe())
print('worst case idx = ', speedup_mem.idxmin())
print('worst case val = ', speedup_mem.min())
print('worst case val = ', df.iloc[speedup_mem.idxmin()])

print('\nbest case idx = ', speedup_mem.idxmax())
print('best case val = ', speedup_mem.max())
print('nbest case val = ', df.iloc[speedup_mem.idxmax()])
