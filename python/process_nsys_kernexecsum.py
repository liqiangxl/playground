import sys
import pandas as pd

file = str(sys.argv[1])+"_kernexecsum.csv"
data = pd.read_csv(file, skiprows=0)
#data = data.loc[data['Kernel Name'].str.contains('ln_fwd')]
data = data[['TAvg (ns)', 'AAvg (ns)', 'QAvg (ns)', 'KAvg (ns)', 'KMin (ns)','KMax (ns)','API Name', 'Kernel Name']]
print(data)
