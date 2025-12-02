

import pandas as pd
import numpy as np

from scipy.interpolate import  lagrange

def ploy (s,n,k=5):
    y=s[list(range(n-k,n))+list(range(n+1,n+k+1))]
    y=y[y.notnull()]
    return lagrange(y.index,list(y))(n)

traj=pd.read_csv("D:\project\DATASET-A.csv",header=None,usecols=[2,3,4]).iloc[1:16]
traj.columns=["timestamp","lon","lat"]
print(traj)

#print(type(traj["timestamp"]))
#print(traj["timestamp"].dtype)
#traj["time_interval"]=traj["timestamp"] - traj["timestamp"].shift(1)

A=pd.to_numeric(traj["timestamp"],errors='coerce')
B=pd.to_numeric(traj["timestamp"].shift(1),errors='coerce')
traj["time_interval"]=A-B
index=traj[traj["time_interval"]>=6].index.to_list()

for i in index:
    timestamp=int(traj["timestamp"].loc[i-1]) + 3
    insertRow =pd.DataFrame([[np.nan,np.nan,timestamp]],columns=['lon','lat','timestamp'])
    traj=pd.concat([traj[:i],insertRow,traj[i:]],ignore_index=True)
    #traj['lon'][1]=ploy(traj['lon'],1)
    #traj['lat'][1]=ploy(traj['lat'],1)
    traj['lon'][i]=str(format(ploy(pd.to_numeric(traj["lon"],errors='coerce'),i),'.7f'))
    traj['lat'][i]=str(format(ploy(pd.to_numeric(traj["lat"],errors='coerce'),i),'.8f'))
traj=traj.drop(['time_interval'],axis=1)
print(traj)
