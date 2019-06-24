from Autoreg import build_ar_model
import pandas as pd
import numpy as np
import os

def preprocessing(dataset):
    remove_index = ['open', 'high', 'low', 'close', 'volume','Adj Close','y']
    df = dataset.drop(remove_index, axis=1)
    date=df['date'].copy()
    df = df.drop('date',axis=1)
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)].astype(np.float64)
    df['date']=date
    return df

file_name=['hdfc','marutisuzuki','infosys']
for i in file_name:
    data=pd.read_csv(i+'_modified.csv')
    try:
        os.mkdir(i)
    except:
        pass
    data=preprocessing(data)
    data=data.set_index('date')
    for j in data.columns:
        build_ar_model(i,data[j])