import pandas as pd
import numpy as np
from joblib import load
import warnings

warnings.filterwarnings("ignore")

def preprocessing(dataset):
    remove_index = ['open','close','low','high','Adj Close','y','volume']
    df = dataset.drop(remove_index, axis=1)
    date=df['date'].copy()
    df = df.drop('date',axis=1)
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)].astype(np.float64)
    df['date']=date
    return df



def getFutureRows(company):
    data=pd.read_csv(company+"_modified.csv")
    data=preprocessing(data)
    data=data.set_index('date')
    t_ind={}
    for i in data.columns:
        model=load(company+'/'+i+'.joblib')
        t_ind[i]=model.get_forecast(steps=3,index=data.index).predicted_mean.values.tolist()
    return pd.DataFrame(data=t_ind)

