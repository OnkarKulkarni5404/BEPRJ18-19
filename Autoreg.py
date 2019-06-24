import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
from joblib import dump
import matplotlib.pyplot as plt
#plt.style.use('fivethirtyeight')

def preprocessing(dataset):
    remove_index = ['open', 'high', 'low', 'close', 'volume','Adj Close','y']
    df = dataset.drop(remove_index, axis=1)
    date=df['date'].copy()
    df = df.drop('date',axis=1)
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)].astype(np.float64)
    df['date']=date
    return df

def build_ar_model(company_name,techinal_indicator):
    # Define the p, d and q parameters to take any value between 0 and 2
    p = d = q = range(0, 2)

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    warnings.filterwarnings("ignore")
    minimum=999999999
    opt_param=0
    opt_param_seasonal=0
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod=sm.tsa.statespace.SARIMAX(techinal_indicator,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
                results = mod.fit(disp=0)
                if minimum>results.aic:
                    minimum=results.aic
                    opt_param=param
                    opt_param_seasonal=param_seasonal
            except:
                continue

    mod=sm.tsa.statespace.SARIMAX(techinal_indicator,order=opt_param,seasonal_order=opt_param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
    results=mod.fit(disp=0)
    dump(results,company_name+'/'+techinal_indicator.name+'.joblib')
