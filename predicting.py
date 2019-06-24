import pandas as pd
import numpy as np
from getTI import getFutureRows
from joblib import load
import warnings
import sys

warnings.filterwarnings("ignore")
company=['marutisuzuki','hdfc','infosys']

for i in company:
    model=load(i+'.joblib')
    X=getFutureRows(i)
    scaler=load(i+'_scaler.joblib')
    X_scaled=scaler.transform(X)
    X_scaled = pd.DataFrame(data=X_scaled, index=X.index, columns=X.columns)
    print(model.predict(X_scaled))

