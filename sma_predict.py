import pandas as pd
import numpy as np
import statsmodels.api as sm
from joblib import load

results=load('SMA.joblib')
pred=results.get_forecast(steps=3)
print(pred.predicted_mean)