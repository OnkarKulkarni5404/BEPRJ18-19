import pandas as pd
import matplotlib.pyplot as mp
import numpy as np
import os
import math


def load_stock(companyname=""):
    csv_path = os.path.join('individual_stocks_5yr',companyname)
    return pd.read_csv(csv_path)


#indicator 1 Simple Moving Average
def calculateSMA(data,window):
    ans=[]
    values=np.asarray(data['close'])
    weights=np.repeat(1.0,window)/window
    smas=np.convolve(values,weights,'valid')
    ans=smas.tolist()
    for x in range(1,window):
        ans.insert(0,None)
    return data.assign(SMA=ans)


#indicator 2 Weighted Moving Average
def calculateWMA(A_data):
    WMA=[]
    num=0;
    for i in range(A_data.shape[0]):
        if i<= 8:
            WMA.append(None)
        else:
            SUM=0
            for j in range(i-9,i+1):
                SUM=SUM+((j+1)*A_data['close'][j])
                num=num+(j+1)
            WMA.append(SUM/num)
    return A_data.assign(WMA=WMA)

#indicator 3 Momentum
def calculateMomentum(data):
    Mom=[]
    for i in range(data.shape[0]):
        if i<=8:
            Mom.append(None)
        else:
            Mom.append(data['close'][i-9]-data['close'][i])
    return data.assign(Momentum=Mom)

#indicator 4 Relative Strength Index
def calculateRSI(A_data):
    n = 14
    delta = A_data['close'].diff()
    # dUp= delta[delta > 0]
    # dDown= delta[delta < 0]

    # dUp = dUp.reindex_like(delta, fill_value=0)
    # dDown = dDown.reindex_like(delta, fill_value=0)

    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0

    RolUp = dUp.rolling(n).mean()
    RolDown = dDown.rolling(n).mean()

    RS = RolUp / RolDown
    rsi = 100.0 - (100.0 / (1.0 + RS))
    return A_data.assign(RSI=rsi)

#indicator 5 William K%
def calculateWilliam(A_data):
    William=[]
    for i in range(A_data.shape[0]):
        if i<14:
            William.append(None)
        else:
            MIN=99
            MAX=0
            for j in range(i-14,i+1):
                MIN=MIN if MIN<A_data['low'][j] else A_data['low'][j]
                MAX=MAX if MAX>A_data['high'][j] else A_data['high'][j]
            William.append((MAX-A_data['close'][j])/(MAX-MIN) *-100)
    return A_data.assign(WilliamK=William)

#indicator  Exponential Moving Average
def calculateEMA(data,window):
    values = np.asarray(data['close'])
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a = np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a

#indicator 6 Moving average convergence divergence
def calculateMACD(data, slow=26, fast=12):
    emaslow = calculateEMA(data, slow)
    emafast = calculateEMA(data, fast)
    #print (len(emafast - emaslow))
    return data.assign(MACD=(emafast-emaslow))


#indicator 7 CCI
'''
CCI = (Typical Price  -  20-period SMA of TP) / (.015 x Mean Deviation)
Typical Price (TP) = (High + Low + Close)/3
Constant = .015
There are four steps to calculating the Mean Deviation:
First, subtract the most recent 20-period average of the typical price from each period's typical price.
Second, take the absolute values of these numbers.
Third, sum the absolute values.
Fourth, divide by the total number of periods (20).
'''


def calculatecci1(values):
    window = 20
    weights = np.repeat(1.0, window) / window
    smas = np.convolve(values, weights, 'valid')
    return smas


def calculateCCI(A_data):
    typprice = []
    Mdar = []
    cci = []
    tf = 14
    x = 0
    while (x < A_data.shape[0]):
        tp = (A_data['low'][x] + A_data['high'][x] + A_data['close'][x]) / 3
        typprice.append(tp)
        x += 1

    smatp = calculatecci1(typprice)
    typprice = typprice[19:]
    #print(len(typprice))
    #print(len(smatp))
    y = tf
    while y < len(smatp):
        considerationTP = typprice[y - tf:y]
        considerationsmatp = smatp[y - tf:y]
        Mds = 0
        z = 0
        while z < len(considerationTP):
            curMD = abs(considerationTP[z] - considerationsmatp[z])
            Mds += curMD
            z += 1
        MD=Mds/tf
        Mdar.append(MD)
        y += 1
    typprice = typprice[14:]
    smatp = smatp[14:]
    # print(len(Mdar))
    #print(len(smatp))
    #rint(len(typprice))
    xx = 0
    while xx < len(smatp):
        ccis = (typprice[xx] - smatp[xx]) / (0.015 * Mdar[xx])
        cci.append(ccis)
        xx += 1
    for x in range(0, tf + 20 - 1):
        cci.insert(0,None)
    return A_data.assign(CCI=cci)

#indicator 8 Accumulation Distribbution Line
def acc_dist(data):
    trend_periods=21
    ad=[]
    for x in range (0,data.shape[0]):
        if(data['high'][x]!=data['low'][x]):
            ac=((data['close'][x]-data['low'][x])-(data['high'][x]-data['close'][x]))/(data['high'][x]-data['low'][x]) * \
               data['volume'][x]
        else:
            ac=0
        data.set_value(x, 'acc_dist', ac)
    data['AD-:' + str(trend_periods)] = data['acc_dist'].ewm(ignore_na=False, min_periods=0, com=trend_periods,
                                                                     adjust=True).mean()
    return data

#indicator 9 Aroon up and Aroon Down
def calculateAroon(data):
    tf=25
    high=np.asarray(data['high'])
    low=np.asarray(data['low'])
    AroonUP=[]
    AroonDown=[]
    x=tf
    while x<data.shape[0]:
        ans1=(((high[x-tf:x].tolist().index(max(high[x-tf:x]))))/float(tf))*100
        ans2=(((low[x-tf:x].tolist().index(min(low[x-tf:x]))))/float(tf))*100
        AroonUP.append(ans1)
        AroonDown.append(ans2)
        x+=1
    for x in range(0,tf):
        AroonUP.insert(0,None)
        AroonDown.insert(0,None)
    data=data.assign(Arronup=AroonUP)
    data=data.assign(AroonDown=AroonDown)
    #print(data['Arronup'])
    return data

#indicator 10 Aroon Oscillator
def AroonOscillator(data):
    AO=[]
    x=0
    while x<data.shape[0]:
        if x<25:
            AO.append(None)
        else:
            AO.append(data['Arronup'][x]-data['AroonDown'][x])
        x+=1
    return data.assign(AroonOscillator=AO)

#indicator 11 Average True Range
def cTR(h,l,yc):
    x=h-l
    y=abs(h-yc)
    z=abs(l-yc)

    if y<=x>=z:
        TR=x
    elif x<=y>=z:
        TR=y
    elif x<=z>=y:
        TR=z

    return TR


def calcTrueRange(data):
    x=0
    TrueRange=[]
    for x in range(1,data.shape[0]):
        TrueRange.append(cTR(data['high'][x],data['low'][x],data['close'][x-1]))
    TrueRange.insert(0,None)
    return data.assign(ATR=TrueRange)

def rateofchange(data):
    n=data.shape[0]
    M=data['close'].diff(n-1)
    N=data['close'].shift(n-1)
    ROC=pd.Series(M/N,name='ROC_'+str(n))
    data=data.join(ROC)
    return data


def ppsr(df):
    """Calculate Pivot Points, Supports and Resistances for given data
        Pivot Point (P) = (High + Low + Close)/3
        Support 1 (S1) = (P x 2) - High
        Support 2 (S2) = P  -  (High  -  Low)
        Resistance 1 (R1) = (P x 2) - Low
        Resistance 2 (R2) = P + (High  -  Low)
    """
    PP = pd.Series((df['high'] + df['low'] + df['close']) / 3)
    R1 = pd.Series(2 * PP - df['low'])
    S1 = pd.Series(2 * PP - df['high'])
    R2 = pd.Series(PP + df['high'] - df['low'])
    S2 = pd.Series(PP - df['high'] + df['low'])
    R3 = pd.Series(df['high'] + 2 * (PP - df['low']))
    S3 = pd.Series(df['low'] - 2 * (df['high'] - PP))
    psr = {'Pivot_Point': PP, 'Resistance_1': R1, 'Support_1': S1, 'Resistance_2': R2, 'Support_2': S2, 'Resistance_3': R3, 'Support_3': S3}
    PSR = pd.DataFrame(psr)
    df = df.join(PSR)
    return df


def stochastic_oscillator_k(df):
    SOk = pd.Series((df['close'] - df['low']) / (df['high'] - df['low']), name='SO%k')
    df = df.join(SOk)
    return df

#Calculating On_Balance_Moving Average
def onbv(data):
    onbv=[]
    for i in range(data.shape[0]):
        if i>0:
            last_obv=onbv[i-1]
            if data['close'][i]>data['close'][i-1]:
                current_obv=last_obv+data['volume'][i]
            elif data['close'][i]<data['close'][i-1]:
                current_obv=last_obv-data['volume'][i]
            else:
                current_obv=last_obv
        else:
            last_obv=0
            current_obv=data['volume'][i]
        onbv.append(current_obv)
    # print(onbv)
    return data.assign(On_Balance_Volume=onbv)

def price_volume_trend(data):
    pvt = []
    for index in range(data.shape[0]):
        if index > 0:
                last_val = pvt[index - 1]
                last_close = data['close'][index - 1]
                today_close = data['close'][index]
                today_vol = data['volume'][index]
                current_val = last_val + (today_vol * (today_close - last_close) / last_close)
        else:
                current_val = data['volume'][index]
        pvt.append(current_val)
    return data.assign(Price_Volume_Trend=pvt)


def find_up_down(data):
    up_down=[]
    for i in range(1,data.shape[0]):
        if(data['close'].iloc[i]>data['close'].iloc[i-1]):
            up_down.append(str('+1'))
        else:
            up_down.append(str('-1'))
    data=data.iloc[1:]
    return data.assign(Status=up_down)


def CalculateTI(companyname):
    data = load_stock(companyname + ".csv")
    data = calculateSMA(data, 10)
    data = calculateWMA(data)
    data = calculateMomentum(data)
    data = calculateRSI(data)
    data = calculateWilliam(data)
    data = calculateMACD(data)
    data = calculateCCI(data)
    data = acc_dist(data)
    data = calculateAroon(data)
    data = AroonOscillator(data)
    data = calcTrueRange(data)
    data = ppsr(data)
    data = stochastic_oscillator_k(data)
    data = onbv(data)
    data = price_volume_trend(data)
    data = find_up_down(data)
    return data

def splitme(data,df):

    columns=list(df.head(0))
    Training=pd.DataFrame(index=df.index.values,columns=columns)
    Holdout=pd.DataFrame(index=df.index.values,columns=columns)

    j=0

    for k in data:
        pcount = math.ceil(k[1] / 2)
        ncount = math.ceil(k[2] / 2)

        pcount2 = k[1] - pcount
        ncount2 = k[2] - ncount

        for i in range(j, df.shape[0]):

            if (pcount == 0 and ncount == 0 and pcount2 == 0 and ncount2 == 0):
                j = i
                break
            if (pcount != 0 and df['Status'].iloc[i] == '+1'):
                Training.loc[df.index[i]] = df.iloc[i]
                pcount -= 1
            elif (pcount2 != 0 and df['Status'].iloc[i] == '+1'):
                Holdout.loc[df.index[i]] = df.iloc[i]
                pcount2 -= 1
            elif (ncount != 0 and df['Status'].iloc[i] == '-1'):
                Training.loc[df.index[i]] = df.iloc[i]
                ncount -= 1
            elif (ncount2 != 0 and df['Status'].iloc[i] == '-1'):
                Holdout.loc[df.index[i]] = df.iloc[i]
                ncount2 -= 1
            else:
                print('Out of Bounds')

    Training = Training.dropna()
    Holdout=Holdout.dropna()
    Training.to_csv('Training.csv', encoding='utf-8', index=False)
    print("Training written")
    Holdout.to_csv('Holdout.csv',encoding='utf-8',index=False)
    print("Holdout written")


def distri(data):
    data = data.dropna()
    start_year = int(data['date'].iloc[0][:-6])

    value_me = []
    dataf = []
    up_count = 0
    down_count = 0
    for i in range(1, data.shape[0]):

        if (data['close'].iloc[i] > data['close'].iloc[i - 1]):
            up_count += 1
        else:
            down_count += 1

        if (int(data['date'].iloc[i][:-6]) != start_year):
            value_me.append(start_year)
            value_me.append(up_count)
            value_me.append(down_count)
            dataf.append(value_me)
            value_me = []
            up_count = 0
            down_count = 0
            start_year = int(data['date'].iloc[i][:-6])

    value_me = []
    value_me.append(start_year)
    value_me.append(up_count)
    value_me.append(down_count)
    dataf.append(value_me)
    splitme(dataf,data)


def main():
    companyname=input("Enter Company name")
    data=CalculateTI(companyname)
    data.to_csv(companyname + '_modified.csv', encoding='utf-8', index=False)
    distri(data)


    print("Main() Run Successfully")






if __name__=='__main__':
    main()
