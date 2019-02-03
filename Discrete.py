import pandas as pd
import math

def D_SMA(data,df):
    l=[]
    for i in range(0,data.shape[0]):
        if(data['close'].iloc[i]>data['SMA'].iloc[i]):
            l.append(+1)
        else:
            l.append(-1)
    df=df.assign(SMA=l)
    return df

def D_WMA(data,df):
    l = []
    for i in range(0, data.shape[0]):
        if (data['close'].iloc[i] > data['WMA'].iloc[i]):
            l.append(+1)
        else:
            l.append(-1)
    df = df.assign(WMA=l)
    return df

def D_MOMENT(data,df):
    l=[]
    for i in range(0, data.shape[0]):
        if (data['Momentum'].iloc[i] > 1):
            l.append(+1)
        else:
            l.append(-1)
    df = df.assign(Momentum=l)
    return df

def D_RSI(data,df):
    return df

def William(data,df):
    l=[]
    l.append(1)
    for i in range(1, data.shape[0]):
        if (data['WilliamK'].iloc[i] > data['WilliamK'].iloc[i-1]):
            l.append(+1)
        else:
            l.append(-1)
    df = df.assign(WilliamK=l)
    return df

def D_MACD(data,df):
    l = []
    l.append(1)
    for i in range(1, data.shape[0]):
        if (data['MACD'].iloc[i] > data['MACD'].iloc[i - 1]):
            l.append(+1)
        else:
            l.append(-1)
    df = df.assign(MACD=l)
    return df

def CCI(data,df):
    return df

def AD(data,df):
    l = []
    l.append(1)
    for i in range(1, data.shape[0]):
        if (data['acc_dist'].iloc[i] > data['acc_dist'].iloc[i - 1]):
            l.append(+1)
        else:
            l.append(-1)
    df = df.assign(AD=l)
    return df



def AD_21(data,df):
    l = []
    l.append(1)
    for i in range(1, data.shape[0]):
        if (data['AD-:21'].iloc[i] > data['AD-:21'].iloc[i - 1]):
            l.append(+1)
        else:
            l.append(-1)
    df = df.assign(AD=l)
    return df

def AROON_OSCI(data,df):
    l = []
    for i in range(0, data.shape[0]):
        if (data['AroonOscillator'].iloc[i] > 1):
            l.append(+1)
        else:
            l.append(-1)
    df = df.assign(AroonOscillator=l)
    return df





def main():
    data=pd.read_csv('Training.csv')
    df=pd.DataFrame(index=data.index.values,columns=data.head(0))
    df=D_SMA(data,df)
    df=D_WMA(data,df)
    df=D_MOMENT(data,df)
    df=D_RSI(data,df)
    df=William(data,df)
    df=D_MACD(data,df)
    df=CCI(data,df)
    df=AD(data,df)
    df=AD_21(data,df)
    df=AROON_OSCI(data,df)



    df.to_csv('Tesst.csv', encoding='utf-8', index=False)
    print("Run_Successful")









if __name__ == '__main__':
    main()
