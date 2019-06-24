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
        if (data['MOM'].iloc[i] > 1):
            l.append(+1)
        else:
            l.append(-1)
    df = df.assign(Momentum=l)
    return df

def D_RSI(data,df):
    l=[]
    for i in range(0,data.shape[0]):
        if(data['RSI'].iloc[i]>70):
            l.append(-1)
        elif(data['RSI'].iloc[i]<30):
            l.append(1)
        else:
            if(i!=0):
                if(data['RSI'].iloc[i] > data['RSI'].iloc[i-1]):
                    l.append(1)
                else:
                    l.append(-1)
            else:
                l.append(-1)

    df = df.assign(RSI=l)
    return df

def William(data,df):
    l=[]
    l.append(1)
    for i in range(1, data.shape[0]):
        if (data['WILLIAMS'].iloc[i] > data['WILLIAMS'].iloc[i-1]):
            l.append(+1)
        else:
            l.append(-1)
    df = df.assign(WILLIAMS=l)
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
        if (data['ADL'].iloc[i] > data['ADL'].iloc[i - 1]):
            l.append(+1)
        else:
            l.append(-1)
    df = df.assign(AD=l)
    return df

def KAMA(data,df):
    l=[]
    l.append(1)
    for i in range(1, data.shape[0]):
        if (data['KAMA'].iloc[i] > data['KAMA'].iloc[i - 1]):
            l.append(+1)
        else:
            l.append(-1)
    df = df.assign(KAMA=l)
    return df


def EMA(data,df):
    l = []
    for i in range(0, data.shape[0]):
        if (data['EMA'].iloc[i] < data['close'].iloc[i]):
            l.append(-1)
        else:
            l.append(+1)
    df = df.assign(EMA=l)
    return df

def HMA(data,df):
    l = []
    for i in range(0, data.shape[0]):
        if (data['HMA'].iloc[i] < data['close'].iloc[i]):
            l.append(-1)
        else:
            l.append(+1)
    df = df.assign(HMA=l)
    return df

def ATR(data,df):
    l=[]
    l.append(1)
    for i in range(1,data.shape[0]):
        if (data['ATR'].iloc[i] < data['ATR'].iloc[i-1]):
            l.append(-1)
        else:
            l.append(+1)
    df = df.assign(ATR=l)
    return df

def stoh(data,df):
    l = []
    l.append(1)
    for i in range(1, data.shape[0]):
        if (data['STOCH'].iloc[i]>75):
            l.append(-1)
        elif(data['STOCH'].iloc[i]<25):
            l.append(+1)
        else:
            if(i!=0):
                l.append(data['STOCH'].iloc[i-1])
    df = df.assign(STOCH=l)

def stoD(data,df):
    l = []
    l.append(1)
    for i in range(1, data.shape[0]):
        if (data['STOCHD'].iloc[i]>75):
            l.append(-1)
        elif(data['STOCHD'].iloc[i]<25):
            l.append(+1)
        else:
            if(i!=0):
                l.append(data['STOCHD'].iloc[i-1])
            else:
                l.append(1)
    df = df.assign(STOCHD=l)

def stoRSI(data,df):
    l = []
    l.append(1)
    for i in range(1, data.shape[0]):
        if (data['STOCHRSI'].iloc[i]>75):
            l.append(-1)
        elif(data['STOCHRSI'].iloc[i]<25):
            l.append(+1)
        else:
            if(i!=0):
                l.append(data['STOCHRSI'].iloc[i-1])
            else:
                l.append(1)
    df = df.assign(STOCHRSI=l)

def TSI(data,df):
    l=[]
    for i in range(0,data.shape[0]):
        if(data['TSI']<30):
            l.append(-1)
        elif (data['TSI'].iloc[i] >30):
            l.append(+1)
        else:
            if (i != 0):
                l.append(data['TSI'].iloc[i - 1])
            else:
                l.append(1)
        df = df.assign(TSI=l)





def main():
    data=pd.read_csv('Training.csv')
    df=pd.DataFrame(index=data.index.values,columns=data.head(0))
    df=df.assign(y=data['y'])
    df=D_SMA(data,df)
    df=D_WMA(data,df)
    df=D_MOMENT(data,df)
    df=D_RSI(data,df)
    df=William(data,df)
    df=D_MACD(data,df)
    df=CCI(data,df)
    df=AD(data,df)
    df=KAMA(data,df)
    df=EMA(data,df)
    df=HMA(data,df)
    df=ATR(data,df)


    df.to_csv('Training.csv', encoding='utf-8', index=False)
    print("Run_Successful")









if __name__ == '__main__':
    main()
