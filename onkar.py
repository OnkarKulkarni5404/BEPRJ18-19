import pandas as pd
import os
import math
import finta as finta

def load_stock(companyname=""):
    csv_path = os.path.join('individual_stocks_5yr',companyname)
    return pd.read_csv(csv_path)





def find_up_down(data):
    up_down=[]
    for i in range(1,data.shape[0]):
        if(data['close'].iloc[i]>data['close'].iloc[i-1]):
            up_down.append(str('+1'))
        else:
            up_down.append(str('-1'))
    data=data.iloc[1:]
    return data.assign(y=up_down)


def CalculateTI(companyname):
    data = pd.read_csv(companyname + ".csv")
    data = data.dropna()
    obj=finta.TA

    data=data.assign(SMA=obj.SMA(data,10))
    data=data.assign(EMA=obj.EMA(data,10))
    data = data.assign(RSI=obj.RSI(data,14))
    data=data.assign(CCI=obj.CCI(data,20))
    data=data.assign(KAMA=obj.KAMA(data))
    data=data.assign(WMA=obj.WMA(data))
    data=data.assign(HMA=obj.HMA(data))

    result=obj.MACD(data)
    data=data.assign(MACD=result['MACD'])
    data=data.assign(MOM=obj.MOM(data))
    data=data.assign(ATR=obj.ATR(data))
    data = data.assign(STOCH=obj.STOCH(data))
    data = data.assign(STOCHD=obj.STOCHD(data))
    data = data.assign(STOCHRSI=obj.STOCHRSI(data))
    data = data.assign(WILLIAMS=obj.WILLIAMS(data))
    data = data.assign(TSI=obj.TSI(data)['TSI'])
    data = data.assign(ADL=obj.ADL(data))
    data = data.assign(CHAIKIN=obj.CHAIKIN(data))
    data = data.assign(OBV=obj.OBV(data))
    data = data.assign(CFI=obj.CFI(data))
    data=data.assign(CMO=obj.CMO(data))
    data=data.assign(VPT=obj.VPT(data))

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
            if (pcount != 0 and df['y'].iloc[i] == '+1'):
                Training.loc[df.index[i]] = df.iloc[i]
                pcount -= 1
            elif (pcount2 != 0 and df['y'].iloc[i] == '+1'):
                Holdout.loc[df.index[i]] = df.iloc[i]
                pcount2 -= 1
            elif (ncount != 0 and df['y'].iloc[i] == '-1'):
                Training.loc[df.index[i]] = df.iloc[i]
                ncount -= 1
            elif (ncount2 != 0 and df['y'].iloc[i] == '-1'):
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


#--------------------------------------------------------------------------------------------------------------------------------------------------------------

def main(companyname):
    #companyname=input("Enter Company name")
    data=CalculateTI(companyname)
    data=find_up_down(data)
    distri(data)
    data.to_csv(companyname + '_modified.csv', encoding='utf-8', index=False)
    print("Main() Run Successfully")



































if __name__=='__main__':
    main()
