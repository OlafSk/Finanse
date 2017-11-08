# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 22:08:04 2017

@author: Olaf
"""

import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt

df = pd.read_csv(r"E:\dane\^spx_d(4).csv")
df.dropna(inplace=True)
df = df.rename(index=str, columns={"Otwarcie":"Open", "Najwyzszy":"High",
   "Najnizszy":"Low", "Zamkniecie":"Close", "Wolumen":"Vol"})
#Momentum measures
df['Momentum'] = talib.MOM(np.array(df['Close']), timeperiod=50)
df['RSI'] = talib.RSI(np.array(df['Close']))
df['APO'] = talib.APO(np.array(df['Close']))
macd, _, _2 = talib.MACD(np.array(df['Close']))
df['MACD'] = macd
df['ROC'] = talib.ROC(np.array(df['Close']))
df['Stoch2'], df['Stoch1'] = talib.STOCH(np.array(df['High']), 
     np.array(df['Low']), np.array(df['Close']))
df['WillR'] = talib.WILLR(np.array(df['High']), np.array(df['Low']),
  np.array(df['Close']))


'''
Zmienne odwolujace sie do ceny, srednio dzialaja gdy cena wchodzi 
na nowe poziomy
df['UpperBB'] , df['MiddleBB'], df['UpperBB'] = talib.BBANDS(np.array(df['Close']))
df['DEMA'] = talib.DEMA(np.array(df['Close']))
df['EMA'] = talib.EMA(np.array(df['Close']))
df['HT_Trend'] = talib.HT_TRENDLINE(np.array(df['Close']))
df['KAMA'] = talib.KAMA(np.array(df['Close']))
df['MA'] = talib.MA(np.array(df['Close']))
df['MAMA'], df['FAMA'] = talib.MAMA(np.array(df['Close']))
#df['MAVP'] = talib.MAVP(np.array(df['Close']),)
df['MidPoint'] = talib.MIDPOINT(np.array(df['Close']))
df['MidPrice'] = talib.MIDPRICE(np.array(df['High']),
                                np.array(df['Low']))
df['Sar'] = talib.SAR(np.array(df['High']), 
                      np.array(df['Low']))
df['SarEXT'] = talib.SAREXT(np.array(df['High']), 
                      np.array(df['Low']))
df['t3'] = talib.T3(np.array(df['Close']))

'''
"""
##Technical paterns
df['Engulf'] =  talib.CDLENGULFING(np.array(df['Open']),
           np.array(df['High']), np.array(df['Low']), 
           np.array(df['Close']))

df['Hammer'] =  talib.CDLHAMMER(np.array(df['Open']),
           np.array(df['High']), np.array(df['Low']), 
           np.array(df['Close']))
df['TwoCrows'] =  talib.CDL2CROWS(np.array(df['Open']),
           np.array(df['High']), np.array(df['Low']), 
           np.array(df['Close']))
df['ThreeBlackCrows'] =  talib.CDL3BLACKCROWS(np.array(df['Open']),
           np.array(df['High']), np.array(df['Low']), 
           np.array(df['Close']))
df['Inside'] =  talib.CDL3INSIDE(np.array(df['Open']),
           np.array(df['High']), np.array(df['Low']), 
           np.array(df['Close']))
df['Outside'] =  talib.CDL3OUTSIDE(np.array(df['Open']),
           np.array(df['High']), np.array(df['Low']), 
           np.array(df['Close']))
df['3StarsInSouth'] =  talib.CDL3STARSINSOUTH(np.array(df['Open']),
           np.array(df['High']), np.array(df['Low']), 
           np.array(df['Close']))
df['WhiteSoldiers'] =  talib.CDL3WHITESOLDIERS(np.array(df['Open']),
           np.array(df['High']), np.array(df['Low']), 
           np.array(df['Close']))
df['AbandondedBaby'] =  talib.CDLABANDONEDBABY(np.array(df['Open']),
           np.array(df['High']), np.array(df['Low']), 
           np.array(df['Close']))
df['AdvancedBlock'] =  talib.CDLADVANCEBLOCK(np.array(df['Open']),
           np.array(df['High']), np.array(df['Low']), 
           np.array(df['Close']))
df['BeltHold'] =  talib.CDLBELTHOLD(np.array(df['Open']),
           np.array(df['High']), np.array(df['Low']), 
           np.array(df['Close']))
df['Breakaway'] =  talib.CDLBREAKAWAY(np.array(df['Open']),
           np.array(df['High']), np.array(df['Low']), 
           np.array(df['Close']))
df['ClosingMarubozu'] =  talib.CDLCLOSINGMARUBOZU(np.array(df['Open']),
           np.array(df['High']), np.array(df['Low']), 
           np.array(df['Close']))
df['ConcealingBabySwallow'] =  talib.CDLCONCEALBABYSWALL(np.array(df['Open']),
           np.array(df['High']), np.array(df['Low']), 
           np.array(df['Close']))
df['CounterAttack'] =  talib.CDLCOUNTERATTACK(np.array(df['Open']),
           np.array(df['High']), np.array(df['Low']), 
           np.array(df['Close']))
df['DarkCloudCover'] =  talib.CDLDARKCLOUDCOVER(np.array(df['Open']),
           np.array(df['High']), np.array(df['Low']), 
           np.array(df['Close']))
df['Doji'] =  talib.CDLDOJI(np.array(df['Open']),
           np.array(df['High']), np.array(df['Low']), 
           np.array(df['Close']))
df['DojiStar'] =  talib.CDLDOJISTAR(np.array(df['Open']),
           np.array(df['High']), np.array(df['Low']), 
           np.array(df['Close']))
df['DragonflyDoji'] =  talib.CDLDRAGONFLYDOJI(np.array(df['Open']),
           np.array(df['High']), np.array(df['Low']), 
           np.array(df['Close']))
df['Englufing'] =  talib.CDLENGULFING(np.array(df['Open']),
           np.array(df['High']), np.array(df['Low']), 
           np.array(df['Close']))
df['EveningDojiStar'] =  talib.CDLEVENINGDOJISTAR(np.array(df['Open']),
           np.array(df['High']), np.array(df['Low']), 
           np.array(df['Close']))
df['EveningStar'] =  talib.CDLEVENINGSTAR(np.array(df['Open']),
           np.array(df['High']), np.array(df['Low']), 
           np.array(df['Close']))
df['GapSideSideWhite'] =  talib.CDLGAPSIDESIDEWHITE(np.array(df['Open']),
           np.array(df['High']), np.array(df['Low']), 
           np.array(df['Close']))
df['GraveStoneDoji'] =  talib.CDLGRAVESTONEDOJI(np.array(df['Open']),
           np.array(df['High']), np.array(df['Low']), 
           np.array(df['Close']))
df['HangingMan'] =  talib.CDLHANGINGMAN(np.array(df['Open']),
           np.array(df['High']), np.array(df['Low']), 
           np.array(df['Close']))

"""

df.to_csv(r'E:\dane\spxTA.csv')
