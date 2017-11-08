# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 18:28:59 2017

@author: Olaf
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from keras.optimizers import RMSprop


import matplotlib.pyplot as plt
df = pd.read_csv(r"E:\dane\spxTA.csv", index_col=0)
nb_back = 5
df_lol = pd.DataFrame()
def prepareData(link, nb_back = 10, thresh = 15, day_shift = 1, pca_n=5):
    df = pd.read_csv(link, index_col = 0)
    df.dropna(inplace=True)
    df = df.rename(index=str, columns={"Otwarcie":"Open", "Najwyzszy":"High",
        "Najnizszy":"Low", "Zamkniecie":"Close", "Wolumen":"Vol"})
    df.drop('Data', axis=1, inplace=True)
    df['5DayReturn'] = df['Close'].shift(-day_shift)
    df['5DayReturn'] = df['5DayReturn'] - df['Open']
    df['Over0'] = df['5DayReturn'] > thresh
    df.dropna(inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    y = df['Over0'].shift(-1)
    y = y[nb_back:-1]
    y_reg = df['5DayReturn'].shift(-1)
    y_reg = y_reg[nb_back:-1]
    y = y.values.reshape(-1,1)
    scY = MinMaxScaler()
    y = scY.fit_transform(y)
    df['LowerShadow'] = df.loc[:,['Close','Open']].min(axis=1) - df['Low']
    df['1DReturn'] = df['Open'] - df['Close']
    df['UpperShadow'] = df['High'] - df.loc[:,['Close','Open']].max(axis=1)
    df.drop(['High','Low','Open','Close','5DayReturn','Over0'], axis=1, inplace=True)
    
    sc = MinMaxScaler()
    df = sc.fit_transform(df)
    pca = PCA(n_components = 5)
    df = pca.fit_transform(df)
    print(pca.explained_variance_ratio_)
    df = pd.DataFrame(df)
    df.columns = ['a','b','c','d','e']
    for col in df.columns:
        for i in range(1, nb_back + 1):
            df[col + '_' + str(i)] = df[col].shift(i)
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)
    df = df.loc[:len(df)-2,:]
    df_lol = df.loc[4300:,:]
    df_train = df.loc[:4500,:]
    df_test = df.loc[4501:,:]
    sc = MinMaxScaler()
    X_train = sc.fit_transform(df_train)
    X_test = sc.transform(df_test)
    y_train = y[:4501]
    y_test = y[4501:]
    y_reg = y_reg[4501:]
    X_train_n = np.zeros((len(y_train), 10,pca_n))
    for j in range(len(y_train)):
        for k in range(0,pca_n):
            for i in range(1,nb_back):
                X_train_n[j, 0, k] = X_train[j,k]
                X_train_n[j, i, k] = X_train[j, pca_n-1 + k * nb_back + i]
    X_test_n = np.zeros((len(y_test), 10,pca_n))
    for j in range(len(y_test)):
        for k in range(0,pca_n):
            for i in range(1,nb_back):
                X_test_n[j, 0, k] = X_test[j,k]
                X_test_n[j, i, k] = X_test[j, pca_n - 1 + k * nb_back + i]
    return X_train_n, X_test_n, y_train, y_test, scY, y_reg
X_train_n, X_test_n, y_train, y_test, sc, y_reg = prepareData(r"E:\dane\spxTA.csv",
                                                  nb_back = 10, thresh = 20, day_shift = 2)


regressor = Sequential()
#perfecto parametry 1 lstm 8 units  listm 4  wtedy roc_auc 65, najlepszy randomseed loss okol0 0.37
regressor.add(LSTM(units = 6, activation="relu", input_shape = (10, 5), return_sequences=False))
#regressor.add(Dropout(0.2))

#regressor.add(LSTM(units = 4, activation='relu', return_sequences=True))

#regressor.add(LSTM(units = 4, activation='relu'))
#regressor.add(Dropout(0.5))
regressor.add(Dense(units = 6, activation="relu"))
#regressor.add(Dropout(0.2))
#regressor.add(Dense(units = 4, activation="relu"))

regressor.add(BatchNormalization())
regressor.add(Dense(units=1, activation="linear"))
#rms = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
#z AdaGradem roc_auc 66

regressor.compile(optimizer="adam", loss='binary_crossentropy')
regressor.summary()
#cb = [EarlyStopping(monitor='', min_delta=0.001, patience=80, verbose=0, mode='auto')]
test_R = []
train_R = []
for i in range(20):
    fit = regressor.fit(X_train_n, y_train, batch_size=500, epochs=10,    
                        validation_data=(X_test_n, y_test))
    y_pred = regressor.predict(X_test_n)
    test_R.append(roc_auc_score(y_test, y_pred) * 2 - 1)
    y_pred_t = regressor.predict(X_train_n)
    train_R.append(roc_auc_score(y_train, y_pred_t) * 2 - 1)
    print(i)

plt.plot(test_R)
plt.plot(train_R)
test_R
y_test.mean()
max(y_test)
y_test.mean()
fpr, tpr, _ = roc_curve(y_test,y_pred)
_
plt.plot(fpr,tpr)

y_reg = np.array(y_reg)
y_reg[y_pred.reshape(-1) > _[30]].mean()
y_reg.mean()
y_reg[3]


y_reg[(y_pred > _[40]).reshape(-1)].mean()


len(y_pred > _[10])
threshold = 0.2
len(y_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred >  _[-10])

y_pred > _[10] 
y_test.mean()
#print(fit.history.keys())
#regressor.save("LSTM_model_SPX500.h5")

#model = load_model("LSTM_model_SPX500.h5")

#plt.plot(fit.history['val_acc'])
#plt.plot(fit.history['acc'])
#plt.show()
plt.plot(fit.history['loss'])
plt.plot(fit.history['val_loss'])
plt.show()
y_pred_class = regressor.predict_classes(X_test)       
m = confusion_matrix(y_test,y_pred_class)
print(m)
y_pred = regressor.predict(X_test) 
x, z, t = roc_curve(y_test, y_pred)
plt.plot(x,z)
plt.show()

y_pred = y_pred[:-1]
plt.figure(figsize=(10,10))
#plt.show()

'''
     df_lol['5DayReturn'] = df_lol['Close'].shift(-3)
     df_lol['5DayReturn'] = df_lol['5DayReturn'] - df_lol['Open']
     df_lol.reset_index(inplace=True, drop=True)
     df_lol2 = df_lol.loc[:,['Open','Close', '5DayReturn']]
     y_true = df_lol['5DayReturn']
     
     y_pred = y_pred > 0.50
     y_pred = y_pred[:-3]
     y_true = y_true[:-3]
     y_true = y_true.reshape(-1,)
     y_pred = y_pred.reshape(-1,)
     np.dot(y_true,y_pred)
     print(confusion_matrix(y_test, y_pred))
     w = y_true.dot(y_true)
     a = np.array([1,2,3])
'''