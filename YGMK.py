# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 22:23:10 2020

@author: Lenovo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
#veri yükleme
veriler=pd.read_csv('Kitap1.csv')
print(veriler)

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN', strategy='mean', axis=0)

tamamlanmisveriler=veriler.iloc[:,1:].values

imputer=imputer.fit(tamamlanmisveriler[:,:])

tamamlanmisveriler[:,:]=imputer.transform(tamamlanmisveriler[:,:])


tamamlanmisveriler= pd.DataFrame(data=tamamlanmisveriler,  columns=['PM10','SO2','CO','NO2','O3'])

tarih=veriler.iloc[:,0:1].values

tarih=pd.DataFrame(data=tarih,columns=['Tarih'])

dataframe=pd.concat([tarih,tamamlanmisveriler],axis=1)

tamamlanmisveriler= pd.DataFrame(data=tamamlanmisveriler,  columns=['PM10','SO2','CO','NO2','O3'])



plt.figure(figsize=(25,15))
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.PM10, color="black", linestyle="--", alpha=0.9)
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.SO2, color="green", linestyle="--", alpha=0.9)
plt.xlabel("Zaman")
plt.ylabel("Ölçüm Parametreleri:PM10, SO2")
plt.show()


plt.figure(figsize=(25,15))
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.PM10, color="red", linestyle="--", alpha=0.9)
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.NO2, color="grey", linestyle="--", alpha=0.9)
plt.xlabel("Zaman")
plt.ylabel("Ölçüm Parametreleri:PM10, NO2")
plt.show()

plt.figure(figsize=(25,15))
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.PM10, color="brown", linestyle="--", alpha=0.9)
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.O3, color="yellow", linestyle="--", alpha=0.9)
plt.xlabel("Zaman")
plt.ylabel("Ölçüm Parametreleri:PM10, O3")
plt.show()

plt.figure(figsize=(25,15))
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.PM10, color="red", linestyle="--", alpha=0.9)
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.CO, color="black", linestyle="--", alpha=0.9)
plt.xlabel("Zaman")
plt.ylabel("Ölçüm Parametreleri:PM10, CO")
plt.show()


plt.figure(figsize=(25,15))
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.NO2, color="orange", linestyle="--", alpha=0.9)
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.SO2, color="green", linestyle="--", alpha=0.9)
plt.xlabel("Zaman")
plt.ylabel("Ölçüm Parametreleri:NO2, SO2")
plt.show()


plt.figure(figsize=(25,15))
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.NO2, color="red", linestyle="--", alpha=0.9)
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.O3, color="green", linestyle="--", alpha=0.9)
plt.xlabel("Zaman")
plt.ylabel("Ölçüm Parametreleri:NO2, O3")
plt.show()

plt.figure(figsize=(25,15))
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.NO2, color="black", linestyle="--", alpha=0.9)
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.CO, color="red", linestyle="--", alpha=0.9)
plt.xlabel("Zaman")
plt.ylabel("Ölçüm Parametreleri:NO2, CO")
plt.show()


plt.figure(figsize=(25,15))
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.CO, color="blue", linestyle="--", alpha=0.9)
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.SO2, color="grey", linestyle="--", alpha=0.9)
plt.xlabel("Zaman")
plt.ylabel("Ölçüm Parametreleri:CO,SO2")
plt.show()

plt.figure(figsize=(25,15))
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.CO, color="yellow", linestyle="--", alpha=0.9)
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.O3, color="black", linestyle="--", alpha=0.9)
plt.xlabel("Zaman")
plt.ylabel("Ölçüm Parametreleri:CO, O3")
plt.show()


PM10=tamamlanmisveriler[['PM10']]
SO2=tamamlanmisveriler[['SO2']]
CO=tamamlanmisveriler[['CO']]
NO2=tamamlanmisveriler[['NO2']]
O3=tamamlanmisveriler[['O3']]
#ÇOKLU DOGRUSAL REGRESYON

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

cokludogrusal=pd.concat([NO2,PM10], axis=1)
x_train, x_test, y_train, y_test=train_test_split(cokludogrusal, O3, test_size=0.3, random_state=0)
cd_reg=LinearRegression()
cd_reg.fit(x_train, y_train)
O3_cd_reg=cd_reg.predict(x_test)



#Karar Ağacı
from sklearn.tree import DecisionTreeRegressor
kararagacı=DecisionTreeRegressor(random_state=10)
kararagacı.fit(NO2,O3)
O3_kararagacı=kararagacı.predict(O3)

cokludogrusalBasari=(r2_score(y_test, O3_cd_reg ))
kararagacıBasari=(r2_score(O3, O3_kararagacı ))

