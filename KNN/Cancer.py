# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:59:31 2024

@author: delll
"""

import pandas as pd
import numpy as np
wbcd=pd.read_csv("C:/Users/delll/Desktop/Python/DataSets/wbcd.xls")
wbcd.head()
wbcd.shape
#row=569 columns=31
wbcd.info()

wbcd.describe()

#replace  B with Beniegn and M with Maligant
wbcd["diagnosis"]=np.where(wbcd["diagnosis"]=="B","Beniegn",wbcd["diagnosis"])
wbcd["diagnosis"]=np.where(wbcd["diagnosis"]=="M","Maligant",wbcd["diagnosis"])

wbcd=wbcd.iloc[:,1:32]

#Normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

wbcd_n=norm_func(wbcd.iloc[:,1:32])

#Now x is input and y as output
x=np.array(wbcd_n.iloc[:,:])
y=np.array(wbcd["diagnosis"])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#Here you are passing x and y instead of dataframe
#there would be chances of unbalanceing of data
#let us assume that you have 100 data points
#out of which 80 NC and 20 cancerous
#so this data points must be equally distributed
#so we used stratified sampling

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=21)
knn.fit(x_train,y_train)
pred=knn.predict(x_test)
pred
#let us check the applicability of the model
#miss classsification,Actual patient is malignant
#cance patient but predicted is beneign 1
#actual patient is beneign and predicted as cancer patient is 5
#hence this model is  not acceptable

###############################################################
#now evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(pred,y_test))
pd.crosstab(pred, y_test)

#let us try to select correct value of k
acc=[]
for i in range(3,50,2):
    neigh=KNeighborsClassifier(n_neighbors=i)
    neigh.fit(x_train, y_train)
    train_acc=np.mean(neigh.predict(x_train)==y_train)
    test_acc=np.mean(neigh.predict(x_test)==y_test)
    acc.append([train_acc,test_acc])
#if you will see the acc,it has got two accuracy,i[0]-train_acc
#i[1]=test_acc

import matplotlib.pyplot as plt
plt.plot(np.arange(3,50,2),[i[0] for i in acc],'ro-')
plt.plot(np.arange(3,50,2),[i[1] for i in acc],'bo-')
#there are 3,5,7,9 are possible values where accurancy is good
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)
pred=knn.predict(x_test)
pred
accuracy_score(pred,y_test)
pd.crosstab(pred, y_test)






