# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 22:57:15 2024

@author: delll
"""

import numpy as np
import pandas as pd


df=pd.read_csv("C:/Users/delll/Desktop\Python/DataSets/NB_Car_Ad.xls")
df.head()

df.shape
df.isnull().sum()

target=df.Purchased
inputs=df.drop("Purchased",axis=1)
inputs.drop("User ID",axis=1,inplace=True)

dummies=pd.get_dummies(inputs.Gender)
dummies.head()

inputs=pd.concat([inputs,dummies],axis="columns")
inputs.head()

inputs.drop("Gender",axis="columns",inplace=True)
inputs.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inputs,target,test_size=0.2)

from sklearn.naive_bayes import MultinomialNB as MB
model=MB()

model.fit(x_train,y_train)

model.score(x_test,y_test)
