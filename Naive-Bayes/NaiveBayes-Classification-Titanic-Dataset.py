# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 21:47:09 2024

@author: delll
"""

import pandas as pd

df=pd.read_csv("C:/Users/delll/Desktop/Python/DataSets/titanic.csv")
df.head()

df.drop(["PassengerId","Name","SibSp","Parch","Ticket","Cabin","Embarked"],axis="columns",inplace=True)
df.head()

target=df.Survived
inputs=df.drop("Survived",axis="columns")

dummies=pd.get_dummies(inputs.Sex)
dummies.head()

inputs=pd.concat([inputs,dummies],axis="columns")
inputs.head()

inputs.drop("Sex",axis="columns",inplace=True)
inputs.head()

inputs.Age.isna().sum()
inputs.isna().sum()
inputs.Age=inputs.Age.fillna(inputs.Age.mean())
inputs.Fare=inputs.Fare.fillna(inputs.Fare.mean())

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inputs,target,test_size=0.2)

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()

model.fit(x_train,y_train)

model.score(x_test, y_test)

model.predict(x_test[:10])

from sklearn.naive_bayes import MultinomialNB as MB
model=MB()

model.fit(x_train,y_train)

model.score(x_test, y_test)

model.predict(x_test[:10])
