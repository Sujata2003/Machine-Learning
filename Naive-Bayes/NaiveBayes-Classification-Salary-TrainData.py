# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 22:19:03 2024

@author: delll
"""
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

df_train=pd.read_csv("C:/Users/delll/Desktop\Python/DataSets/SalaryData_Train.xls")
df_test=pd.read_csv("C:/Users/delll/Desktop\Python/DataSets/SalaryData_Test.xls")

df_train.head()
df_train.shape
df_train.columns
df_train.info()
df_train.isnull().sum()

df_train.describe()

col=["educationno","maritalstatus","relationship","race","native"]
df_train.drop(col,axis=1,inplace=True)
df_train.head()

sns.heatmap(df_train.corr(),cmap='coolwarm',annot=True,fmt='.2f')
plt.show()

df_bow=CountVectorizer().fit(df_train)
df_matrix=df_bow.transform(df_train)

train_df_matrix=df_bow.transform(df_train)
test_df_matrix=df_bow.transform(df_test)

tfidf_Transformer=TfidfTransformer().fit(df_matrix)
train_tfidf=tfidf_Transformer.transform(train_df_matrix)
test_tfidf=tfidf_Transformer.transform(test_df_matrix)

test_tfidf.shape

from sklearn.naive_bayes import MultinomialNB as MB
classifer_mb=MB()
classifer_mb.fit(train_tfidf,df_train.Salary)

test_pred_m=classifer_mb.predict(test_tfidf)
accuracy_test_m=np.mean(test_pred_m==df_test.Salary)
accuracy_test_m
