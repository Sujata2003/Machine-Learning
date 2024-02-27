# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:08:26 2024

@author: delll
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

df_train=pd.read_csv("C:/Users/delll/Desktop\Python/DataSets/SalaryData_Train.xls")
df_test=pd.read_csv("C:/Users/delll/Desktop\Python/DataSets/SalaryData_Test.xls")

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
