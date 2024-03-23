# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 23:14:40 2024

@author: delll
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
df=pd.read_csv("C:/Users/delll/Desktop\Python/DataSets/Disaster_tweets_NB.xls")
df.head()
df.columns

df.drop("id",axis=1,inplace=True)
df.head()
df.isnull().sum()
df.fillna(df.location.mode(),inplace=True)
import re
def cleaning_text(i):
    w=[]
    i=re.sub("{^A-Za-z""}+"," ",i).lower()
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

cleaning_text("All residents asked to 'shelter in place' are being notified by officers. No other evacuation or shelter in place orders are expected")

df.text=df.text.apply(cleaning_text)
df.head()

df=df.loc[df.text!=" ",:]

from sklearn.model_selection import train_test_split
df_train,df_test=train_test_split(df,test_size=0.2)

def split_into_words(i):
    return [word for word in i.split(" ")]

df_bow=CountVectorizer(analyzer=split_into_words).fit(df.text)
all_df_matrix=df_bow.transform(df.text)

train_df_matrix=df_bow.transform(df.text)
test_df_matrix=df_bow.transform(df_test.text)

tfidf_Transformer=TfidfTransformer().fit(all_df_matrix)
train_tfidf=tfidf_Transformer.transform(train_df_matrix)
test_tfidf=tfidf_Transformer.transform(test_df_matrix)

test_tfidf.shape


from sklearn.naive_bayes import MultinomialNB as MB
classifer_mb=MB()
classifer_mb.fit(train_tfidf,df_train.target)

test_pred_m=classifer_mb.predict(test_tfidf)
accuracy_test_m=np.mean(test_pred_m==df_test.target)
accuracy_test_m















