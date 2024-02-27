# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 13:44:19 2023

@author: delll
"""

import pandas as pd

movies=pd.read_csv(r'C:\Users\delll\Desktop\Python\DataSets\Entertainment.xls',encoding='utf8')

movies.shape
movies.columns
movies.Category

from sklearn.feature_extraction.text import TfidfVectorizer
#this is term freq inverse document
#each row treated as document
tfidf=TfidfVectorizer(stop_words='english')
#it is going to create TfidfVectorizer to separate all stop words
#it is going to seperate all words from rows
#now check for null values
movies['Category'].isnull().sum()
tfidf_matrix=tfidf.fit_transform(movies['Category'])
tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel
cosine_sim_matrix=linear_kernel(tfidf_matrix,tfidf_matrix)
movies_index=pd.Series(movies.index,index=movies['Titles']).drop_duplicates()
movies_id=movies_index['Toy Story (1995)']
movies_id

def get_recommandations(Name,topN):
    topN=10
    #Name='Assassins (1995)'
    movies_id=movies_index[Name]
    #we want to capture whole row of given movie
    #name ,its score and column id
    #for that ,apply cosine_sim_matrix to enumerte()
    #enumerate function create a object,
    #which we need to create in list format
    #we are using enumerate()
    #it does,suppose we have given (2,10,15,18) if we apply to enumerate() then it will
    #crete a list[(0,2),(1,10),(2,15)..]
    cosine_scores=list(enumerate(cosine_sim_matrix[movies_id]))
    #the cosine score captured ,we want to arrage in descending order
    #we want recommand top 10 movie based on highest similarity
    #i.e score .If we check cosine score,it compries ofindex:cosine score
    #x[0]index and x[1]=cosine score
    #arrage tuples in descending order of the score
    cosine_scores=sorted(cosine_scores,key=lambda x:x[1],reverse=True)
    #get the score of top N most similar movies
    #to compare top n movies ,you need to give topN+1
    cosine_scores_N=cosine_scores[0:topN+1]
    #getting the movie index
    movies_idx=[i[0] for i in cosine_scores_N]
    #getting cosine score
    movies_scores=[i[1] for i in cosine_scores_N]
    #we use this info to create a dataframe
    movies_similar_show=pd.DataFrame(columns=['name','score'])
    #assin anime_idx to name column
    movies_similar_show['name']=movies.loc[movies_idx,'Titles']
    #assign score to score col
    movies_similar_show['score']=movies_scores
    #while assigning values,it is by default capturing original
    #index of movie
    #we want to reset the index
    movies_similar_show.reset_index(inplace=True)
    print(movies_similar_show)
    
get_recommandations('Toy Story (1995)', 10)
