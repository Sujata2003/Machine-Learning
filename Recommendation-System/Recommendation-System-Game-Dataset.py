# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 22:10:55 2023

@author: delll
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
data=pd.read_csv(r'C:\Users\delll\Desktop\Python\DataSets\game.csv.xls',encoding='utf8')



# Assuming your data has columns: 'user_id', 'game_title', 'rating'
# Adjust the column names as per your dataset
 # Replace 'your_dataset.csv' with your actual file

# Create a user-item matrix
user_item_matrix = data.pivot_table(index='userId', columns='game', values='rating', fill_value=0)

# Calculate cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)

# Convert the similarity matrix to a DataFrame for easier handling
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Define a function to get top N recommendations for a user
def get_top_recommendations(user_id, top_n=5):
    user_ratings = user_item_matrix.loc[user_id]
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]  # Exclude the user itself
    unrated_games = user_ratings[user_ratings == 0].index  # Games not rated by the user

    recommendations = []

    for game in unrated_games:
        weighted_sum = 0
        similarity_sum = 0

        for similar_user in similar_users:
            if user_item_matrix.loc[similar_user, game] != 0:  # Exclude games not rated by the similar user
                weighted_sum += user_similarity_df.loc[user_id, similar_user] * user_item_matrix.loc[similar_user, game]
                similarity_sum += abs(user_similarity_df.loc[user_id, similar_user])

        if similarity_sum != 0:
            predicted_rating = weighted_sum / similarity_sum
            recommendations.append((game, predicted_rating))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:top_n]

# Example: Get top recommendations for a user
target_user_id = 6
top_recommendations = get_top_recommendations(target_user_id, top_n=5)

print("Top recommendations for user {}: {}".format(target_user_id, top_recommendations))
