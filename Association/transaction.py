# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 23:29:23 2023

@author: delll
"""

from mlxtend.frequent_patterns import apriori,association_rules
import pandas as pd

tranction=pd.read_csv(r"C:\Users\delll\Desktop\Python\DataSets\transactions_retail1.xls")
tranction.isnull().sum()
tranction['HEART'].fillna(0)
# Convert the data into a binary format
binary_data = tranction.applymap(lambda x: 1 if pd.notna(x) else 0)

# Explore the preprocessed data
print(binary_data.head())

from mlxtend.frequent_patterns import apriori, association_rules

# Find frequent itemsets using Apriori
frequent_itemsets = apriori(binary_data, min_support=0.2, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Explore the generated rules
print(rules)
#Based on the association rules, you can provide recommendations 
#for product placement on shelves. For example, 
#if products A and B are frequently bought together, 
#consider placing them close to each other on the shelves.