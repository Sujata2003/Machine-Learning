# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 23:26:18 2023

@author: delll
"""

from mlxtend.frequent_patterns import apriori,association_rules
import pandas as pd

phone=pd.read_csv(r"C:\Users\delll\Desktop\Python\DataSets\myphonedata.xls")
frequent_itemset=apriori(phone,min_support=0.075,max_len=4,use_colnames=True)
frequent_itemset.sort_values('support',ascending=False,inplace=True)
rules=association_rules(frequent_itemset,metric='lift',min_threshold=1)
rules.head(20)
rules.sort_values('lift',ascending=False).head(10)
