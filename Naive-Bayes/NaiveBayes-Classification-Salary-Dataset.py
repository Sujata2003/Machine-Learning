# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:08:26 2024

@author: Sujata Mandale
"""
# Problem Statement:
'''Prepare a classification model using the Naive Bayes algorithm for the salary dataset. Train and test datasets are given separately. Use both for model building. '''
################################################################################################################
"""
**Business Problem:

The business problem revolves around predicting the salary of individuals based on certain features or attributes. This prediction can be crucial for various purposes such as hiring decisions, salary negotiations, or understanding the factors that contribute to salary levels within a particular industry or organization.

**Objective:

The objective is to build a classification model using the Naive Bayes algorithm to predict whether an individual's salary falls above or below a certain threshold based on their attributes. By doing so, we aim to provide insights into the factors that significantly influence salary levels and create a tool that can assist in making informed decisions related to salary estimation.

**Constraints:

1. **Data Quality: Ensure the quality and reliability of the dataset, as the model's performance heavily depends on the quality of input data.
2. **Interpretability: The model should be interpretable, allowing stakeholders to understand the factors influencing salary predictions.
3. **Fairness: Ensure that the model does not introduce bias or discrimination based on sensitive attributes such as gender, race, or ethnicity.
4. **Scalability: Design the model in a way that it can be scalable to handle larger datasets or additional features in the future.
5. **Privacy: Protect the privacy of individuals' sensitive information contained within the dataset, adhering to relevant privacy regulations and guidelines.

"""
# Data Dictionary
"""
## Data Dictionary:

1. Age: This column represents the age of individuals living in the suburban locality. Age can be a significant factor in determining demographic trends, economic activity, and housing preferences.

2. Workclass: Workclass refers to the type of employment or work arrangement of individuals, such as private sector, government, self-employed, etc. This variable provides insights into the occupational distribution within the population.

3. Education: Education level attained by individuals, which can include categories such as high school graduate, bachelor's degree, master's degree, etc. Education often correlates with income levels and socio-economic status.

4. Educationno: This likely represents the numerical encoding or level assigned to different levels of education. It could be ordinal values representing the educational attainment of individuals.

5. Maritalstatus: Marital status indicates the marital situation of individuals, such as married, single, divorced, etc. This variable may affect household income, family size, and housing preferences.

6. Occupation: Occupation refers to the type of work or profession individuals are engaged in, such as management, sales, technical, etc. Occupation is closely related to income levels and economic activity.

7. Relationship: Relationship denotes the familial relationship status of individuals, such as husband, wife, own-child, unmarried, etc. This variable provides insights into household composition and dependency ratios.

8. Race: Race represents the racial or ethnic background of individuals, which can influence socio-economic factors, cultural dynamics, and community characteristics.

9. Sex: Sex indicates the gender of individuals, either male or female. Gender demographics may impact workforce participation, income disparities, and housing preferences.

10. Capitalgain: Capital gain refers to the profit earned from the sale of assets or investments, which can include stocks, real estate, etc. This variable reflects individual wealth accumulation and investment behavior.

11. Capitalloss: Capital loss represents the loss incurred from the sale of assets or investments. It complements capital gain and provides insights into financial risk-taking and investment outcomes.

12. Hoursperweek: Hours per week denotes the number of hours individuals work on average per week. This variable influences income levels and employment status.

13. Native: Native likely indicates the place of origin or nationality of individuals. It may provide insights into migration patterns, cultural diversity, and demographic composition.

14. Salary: This column likely indicates the salary or income level of individuals, which is the target variable for prediction in the context of the problem statement.

"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

df_train=pd.read_csv("C:/Users/delll/Desktop\Python/DataSets/SalaryData_Train.xls")
df_test=pd.read_csv("C:/Users/delll/Desktop\Python/DataSets/SalaryData_Test.xls")

df_train.shape
#there are 30161 rows and 14 columns

df_test.shape
#rows=15060 and columns=14

df_train.columns
"""
['age', 'workclass', 'education', 'educationno', 'maritalstatus',
 'occupation', 'relationship', 'race', 'sex', 'capitalgain',
  'capitalloss', 'hoursperweek', 'native', 'Salary']
"""
df_train.dtypes
df_train.info
df_train.isnull().sum()
#there are no null values in the dataset

df_train.describe()

# draw boxplot to check outliers
i=1
plt.figure(figsize=(16,13))
for col in df_train.columns:
    plt.subplot(3,4,i)
    sns.boxplot(df_train[col])
    plt.title(col)
    i=i+1
# All numerical columns have outliers
#remove all outliers
from feature_engine.outliers import Winsorizer
cols=['age', 'educationno', 'hoursperweek']
for col in cols:
    winsor=Winsorizer(capping_method="iqr",fold=1.5,variables=col,tail="both")
    df_train[col]=winsor.fit_transform(df_train[[col]])
i=1
plt.figure(figsize=(16,13))
for col in df_train.columns:
    plt.subplot(3,4,i)
    sns.boxplot(df_train[col])
    plt.title(col)
    i=i+1    

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
