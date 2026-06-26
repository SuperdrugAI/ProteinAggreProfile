# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:23:56 2023

@author: yy
"""

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# Load data
data = pd.read_csv(r'C:\Users\81036\Desktop\1.csv', encoding='gb2312')

# Separate features and target
X = data.iloc[:,1:]
y = data.iloc[:, 0].values 

# Perform feature selection
selector = SelectKBest(score_func= f_classif, k=3)
selector.fit(X, y)

# Get scores for each feature
scores = selector.scores_

# Create a DataFrame to hold feature names and scores
df_scores = pd.DataFrame({
    'Feature': X.columns,
    'Score': scores
})

# Write DataFrame to Excel
df_scores.to_excel(r'C:\Users\81036\Desktop\feature_scores.xlsx', index=False)

# Print scores
print(df_scores)
