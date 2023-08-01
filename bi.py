import numpy as np
import pandas as pd
import seaborn as ans
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
data = pd.read_csv("data/data.csv")

data = data.drop_duplicates(subset=['name'])
data = data.sort_values(by=['popularity'], ascending=False).head(10000)
for i, row in data.iterrows():
    # loop through each column except the keep column
    for col in data.columns:
        if col != 'name':
            # check if the value contains only alphabetical characters
            if not row[col].isalpha():
                # remove the row if any of the values contain non-alphabetical characters
                data.drop(i, inplace=True)
                break
#df = pd.read_csv('data/data_by_genres.csv', usecols=['genres', 'key'])
#result = pd.merge(data, df, on='key')
#data = result.groupby('name').agg({'name': 'first', 'popularity': 'first', 'artists': 'first', 'genres': lambda x: ' '.join(set(x))})
print(data)


