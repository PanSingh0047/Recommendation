import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

import warnings
user_data = None
warnings.filterwarnings('ignore')
data = pd.read_csv("data/data.csv", usecols=['name', 'popularity', 'key', 'artists', 'danceability', 'loudness', 'year'])
# data = data.drop(['id','release_date'], axis = 1)
data.drop_duplicates(subset=['name'], keep='first', inplace=True)

data = data.sort_values(by=['popularity'], ascending=False).head(5000)
df = pd.read_csv('data/data_by_genres.csv', usecols=['genres', 'key'])
result = pd.merge(data, df, on='key')
data = result.groupby('name').agg({'name': 'first', 'popularity': 'first', 'artists': 'first', 'danceability': 'first', 'loudness': 'first','year':'first', 'genres': lambda x: ' '.join(set(x))})
song_vectorizer = CountVectorizer()
song_vectorizer.fit(data['genres'])



def jack_similarity(song_name, data1):
    sim = []
    set1 = set(data1[data1['name'] == song_name]['genres'])
    for idx, row in data1.iterrows():
        name = row['name']
        set2 = set(data1[data1['name'] == name]['genres'])
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        jaccard_similarity = intersection / union
        sim.append(jaccard_similarity)
    return sim


def get_similarities(song_name, data1):
    # Getting vector for the input song.
    text_array1 = song_vectorizer.transform(data1[data1['name'] == song_name]['genres']).toarray()
    #print(text_array1)
    num_array1 = data1[data1['name'] == song_name].select_dtypes(include=np.number).to_numpy()

    # We will store similarity for each row of the dataset.
    sim = []
    for idx, row in data1.iterrows():
        name = row['name']

        # Getting vector for current song.
        text_array2 = song_vectorizer.transform(data1[data1['name'] == name]['genres']).toarray()
        num_array2 = data1[data1['name'] == name].select_dtypes(include=np.number).to_numpy()

        # Calculating similarities for text as well as numeric features
        text_sim = cosine_similarity(text_array1, text_array2)[0][0]
        num_sim = cosine_similarity(num_array1, num_array2)[0][0]
        sim.append(text_sim + num_sim)

    return sim


def recommend_songs(song_name, data_recom=data):
    # Base case
    if data_recom[data_recom['name'] == song_name].shape[0] == 0:
        print('This song is either not so popular or you\
        have entered invalid_name.\n Some songs you may like:\n')
        song_names = {'name': []}
        df = pd.DataFrame(song_names)
        for song in data_recom.sample(n=5)['name'].values:
            new_row = {'name': song}
            df.loc[len(df)] = new_row
        return df
    data_recom['cosine_factor'] = get_similarities(song_name, data_recom)
    data_recom['similarity_factor'] = jack_similarity(song_name, data_recom)
    data_recom.sort_values(by=['similarity_factor', 'popularity', 'cosine_factor'], ascending=[False, False, False], inplace=True)
    return data_recom['name'][1:7]


songs = recommend_songs("user_data")
print(songs)
