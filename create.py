import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
data = pd.read_csv("data/data.csv", usecols=['name', 'popularity', 'key', 'artists', 'danceability', 'loudness', 'year'])
data1 = data.get(['year'])

plt.figure(figsize = (10, 5))
sb.countplot(data['year'])
plt.axis('off')
plt.show()




