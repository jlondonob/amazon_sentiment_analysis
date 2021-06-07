#Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px

#Get data
df = pd.read_csv('data/raw/Reviews.csv')
df.head()
df.info()

fig = px.histogram(df, x="Score")
fig.update_traces(marker_color = "turquoise",
                  marker_line_color = 'rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text = "Product Score")
fig.show()

#-------------------------------------------------------#
#-------------------- WORDCLOUDS -----------------------#
#-------------------------------------------------------#

import nltk  #Natural language toolkit
from nltk.corpus import stopwords
from wordcloud import WordCloud

#Stopword list
#nltk.download('stopwords') -- Uncomment if first time running script
stop_words = set(stopwords.words('english'))
stop_words.update(['br', 'href'])
textt = " ".join(review for review in df.Text) #Appends every review in a sigle string


wordcloud = WordCloud(stopwords = stop_words).generate(textt)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#-------------------------------------------------------#
#------------------ CLASSIFIYING -----------------------#
#-------------------------------------------------------#

df = df[df.Score != 3]
df['Sentiment'] = df.Score.apply(lambda rating : +1 if rating > 3 else -1)
df.head()

#Splitting df to positive and negative sentiment dfs
positive = df[df.Sentiment == 1]
negative = df[df.Sentiment == -1]

#Remove "good" and "great" because they appeared in both texts
stop_words.update(['good','great'])

positive_text = " ".join(review for review in positive.Text)
negative_text = " ".join(review for review in negative.Text)

wordcloud_pos = WordCloud(stopwords = stop_words).generate(positive_text)
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis("off")
plt.show()

wordcloud_neg = WordCloud(stopwords = stop_words).generate(negative_text)
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis("off")
plt.show()
