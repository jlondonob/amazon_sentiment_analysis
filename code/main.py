#Imports
import pandas as pd
import numpy as np
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
#------------------- CLASSIFYING -----------------------#
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

#Plotting negative vs positive sentiment
df['Sentimentt'] = df['Sentiment'].replace({-1 : 'negative'})
df['Sentimentt'] = df['Sentimentt']. replace({1 : 'positive'})

fig = px.histogram(df, x='Sentimentt')
fig.update_traces(marker_color = "indianred",
                  marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text = 'Product Sentiment')
fig.show()

#-------------------------------------------------------#
#-------------------- THE MODEL ------------------------#
#-------------------------------------------------------#

#Creating function to remove punctuation
def remove_punctuation(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", '"'))
    return final

#Removing punctuation
df['Text'] = df['Text'].apply(remove_punctuation) #Remove punctuation in Text
df = df.dropna(subset = ['Summary']) #Remove if na in Summary
df['Summary'] = df['Summary'].apply(remove_punctuation) #Remove punctuation in Summary

#Creating new dataframe
dfNew = df[['Summary', 'Sentiment']]
dfNew.head()

import numpy as np
from sklearn.model_selection import train_test_split
#Splitting train and test datasets
X_train, X_test, y_train, y_test = train_test_split(dfNew.Summary, dfNew.Sentiment, test_size=0.2, random_state=0)

#Matrix representation of token counts
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(X_train) #Train count matrix
test_matrix = vectorizer.transform(X_test)   #Test count matrix

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
lr.fit(train_matrix, y_train)

predictions = lr.predict(test_matrix)

#-------------------------------------------------------#
#-------------- TESTING ACCURACY -----------------------#
#-------------------------------------------------------#

from sklearn.metrics import confusion_matrix, classification_report
new = np.asarray(y_test)
confusion_matrix(predictions, y_test)
print(classification_report(predictions, y_test))

#Classes are unbalanced, thus a balanced accuracy score is better indicator
from sklearn.metrics import balanced_accuracy_score
balanced_accuracy_score(predictions, y_test)



