# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np 
import tensorflow as tf 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from bs4 import BeautifulSoup
import spacy
from tensorflow.keras import layers 
#import unidecode
#from word2number import w2n
#import pycontractions
#import contractions 


# %%
data = pd.read_csv(r'D:\project\resnet_basics\all_tickets.csv')
#data['ticket_type'].plot(kind = 'hist')


# %%
print(data.info())

# %% [markdown]
# # EXTRACTING REQUIRED VALUES

# %%
df = data.loc[:, ['title','body','category','sub_category1','sub_category1','urgency',]].copy()
df.head()


# %%
size = len(df)
size

# %% [markdown]
# # ADDING EXTRA COLUMN

# %%
# 
df['department'] = np.random.randint(0,4,size=size)


# %%
df.isna().sum()


# %%
# taking care of null values
df['title'] = df['title'].fillna("Value Missing")


# %%
def strip_html_tags(text): 
    """remove html tags""" 
    soup = BeautifulSoup(text, "html.parser") 
    stripped_text = soup.get_text(separator=" ") 
    return stripped_text 

def word2number(text):
    help_dict = { 
    'one': '1', 
    'two': '2', 
    'three': '3', 
    'four': '4', 
    'five': '5', 
    'six': '6', 
    'seven': '7', 
    'eight': '8', 
    'nine': '9', 
    'zero' : '0'
} 

    return  " ".join([help_dict[word] for word in text.split() if word in help_dict])

import nltk

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])

# applying lemmatization
df['title'] = df['title'].apply(lemmatize_text)
df['body'] = df['body'].apply(lemmatize_text)


# %%
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction import text 
mystopwords = text.ENGLISH_STOP_WORDS
vectorizer_body = TfidfVectorizer(
    strip_accents='ascii',
    lowercase=True,
    stop_words='english',
)

body_ = vectorizer_body.fit_transform(df['body']) 


vectorizer_title = TfidfVectorizer(
    strip_accents='ascii',
    lowercase=True,
    stop_words=mystopwords
)

title_ = vectorizer_title.fit_transform(df['title'])



# %%
# extracting tags
tags = df.loc[:,['category','sub_category1','sub_category1']]
tags


# %%
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
tags_ = enc.fit_transform(tags)


# %%
priority_ = tf.keras.utils.to_categorical(df['urgency'])
department_ = tf.keras.utils.to_categorical(df['department'])


# %%
title_.shape[0]


# %%
title = tf.keras.Input(shape=(title_.shape[1],), name = 'title')
text_body = tf.keras.Input(shape=(body_.shape[1],),name = 'body') 
tags = tf.keras.Input(shape=(tags_.shape[1],),name = 'tags')

features = layers.Concatenate()([title, text_body, tags])
batch_normalization = layers.BatchNormalization()(features)

features = layers.Dense(512, activation = 'relu')(batch_normalization)
batch_normalization = layers.BatchNormalization()(features)
dropout = layers.Dropout(0.3)(batch_normalization)

features = layers.Dense(512, activation = 'relu')(dropout)
batch_normalization = layers.BatchNormalization()(features)
dropout = layers.Dropout(0.3)(batch_normalization)

features = layers.Dense(128, activation = 'relu')(dropout)
batch_normalization = layers.BatchNormalization()(features)
dropout = layers.Dropout(0.3)(batch_normalization)
 
priority = layers.Dense(1, activation = 'sigmoid', name = 'priority')(dropout)

department = layers.Dense(4, activation = 'softmax', name = 'department')(dropout)

model = tf.keras.Model(inputs = [title,text_body, tags], outputs = [priority, department]) 

model.summary()


# %%
tf.keras.utils.plot_model(model, 'sample.jpg')


# %%
model.compile(optimizer='adam',
 loss=['mean_squared_error', 'categorical_crossentropy'],
 metrics=[['mean_absolute_error'], ['accuracy']]) 


# %%
model.fit({'title':title_.toarray(), 'body':body_.toarray(), 'tags': tags_.toarray()},
[priority_, department_],
epochs = 1,
batch_size = 32
) 


# %%
iterator = iter(({'title':title_.toarray(), 'body':body_.toarray(), 'tags': tags_.toarray()},
[priority_, department_]))


# %%



