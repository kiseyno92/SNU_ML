
# coding: utf-8

# In[10]:

import os
import nltk
from nltk.corpus import stopwords
files = os.listdir('aclimdb/train/pos')


# In[11]:

words =[]
for file in files:
    with open('aclimdb/train/pos/{}'.format(file), 'r' , encoding = 'utf-8') as f:
        review = nltk.word_tokenize(f.read())
        for token in review:
            words.append(token)
print(len(words))


# In[12]:

files = os.listdir('aclimdb/train/neg')


# In[13]:

for file in files:
    with open('aclimdb/train/neg/{}'.format(file), 'r' , encoding = 'utf-8') as f:
        review = nltk.word_tokenize(f.read())
        for token in review:
            words.append(token)
print(len(words))


# In[14]:

words = nltk.FreqDist(words)
word_features = list(words.keys())[:3000]


# In[15]:

def find_features(doc):
    words = set(doc)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


# In[17]:

with open('aclimdb/train/neg/{}'.format(files[0]), 'r' , encoding = 'utf-8') as f:
    review = nltk.word_tokenize(f.read())
find_features(review)


# In[20]:

feature_sets=[]
files = os.listdir('aclimdb/train/pos')[:1000]
for file in files:
    with open('aclimdb/train/pos/{}'.format(file), 'r' , encoding = 'utf-8') as f:
        review = nltk.word_tokenize(f.read())
        feature_sets.append((find_features(review), 'pos'))
     


# In[21]:

clf = nltk.NaiveBayesClassifier.train(training_set)


# In[ ]:



