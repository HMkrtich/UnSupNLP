#!/usr/bin/env python3

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import random
import pickle
import nltk

nltk.download('wordnet')

# Load documents

newsgroups_train = fetch_20newsgroups(subset='train')

# Load test data
newsgroups_test = fetch_20newsgroups(subset='test')

print(len(newsgroups_train.data), " documents loaded.")
print(len(newsgroups_test.data), " documents loaded.")

print("Example document:")
print(newsgroups_train.data[0])




# Preprocess documents - lemmatization and stemming

def lemmatize_stemming(text):
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

processed_docs = list(map(preprocess, newsgroups_train.data))
processed_docs_test = list(map(preprocess, newsgroups_test.data))

print("Example document - lemmatized and stemmed:")
print(processed_docs[0])


# Construct dictionary

dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

print("Dictionary size: ", len(dictionary))

# Filter words in documents

docs = list()
maxdoclen = 0 
for doc in processed_docs:
    docs.append(list(filter(lambda x: x != -1, dictionary.doc2idx(doc))))
    maxdoclen = max(maxdoclen, len(docs[-1]))

docs_test = list()
maxdoclen_test = 0
for doc in processed_docs_test:
    docs_test.append(list(filter(lambda x: x != -1, dictionary.doc2idx(doc))))
    maxdoclen_test = max(maxdoclen_test, len(docs_test[-1]))

print("Example document - filtered:")
print(docs[0])

print("Maximum document length:", maxdoclen)


# Set the hyperparameters

iterations = 100 
topics = 20 
alpha = 0.01 
gamma = 0.01

doc_cnt = len(docs)
wrd_cnt = len(dictionary)

# Save the documents in local directory
# Save preprocessed documents
with open('preprocessed_docs.pkl', 'wb') as f:
    pickle.dump(docs, f)

# Save dictionary
with open('dictionary.pkl', 'wb') as f:
    pickle.dump(dictionary, f)

# Save test documents
with open('preprocessed_docs_test.pkl', 'wb') as f:
    pickle.dump(docs_test, f)




