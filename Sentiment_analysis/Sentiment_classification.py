import numpy as np
import pandas as pd
import re
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

print("ok")

imdb_data = pd.read_table("dataset/imdb_labelled.txt", names = ['sentence', 'label'])
amazon_data = pd.read_table("dataset/amazon_cells_labelled.txt", names = ['sentence', 'label'])
yelp_data = pd.read_table("dataset/yelp_labelled.txt", names = ['sentence', 'label'])

data = pd.concat([imdb_data,amazon_data,yelp_data])
print(data.head(5))



def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()