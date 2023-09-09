import re
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#Load data

train_positive =  open("/data1/home/mukeshram/Natural_Language_Processing/Natural_Language_Processing/Sentiment_analysis/dataset/Train.pos" , "r", encoding="latin-1").read()
train_negative =  open("/data1/home/mukeshram/Natural_Language_Processing/Natural_Language_Processing/Sentiment_analysis/dataset/Train.neg" , "r", encoding="latin-1").read()
test_data =  open("/data1/home/mukeshram/Natural_Language_Processing/Natural_Language_Processing/Sentiment_analysis/dataset/TestData" , "r", encoding="latin-1").read()


#data Preprocessing

def clean(data):
    cleaned_data = []

    for i in data:
        i = i.lower()
        pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        i = pattern.sub('', i)
        i = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", i)
        tokens = word_tokenize(i)
        stop_words = set(stopwords.words("english"))
        stop_words.discard("not")
        words = [w for w in tokens if w.isalpha() and w not in stop_words]
        words = ' '.join(words)
        cleaned_data.append(words)
    return cleaned_data

#Seprate features and labels

train_pos = clean(train_positive.split("\n")[:-1])
train_neg = clean(train_negative.split("\n")[:-1])
test = clean(test_data.split("\n")[:-1])

x_train_p = [i +"1" for i in train_pos]
x_train_n = [i +"0" for i in train_neg]
x_data = x_train_p + x_train_n

import random
random.seed(42)
random.shuffle(x_data)

x_train_data = [i[:-1] for i in x_data]
y_train = [int(i[-1]) for i in x_data]


#load Word2vec model
from gensim.models import KeyedVectors
word2vec_model = KeyedVectors.load("/data1/home/mukeshram/NLP/word2vec_google_news_model")

#load glove model
from gensim.models import KeyedVectors
glove_model = KeyedVectors.load("/data1/home/mukeshram/NLP/glove-wiki-gigaword-300")


# Sentence embedding
import numpy as np

def sentence_embedding(sentence, model):
    words = sentence.split()
    sentence_vector = np.zeros(model.vector_size)
    
    for word in words:
        if word in word2vec_model:
            sentence_vector += word2vec_model[word]

    num_words = len(words)
    if num_words > 0:
        sentence_vector /= num_words
    
    return sentence_vector


x_train_w2v=[]
x_train_glv=[]

for i in x_train_data:
    x_train_w2v.append(sentence_embedding(i, word2vec_model))
    x_train_glv.append(sentence_embedding(i, glove_model))


x_test_w2v=[]
x_test_glv=[]

for i in test:
    x_test_w2v.append(sentence_embedding(i, word2vec_model))
    x_test_glv.append(sentence_embedding(i, glove_model))


x_train_w2v = np.array(x_train_w2v)
y_train = np.array(y_train)
x_test_w2v = np.array(x_test_w2v)


x_train_glv = np.array(x_train_glv)
y_train = np.array(y_train)
x_test_glv = np.array(x_test_glv)


from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train_w2v, X_test_w2v, Y_train_w2v, Y_test_w2v = train_test_split(x_train_w2v, y_train, test_size=0.2, random_state=42)

clf = svm.SVC(kernel='linear')

clf.fit(X_train_w2v, Y_train_w2v)

y_pred_w2v= clf.predict(X_test_w2v)

accuracy = accuracy_score(Y_test_w2v, y_pred_w2v)
print(f"Accuracy with word2vec: {accuracy * 100:.4f}%")




X_train_glv, X_test_glv, Y_train_glv, Y_test_glv = train_test_split(x_train_glv, y_train, test_size=0.2, random_state=42)

clf = svm.SVC(kernel='linear')

clf.fit(X_train_glv, Y_train_glv)

y_pred_glv= clf.predict(X_test_glv)

accuracy = accuracy_score(Y_test_glv, y_pred_glv)
print(f"Accuracy with glove: {accuracy * 100:.4f}%")