import re
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#Load data

train_positive =  open("/data1/home/mukeshram/Natural_Language_Processing/Natural_Language_Processing/Sentiment_analysis/dataset/Train.pos" , "r", encoding="latin-1").read()
train_negative =  open("/data1/home/mukeshram/Natural_Language_Processing/Natural_Language_Processing/Sentiment_analysis/dataset/Train.neg" , "r", encoding="latin-1").read()
# test_data =  open("/data1/home/mukeshram/Natural_Language_Processing/Natural_Language_Processing/Sentiment_analysis/dataset/TestData" , "r", encoding="latin-1").read()


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
# test = clean(test_data.split("\n")[:-1])

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

#w2v
x_train_w2v=[]
x_train_glv=[]

for i in x_train_data:
    x_train_w2v.append(sentence_embedding(i, word2vec_model))
    x_train_glv.append(sentence_embedding(i, glove_model))

# #glv
# x_test_w2v=[]
# x_test_glv=[]

# for i in test:
#     x_test_w2v.append(sentence_embedding(i, word2vec_model))
#     x_test_glv.append(sentence_embedding(i, glove_model))


x_train_w2v = np.array(x_train_w2v)
y_train = np.array(y_train)
# x_test_w2v = np.array(x_test_w2v)


x_train_glv = np.array(x_train_glv)
y_train = np.array(y_train)
# x_test_glv = np.array(x_test_glv)

x_train_w2vglv =np.concatenate((x_train_w2v, x_train_glv), axis=1)
# x_test_w2vglv =np.concatenate((x_test_w2v, x_test_glv),axis=1)


print(x_train_w2vglv.shape,y_train.shape)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping


x_train_w2vglv = x_train_w2vglv.reshape((x_train_w2vglv.shape[0], x_train_w2vglv.shape[1], 1))

x_train, x_test, y_train, y_test = train_test_split(x_train_w2vglv, y_train, test_size=0.2, random_state=42)


# Define the RNN model
model = Sequential()
model.add(SimpleRNN(128, input_shape=(600, 1), activation='relu'))  # Adjust the number of units as needed
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])


y_pred = model.predict(x_test)
y_pred_binary = (y_pred > 0.5).astype(int)  # Convert predicted probabilities to binary labels
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Test Accuracy of RNN: {accuracy:.4f}")