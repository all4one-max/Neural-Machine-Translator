# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 19:52:17 2021

@author: jhasa
"""

import string
import re
import numpy as np
from numpy import array, argmax, random, take
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Bidirectional, RepeatVector, TimeDistributed
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
#% matplotlib inline
pd.set_option('display.max_colwidth', 200)

### Defining the functions which will enable us to read the text file
def read_text(filename):
    # open the file
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    file.close()
    return text

def to_lines(text):
    sents = text.strip().split('\n')
    sents = [i.split('\t') for i in sents]
    return sents

filepath = "C:/Users/jhasa/Desktop/neural machine translator/"

data = read_text(filepath+"fra.txt")
fra_eng = to_lines(data)
fra_eng = array(fra_eng)

fra_eng

fra_eng[:,0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in fra_eng[:,0]]
fra_eng[:,1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in fra_eng[:,1]]
fra_eng

### Converting the text to lower case
for i in range(len(fra_eng)):
    fra_eng[i,0] = fra_eng[i,0].lower()
    
    fra_eng[i,1] = fra_eng[i,1].lower()
eng_l = []
fra_l = []

# populate the language lists with sentence lengths
for i in fra_eng[:,0]:
    eng_l.append(len(i.split()))

for i in fra_eng[:,1]:
    fra_l.append(len(i.split()))

length_df = pd.DataFrame({'eng':eng_l, 'fra':fra_l})
length_df.hist(bins = 30)
plt.show()

#The plot shows frequency of occurence v/s length of phrase for both languages

length_df['eng'].value_counts()

### We can see that the maximum length sequence in english is 8
length_df['fra'].value_counts()

### We can see that the maximum length sequence in French is 14
#Tokenization is the process of converting each word in the vocabulary into an integer based on frequency of occurence

def tokenization(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

eng_tokenizer = tokenization(fra_eng[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1

eng_length = 8
print('English Vocabulary Size: %d' % eng_vocab_size)

fra_tokenizer = tokenization(fra_eng[:, 1])
fra_vocab_size = len(fra_tokenizer.word_index) + 1

fra_length = 14
print('French Vocabulary Size: %d' % fra_vocab_size)

# encode and pad sequences
#encoding means replacing each word with its corresponding number
#Padding essentially means adding zeros to make the length of every sequence equal
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    seq = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq 

trainX = encode_sequences(fra_tokenizer, fra_length, fra_eng[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, fra_eng[:, 0])

### Now we'll build the Sequential model.
### The first layer is the embedding layer which projects each token in an N dimensional vector space
### LSTM is the artificial recurrent neural net architecture.
### It can not only proces past data but take feedback from future data as well.

### In the second LSTM layer, we have set return sequences as True becuase we need outputs of all hidden units and not just the last one.

def build_model(in_vocab, out_vocab, in_timesteps, out_timesteps, units):
    model = Sequential()
    model.add(Embedding(in_vocab, units, input_length=in_timesteps, mask_zero=True))
    model.add(LSTM(units))
    model.add(RepeatVector(out_timesteps))
    model.add(LSTM(units, return_sequences=True))
    model.add(Dense(out_vocab, activation='softmax'))
    return model

model = build_model(fra_vocab_size, eng_vocab_size, fra_length, eng_length, 512)
rms = optimizers.RMSprop(lr=0.001)
model.compile(optimizer=rms, loss='sparse_categorical_crossentropy')

model.summary()

filename = 'model_params.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1), 
          epochs=30, batch_size=512, 
          validation_split = 0.2,
          callbacks=[checkpoint], verbose=1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','validation'])
plt.show()

# saving
with open('french_tokenizer.pickle', 'wb') as handle:
    pickle.dump(fra_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
# saving
with open('english_tokenizer.pickle', 'wb') as handle:
    pickle.dump(eng_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

filepath = "C:/Users/jhasa/Desktop/neural machine translator/"

model = load_model(filepath + 'model.h1.25_sep_20')

# loading
with open(filepath + 'french_tokenizer.pickle', 'rb') as handle:
    french_tokenizer_rec = pickle.load(handle)
# loading
with open(filepath + 'english_tokenizer.pickle', 'rb') as handle:
    eng_tokenizer_rec = pickle.load(handle)
preds = model.predict_classes(testX.reshape((testX.shape[0],testX.shape[1])))



def get_word(n, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return None

def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    seq = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq 

def get_english_sentence(pred_french_sentence):
    temp = []
    for j in range(len(pred_french_sentence)):
        t = get_word(pred_french_sentence[j], eng_tokenizer_rec)
        if j > 0: #If it is not the first word
            if (t == get_word(pred_french_sentence[j-1], eng_tokenizer_rec)) or (t == None):  #if the next word is same as the previous
                temp.append('')
            else:
                temp.append(t)
             
        else: #if it's not the first word
            if(t == None): #if we didn't get a valid code from dictionary 
                temp.append('')
            else:
                temp.append(t)
    return ' '.join(temp)

fra_length = 14

french_sentence = "Je suis saurav"
french_sentence = french_sentence.translate(str.maketrans('', '', string.punctuation))
french_sentence = french_sentence.lower()
encode_french_sentence = encode_sequences(french_tokenizer_rec, fra_length, [french_sentence])
pred_french_sentence = model.predict_classes(encode_french_sentence.reshape((encode_french_sentence.
                                                                             shape[0],encode_french_sentence.shape[1])))
eng_sentence = get_english_sentence(pred_french_sentence[0])



preds_text = []
for i in preds:
    temp = []
    for j in range(len(i)):
        t = get_word(i[j], eng_tokenizer)
        if j > 0: #If it is not the first word
            if (t == get_word(i[j-1], eng_tokenizer)) or (t == None):  #if the next word is same as the previous
                temp.append('')
            else:
                temp.append(t)
             
        else: #if it's not the first word
            if(t == None): #if we didn't get a valid code from dictionary 
                temp.append('')
            else:
                temp.append(t)            
        
    preds_text.append(' '.join(temp))
    
pred_df = pd.DataFrame({'actual' : test[:,0], 'predicted' : preds_text})
pd.set_option('display.max_colwidth', 200)

pred_df.tail(25)