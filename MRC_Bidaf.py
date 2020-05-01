# -*- coding: utf-8 -*-
"""

# BIDIRECTIONAL LSTM WITH ATTENTION
"""

import utils
import numpy as np
import io
import nltk
import keras as k
from sklearn.metrics import f1_score,accuracy_score
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, merge, Dropout, RepeatVector, Activation, merge, Lambda, Flatten, Reshape
from keras.layers import LSTM, Bidirectional, TimeDistributed, GRU, concatenate
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras import optimizers
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
import string
import argparse
from collections import Counter
import re
import argparse
import json
import sys
from ipynb.fs.full.Preprocess import splitDatasets, splitValDatasets,vectorizeValData, vectorizeData
from ipynb.fs.full.Attention import Attention

"""# Creating word embeddings with glove"""

embeddings_index = {}
f = open( 'glove.6B.100d.txt',encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

"""Import train and dev datasets"""

with open('train-v2.0.json') as json_data:
    d = json.load(json_data)
trainData = d['data']

tContext, tQuestion, tQuestion_id, tAnswerBegin, tAnswerEnd, tAnswerText, maxLenTContext, maxLenTQuestion = splitDatasets(trainData)

with open('dev-v2.0.json') as json_data:
    d = json.load(json_data)
valData = d['data']

te_Context, te_Question, te_Question_id, te_AnswerBegin, te_AnswerEnd, te_AnswerText, te_maxLenTContext, te_maxLenTQuestion = splitDatasets(valData)

vContext, vToken2CharIdx, vContextOriginal, vQuestion, vQuestion_id, maxLenVContext, maxLenVQuestion = splitValDatasets(valData)

"""# Creation of vocabulary"""

vocab = {}
for words in tContext + tQuestion + vContext + vQuestion:
    for word in words:
        if word not in vocab:
            vocab[word] = 1
vocab = sorted(vocab.keys())

vocab_size = len(vocab) + 1
print(vocab_size)
word_index = dict((c, i + 1) for i, c in enumerate(vocab))
context_maxlen = max(maxLenTContext, maxLenVContext)
question_maxlen = max(maxLenTQuestion, maxLenVQuestion)

"""# Vectorize train and datasets"""

tX, tXq, tYBegin, tYEnd = vectorizeData(tContext, tQuestion, tAnswerBegin, tAnswerEnd, word_index, context_maxlen, question_maxlen)

vX, vXq, vYBegin, vYEnd = vectorizeData(te_Context, te_Question, te_AnswerBegin, te_AnswerEnd, word_index, context_maxlen, question_maxlen)

"""# Creation of embedding for Question and Context"""

nb_words = len(word_index)
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = context_maxlen

embedding_matrix = np.zeros((nb_words+1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,trainable=True)

"""# MODEL CREATION"""

question_input = Input(shape=(question_maxlen,), dtype='int32', name='question_input')
context_input = Input(shape=(context_maxlen,), dtype='int32', name='context_input')
questionEmbd = Embedding(output_dim=EMBEDDING_DIM, input_dim=vocab_size,
                         mask_zero=True, weights=[embedding_matrix], 
                         input_length=question_maxlen, trainable=False)(question_input)
contextEmbd = Embedding(output_dim=EMBEDDING_DIM, input_dim=vocab_size,
                         mask_zero=True, weights=[embedding_matrix], 
input_length=context_maxlen, trainable=False)(context_input)

Q = Bidirectional(LSTM(128, return_sequences=True))(questionEmbd)
D = Bidirectional(LSTM(128, return_sequences=True))(contextEmbd)
Q2 = Attention(question_maxlen)(Q)
D2 = Attention(context_maxlen)(D)
L = concatenate([D2, Q2], axis=1)
answerPtrBegin_output = Dense(context_maxlen, activation='softmax')(L)
Lmerge = concatenate([L, answerPtrBegin_output],axis = 1)
answerPtrEnd_output = Dense(context_maxlen, activation='softmax')(Lmerge)
model = Model(input=[context_input, question_input], output=[answerPtrBegin_output, answerPtrEnd_output])
am = optimizers.Adam(lr=0.0005)
model.compile(optimizer=am, loss='categorical_crossentropy',
              loss_weights=[.04, 0.04], metrics=['accuracy'])
model.summary()


train_slice = 10000
model_history = model.fit([tX, tXq], [tYBegin, tYEnd],batch_size= 128, verbose=2,
                          callbacks = callbacks_list,epochs=30)


"""# Predictions with test data"""

predictions = model.predict([vX, vXq], batch_size=128)
print(predictions[0].shape, predictions[1].shape)

"""Answer Tokens are extracted with answer begin, end and joint"""

ansBegin = np.zeros((predictions[0].shape[0],), dtype=np.int32)
ansEnd = np.zeros((predictions[0].shape[0],),dtype=np.int32) 
for i in range(predictions[0].shape[0]):
    ansBegin[i] = predictions[0][i, :].argmax()
    ansEnd[i] = predictions[1][i, :].argmax()
print(ansBegin.min(), ansBegin.max(), ansEnd.min(), ansEnd.max())
answers = {}
for i in range(len(vQuestion_id)):
    if ansBegin[i] >= len(vContext[i]):
        answers[vQuestion_id[i]] = ""
    elif ansEnd[i] >= len(vContext[i]):
        answers[vQuestion_id[i]] = vContextOriginal[i][vToken2CharIdx[i][ansBegin[i]]:]
    else:
        answers[vQuestion_id[i]] = vContextOriginal[i][vToken2CharIdx[i][ansBegin[i]]:vToken2CharIdx[i][ansEnd[i]]+len(vContext[i][ansEnd[i]])]

"""Saving answers to json file"""

with open('BidafResults_final.json', 'w', encoding='utf-8') as f:
    f.write((json.dumps(answers, ensure_ascii=False)))

"""# Evaluation of Model: F1 Score"""

def f1_eval():
    begin = f1_score(te_AnswerBegin,ansBegin,average="macro")
    end = f1_score(te_AnswerEnd,ansBegin,average="macro")
    f1 = (begin + end) * 100
    return f1

f1 = f1_eval()
print("MODEL F1 SCORE")
f1

