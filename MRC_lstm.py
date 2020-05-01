# -*- coding: utf-8 -*-
"""

# LSTM MODEL
"""

import json
import numpy as np
import re
import io
import nltk
import h5py
from keras import backend as K
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, Dropout, RepeatVector, Activation, merge, Lambda, Flatten, Reshape
from keras.layers import LSTM, Bidirectional, TimeDistributed, GRU
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras import optimizers
from keras.optimizers import Adam, RMSprop
from keras.layers import concatenate


embeddings_index = {}
f = open('/data/Glove/glove.6B.100d.txt',encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

#print('Found %s word vectors.' % len(embeddings_index))

from __future__ import print_function
import string
import argparse
from collections import Counter
import re
import argparse
import json
import sys
import nltk
nltk.download('punkt')
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

def tokenize(sent):
   
    return [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(sent)]

def tokenizeVal(sent):
    tokenizedSent = [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(sent)]
    tokenIdx2CharIdx = [None] * len(tokenizedSent)
    idx = 0
    token_idx = 0
    while idx < len(sent) and token_idx < len(tokenizedSent):
        word = tokenizedSent[token_idx]
        if sent[idx:idx+len(word)] == word:
            tokenIdx2CharIdx[token_idx] = idx
            idx += len(word)
            token_idx += 1 
        else:
            idx += 1
    return tokenizedSent, tokenIdx2CharIdx
    
    
def splitDatasets(f):
   
    xContext = [] # list of contexts paragraphs
    xQuestion = [] # list of questions
    xQuestion_id = [] # list of question id
    xAnswerBegin = [] # list of indices of the beginning word in each answer span
    xAnswerEnd = [] # list of indices of the ending word in each answer span
    xAnswerText = [] # list of the answer text
    maxLenContext = 0
    maxLenQuestion = 0

    for data in f:
        paragraphs = data['paragraphs']
        for paragraph in paragraphs:
            context = paragraph['context']
            context1 = context.replace("''", '" ')
            context1 = context1.replace("``", '" ')
            contextTokenized = tokenize(context.lower())
            contextLength = len(contextTokenized)
            if contextLength > maxLenContext:
                maxLenContext = contextLength
            qas = paragraph['qas']
            for qa in qas:
                question = qa['question']
                question = question.replace("''", '" ')
                question = question.replace("``", '" ')
                questionTokenized = tokenize(question.lower())
                if len(questionTokenized) > maxLenQuestion:
                    maxLenQuestion = len(questionTokenized)
                question_id = qa['id']
                answers = qa['answers']
                for answer in answers:
                    answerText = answer['text']
                    answerTokenized = tokenize(answerText.lower())
                    # find indices of beginning/ending words of answer span among tokenized context
                    contextToAnswerFirstWord = context1[:answer['answer_start'] + len(answerTokenized[0])]
                    answerBeginIndex = len(tokenize(contextToAnswerFirstWord.lower())) - 1
                    answerEndIndex = answerBeginIndex + len(answerTokenized) - 1
                    
                    xContext.append(contextTokenized)
                    xQuestion.append(questionTokenized)
                    xQuestion_id.append(str(question_id))
                    xAnswerBegin.append(answerBeginIndex)
                    xAnswerEnd.append(answerEndIndex)
                    xAnswerText.append(answerText)
    return xContext, xQuestion, xQuestion_id, xAnswerBegin, xAnswerEnd, xAnswerText, maxLenContext, maxLenQuestion
    
    

def splitValDatasets(f):
    xContext = [] # list of contexts paragraphs
    xQuestion = [] # list of questions
    xQuestion_id = [] # list of question id
    xToken2CharIdx = []
    xContextOriginal = []
    maxLenContext = 0
    maxLenQuestion = 0

    for data in f:
        paragraphs = data['paragraphs']
        for paragraph in paragraphs:
            context = paragraph['context']
            context1 = context.replace("''", '" ')
            context1 = context1.replace("``", '" ')
            contextTokenized, tokenIdx2CharIdx = tokenizeVal(context1.lower())
            contextLength = len(contextTokenized)
            if contextLength > maxLenContext:
                maxLenContext = contextLength
            qas = paragraph['qas']
            for qa in qas:
                question = qa['question']
                question = question.replace("''", '" ')
                question = question.replace("``", '" ')
                questionTokenized = tokenize(question.lower())
                if len(questionTokenized) > maxLenQuestion:
                    maxLenQuestion = len(questionTokenized)
                question_id = qa['id']
                answers = qa['answers']
                
                xToken2CharIdx.append(tokenIdx2CharIdx)
                xContextOriginal.append(context)
                xContext.append(contextTokenized)
                xQuestion.append(questionTokenized)
                xQuestion_id.append(str(question_id))

    return xContext, xToken2CharIdx, xContextOriginal, xQuestion, xQuestion_id, maxLenContext, maxLenQuestion
    
    

def vectorizeData(xContext, xQuestion, xAnswerBeing, xAnswerEnd, word_index, context_maxlen, question_maxlen):
    X = []
    Xq = []
    YBegin = []
    YEnd = []
    for i in range(len(xContext)):
        x = [word_index[re.sub(r'["`]+','', w)] if re.sub(r'["`]+','', w) in word_index else word_index['the'] for w in xContext[i] if len(re.sub(r'["`]+','', w))>0]
        xq = [word_index[re.sub(r'["`]+','', w)] if re.sub(r'["`]+','', w) in word_index else word_index['the'] for w in xQuestion[i] if len(re.sub(r'["`]+','', w))>0]
        # map the first and last words of answer span to one-hot representations
        y_Begin =  np.zeros(len(xContext[i]))
        y_Begin[xAnswerBeing[i]] = 1
        y_End = np.zeros(len(xContext[i]))
        y_End[xAnswerEnd[i]] = 1
        X.append(x)
        Xq.append(xq)
        YBegin.append(y_Begin)
        YEnd.append(y_End)
    return pad_sequences(X, maxlen=context_maxlen, padding='post'), pad_sequences(Xq, maxlen=question_maxlen, padding='post'), pad_sequences(YBegin, maxlen=context_maxlen, padding='post'), pad_sequences(YEnd, maxlen=context_maxlen, padding='post')
    
    

def vectorizeValData(xContext, xQuestion, word_index, context_maxlen, question_maxlen):
    X = []
    Xq = []
    YBegin = []
    YEnd = []
    for i in range(len(xContext)):
        x = [word_index[w] for w in xContext[i]]
        xq = [word_index[w] for w in xQuestion[i]]

        X.append(x)
        Xq.append(xq)

    return pad_sequences(X, maxlen=context_maxlen, padding='post'), pad_sequences(Xq, maxlen=question_maxlen, padding='post')

context = h5py.File('data/context.h5','r')
questions = h5py.File('data/questions.h5','r')
answers = h5py.File('data/answers.h5','r')
ans_begin = h5py.File('data/begin.h5','r')
ans_end = h5py.File('data/end.h5','r')

c_data = context['context'][:]
qn_data = questions['questions'][:]
ans_data = answers['answers'][:]

begin_ans = ans_begin['begin'][:]
end_ans = ans_end['end'][:]

# loding vocabulary
word_index = np.load('data/words.npy', allow_pickle=True).item()

embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

vocab_size = len(word_index) + 1
#embedding_vector_length = 50
batch = 128
max_span_begin = np.amax(begin_ans)
max_span_end = np.amax(end_ans)
slce = 10000

print("Vocab Size")
vocab_size

context_input = Input(shape=(700, ), dtype='int32', name='c_data')
context_embed = Embedding(input_dim=vocab_size, output_dim=100, weights=[embedding_matrix], 
              input_length=700, trainable=False)(context_input)
#lstm_out = (LSTM(256, return_sequences=True, implementation=2))(x)
drop_1 = Dropout(0.5)(context_embed)
#drop_1 = Dropout(0.5)(lstm_out)

ques_input = Input(shape=(100, ), dtype='int32', name='qn_data')
question_embed = Embedding(input_dim=vocab_size, output_dim=100, weights=[embedding_matrix], 
              input_length=100, trainable=False)(ques_input)
#lstm_out = (LSTM(256, return_sequences=True, implementation=2))(x)
drop_2 = Dropout(0.5)(question_embed)
#drop_2 = Dropout(0.5)(lstm_out)

merge_layer = concatenate([drop_1, drop_2], axis=1)
lstm_layer = (LSTM(512, implementation=2))(merge_layer)
drop_3 =  Dropout(0.5)(lstm_layer)
softmax_1 = Dense(max_span_begin, activation='softmax')(lstm_layer)
softmax_2 = Dense(max_span_end, activation='softmax')(lstm_layer)
model = Model(inputs=[context_input, ques_input], outputs=[softmax_1, softmax_2])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model_history = model.fit([c_data[:slce], qn_data[:slce]],
                        [begin_ans[:slce], end_ans[:slce]], verbose=2,
                         batch_size=batch, epochs=50)

"""# PREDICTIONS USING TEST DATA"""

t_context = h5py.File('/data/context_test.h5','r')
t_questions = h5py.File('/data/questions_test.h5','r')
t_answers = h5py.File('/data/answers_test.h5','r')
t_ans_begin = h5py.File('/data/begin_test.h5','r')
t_ans_end = h5py.File('/data/end_test.h5','r')

t_c_data = t_context['context'][:]
t_qn_data = t_questions['questions'][:]
t_ans_data = t_answers['answers'][:]
t_begin_ans = t_ans_begin['begin'][:]
t_end_ans = t_ans_end['end'][:]

index_train = np.load('data/indxes.npy', allow_pickle=True)
index_test = np.load('/data/indxes_test.npy', allow_pickle=True)

predictions = model.predict([t_c_data,t_qn_data], batch_size=128)

predictions

print(predictions[0].shape, predictions[1].shape)

ansBegin = np.zeros((predictions[0].shape[0],), dtype=np.int32)
ansEnd = np.zeros((predictions[0].shape[0],),dtype=np.int32)

for i in range(predictions[0].shape[0]):
    ansBegin[i] = predictions[0][i, :].argmax()
    ansEnd[i] = predictions[1][i, :].argmax()
print(ansBegin.min(), ansBegin.max(), ansEnd.min(), ansEnd.max())

import pandas as pd
pd.Series(ansEnd).value_counts()

with open('data/dev-v1.1.json') as json_data:
    d = json.load(json_data)
valData = d['data']

te_Context, te_Question, te_Question_id, te_AnswerBegin, te_AnswerEnd, te_AnswerText, te_maxLenTContext, te_maxLenTQuestion = splitDatasets(valData)

vContext, vToken2CharIdx, vContextOriginal, vQuestion, vQuestion_id, maxLenVContext, maxLenVQuestion = splitValDatasets(valData)

answers = {}
for i in range(len(vQuestion_id)):
    if ansBegin[i] >= len(vContext[i]):
        answers[vQuestion_id[i]] = ""
    elif ansEnd[i] >= len(vContext[i]):
        answers[vQuestion_id[i]] = vContextOriginal[i][vToken2CharIdx[i][ansBegin[i]]:]
    else:
        answers[vQuestion_id[i]] = vContextOriginal[i][vToken2CharIdx[i][ansBegin[i]]:vToken2CharIdx[i][ansEnd[i]]+len(vContext[i][ansEnd[i]])]

answers

"""Saving answers to json file"""

with open('LSTMResults_final.json', 'w', encoding='utf-8') as f:
    f.write((json.dumps(answers, ensure_ascii=False)))

def f1_eval():
    begin = f1_score(te_AnswerBegin,ansBegin,average="macro")
    end = f1_score(te_AnswerEnd,ansBegin,average="macro")
    f1 = (begin + end) * 100
    return f1

from sklearn.metrics import f1_score,accuracy_score
f1 = f1_eval()
print("MODEL F1 SCORE")
f1