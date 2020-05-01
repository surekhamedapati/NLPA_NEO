# -*- coding: utf-8 -*-
"""

# PREPROCESS DATA
"""

from __future__ import print_function
import string
import argparse
from collections import Counter
import re
import argparse
import json
import sys

"""# Tokenize Data"""

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

"""# Split datasets into Content, Question, id, Answerbegin, Answer end and answer content for each text in data"""

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

"""# VECTORIZE TRAIN AND TEST DATA
1. Vectorize the words to their respective index and pad context to max context length and question to max question length.
2. Answers are also padded in same way
"""

def vectorizeData(xContext, xQuestion, xAnswerBeing, xAnswerEnd, word_index, context_maxlen, question_maxlen):
     X = []
    Xq = []
    YBegin = []
    YEnd = []
    for i in range(len(xContext)):
        x = [word_index[w] for w in xContext[i]]
        xq = [word_index[w] for w in xQuestion[i]]
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