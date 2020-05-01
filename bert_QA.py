# -*- coding: utf-8 -*-
"""

"""

from keras_bert import Tokenizer
import keras
from keras_bert import get_base_dict, get_model, gen_batch_inputs
import json
from pprint import pprint
import numpy as np
import re
import io
import nltk
import keras as k
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, merge, Dropout, RepeatVector, Activation, merge, Lambda, Flatten, Reshape
from keras.layers import LSTM, Bidirectional, TimeDistributed, GRU, concatenate
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras import optimizers
from keras.optimizers import Adam, RMSprop


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

with open('/content/drive/My Drive/Squad/dev-v2.0.json') as json_data:
    d = json.load(json_data)
trainData = d['data']




nltk.download('punkt')

tContext, tQuestion=splitDataset1(trainData)

tContext, tQuestion, tQuestion_id, tAnswerBegin, tAnswerEnd, tAnswerText, maxLenTContext, maxLenTQuestion = splitDatasets(trainData)

token_dict = get_base_dict()

for pairs in tContext:
    for token in pairs[0] + pairs[1]:
        if token not in token_dict:
            token_dict[token] = len(token_dict)
token_list = list(token_dict.keys())  # Used for selecting a random word

model = get_model(
    token_num=len(token_dict),
    head_num=5,
    transformer_num=12,
    embed_dim=25,
    feed_forward_dim=100,
    seq_len=20,
    pos_num=20,
    dropout_rate=0.05,
)
model.summary()

!pip install transformers

from transformers import BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

question = "How many parameters does BERT-large have?"
answer_text = "BERT-large is really big... it has 24-layers and an embedding size of 1,024, for a total of 340M parameters! Altogether it is 1.34GB, so expect it to take a couple minutes to download to your Colab instance."

def answer_question(question, answer_text):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''
    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, answer_text)

    # Report how long the input sequence is.
    print('Query\n',question)
    print('Query has {:,} tokens.\n'.format(len(input_ids)))

    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========
    # Run our example question through the model.
    start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                    token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):
        
        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        
        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    print('Answer: "' + answer + '"')

data['paragraph'][0]

data.head()

data

data.documents[0]

for i in range(0,15):
   answer_question(data.questions[i], data.documents[i])

for i in range(0,100):
   answer_question(tQuestion[i], tContext[i])

# Apply the tokenizer to the input text, treating them as a text-pair.
input_ids = tokenizer.encode(question, answer_text)

print('The input has a total of {:} tokens.'.format(len(input_ids)))

# BERT only needs the token IDs, but for the purpose of inspecting the 
# tokenizer's behavior, let's also get the token strings and display them.
tokens = tokenizer.convert_ids_to_tokens(input_ids)

# For each token and its id...
for token, id in zip(tokens, input_ids):
    
    # If this is the [SEP] token, add some space around it to make it stand out.
    if id == tokenizer.sep_token_id:
        print('')
    
    # Print the token string and its ID in two columns.
    print('{:<12} {:>6,}'.format(token, id))

    if id == tokenizer.sep_token_id:
        print('')

import torch

# Search the input_ids for the first instance of the `[SEP]` token.
sep_index = input_ids.index(tokenizer.sep_token_id)

# The number of segment A tokens includes the [SEP] token istelf.
num_seg_a = sep_index + 1

# The remainder are segment B.
num_seg_b = len(input_ids) - num_seg_a

# Construct the list of 0s and 1s.
segment_ids = [0]*num_seg_a + [1]*num_seg_b

# There should be a segment_id for every input token.
assert len(segment_ids) == len(input_ids)

# Run our example through the model.
start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                 token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text

# Find the tokens with the highest `start` and `end` scores.
answer_start = torch.argmax(start_scores)
answer_end = torch.argmax(end_scores)

# Combine the tokens in the answer and print it out.
answer = ' '.join(tokens[answer_start:answer_end+1])

print('Answer: "' + answer + '"')

# Pull the scores out of PyTorch Tensors and convert them to 1D numpy arrays.
s_scores = start_scores.detach().numpy().flatten()
e_scores = end_scores.detach().numpy().flatten()

# We'll use the tokens as the x-axis labels. In order to do that, they all need
# to be unique, so we'll add the token index to the end of each one.
token_labels = []
for (i, token) in enumerate(tokens):
    token_labels.append('{:} - {:>2}'.format(token, i))

import pandas as pd

# Store the tokens and scores in a DataFrame. 
# Each token will have two rows, one for its start score and one for its end
# score. The "marker" column will differentiate them. A little wacky, I know.
scores = []
for (i, token_label) in enumerate(token_labels):

    # Add the token's start score as one row.
    scores.append({'token_label': token_label, 
                   'score': s_scores[i],
                   'marker': 'start'})
    
    # Add  the token's end score as another row.
    scores.append({'token_label': token_label, 
                   'score': e_scores[i],
                   'marker': 'end'})
    
df = pd.DataFrame(scores)

import matplotlib.pyplot as plt
import seaborn as sns

# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
#sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (16,8)

# Draw a grouped barplot to show start and end scores for each word.
# The "hue" parameter is where we tell it which datapoints belong to which
# of the two series.
g = sns.catplot(x="token_label", y="score", hue="marker", data=df,
                kind="bar", height=6, aspect=4)

# Turn the xlabels vertical.
g.set_xticklabels(g.ax.get_xticklabels(), rotation=90, ha="center")

# Turn on the vertical grid to help align words to scores.
g.ax.grid(True)

import textwrap

# Wrap text to 80 characters.
wrapper = textwrap.TextWrapper(width=80) 

bert_abstract = "the norman dynasty had a major political , cultural and military impact on medieval europe and even the near east . the normans were famed for their martial spirit and eventually for their christian piety , becoming exponents of the catholic orthodoxy into which they assimilated . they adopted the gallo-romance language of the frankish land they settled , their dialect becoming known as norman , normaund or norman french , an important literary language . the duchy of normandy , which they formed by treaty with the french crown , was a great fief of medieval france , and under richard i of normandy was forged into a cohesive and formidable principality in feudal tenure . the normans are noted both for their culture , such as their unique romanesque architecture and musical traditions , and for their significant military accomplishments and innovations . norman adventurers founded the kingdom of sicily under roger ii after conquering southern italy on the saracens and byzantines , and an expedition on behalf of their duke , william the conqueror , led to the norman conquest of england at the battle of hastings in 1066. norman cultural and military influence spread from these new european centres to the crusader states of the near east , where their prince bohemond i founded the principality of antioch in the levant , to scotland and wales in great britain , to ireland , and to the coasts of north africa and the canary islands ."

print(wrapper.fill(bert_abstract))

f1= f1_match(true, predicted)