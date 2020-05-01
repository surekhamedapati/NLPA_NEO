# -*- coding: utf-8 -*-
"""
# PREPROCESS SQUAD DATASET:

1. SPLIT TRAIN AND TEST DATASET INTO QUESTION, CONTEXT, ANSWER START, ANSWER END, ANSWER TEXT AND ID FOR EACH ENTRY
2. VECTORIZE THE SPLIT DATA AND STORE IN H5PY FILE FOR FUTURE USE OF MODEL TRAINING
"""

import json
import numpy as np
import string
import re
import h5py

with open('train-v1.1.json') as json_data:
    d = json.load(json_data)

dataset = d['data']

print(dataset[1])

words = {}
indexes = {}
index = 1
context_list = []
question_list = []
answer_list = []
answer_begin = []
answer_end = []
id_list = []

def modify(cont):
    global index
    temp_list = []
    temp_str = ""
    for i in range(len(cont)):
        if cont[i] == '"' or cont[i] == '/' or cont[i] == ';' or cont[i] == ',':
            continue

        if cont[i] == '?' or cont[i] == ' ' or cont[i] == '.':
            if cont[i] == ',' or cont[i] == '.' or cont[i] == '?':
                i += 1

            word = temp_str.lower()
            temp_str = ""
            temp_index = 0

            if word not in words:
                words[word] = index
                indexes[index] = word
                temp_index = index
                index += 1
            else:
                temp_index = words[word]

            temp_list.append(temp_index)
        else:
            temp_str += cont[i]
    return temp_list

for article in dataset:
    for paragraph in article['paragraphs']:
        for qa in paragraph['qas']:
            id_list.append(qa['id'])
            for ans in qa['answers']:
                # append both context and questions many times for more than one question/answer
                ques = qa['question']
                if len(modify(ques)) < 100:
                    question_list.append(modify(ques))
                    cont = paragraph['context']
                    context_list.append(modify(cont))

                    an = ans['text']
                    answer_list.append(modify(an))

                    answer_begin.append(ans['answer_start'])
                    answer_end.append(ans['answer_start']+len(ans['text']))

context_array = np.zeros((len(context_list), 700), dtype=np.int)
question_array = np.zeros((len(question_list), 100), dtype=np.int)
answer_array = np.zeros((len(answer_list), 100), dtype=np.int)
begin_array = np.zeros((len(answer_begin), ), dtype=np.int)
end_array = np.zeros((len(answer_end), ), dtype=np.int)

for i in range(len(context_list)):
    for j in range(len(context_list[i])):
        context_array[i][j] = context_list[i][j]

for i in range(len(question_list)):
    for j in range(len(question_list[i])):
        question_array[i][j] = question_list[i][j]


for i in range(len(answer_list)):
    for j in range(len(answer_list[i])):
        answer_array[i][j] = answer_list[i][j]

for i in range(len(answer_begin)):
    begin_array[i] = answer_begin[i]

for i in range(len(answer_end)):
    end_array[i] = answer_end[i]

print(context_array.shape)
print(question_array.shape)
print(answer_array.shape)
print(begin_array.shape)
print(end_array.shape)

with h5py.File('context.h5', 'w') as hf:
    hf.create_dataset('context', data=context_array)
with h5py.File('questions.h5', 'w') as hf:
    hf.create_dataset('questions', data=question_array)
with h5py.File('answers.h5', 'w') as hf:
    hf.create_dataset('answers', data=answer_array)
with h5py.File('begin.h5', 'w') as hf:
    hf.create_dataset('begin', data=begin_array)
with h5py.File('end.h5', 'w') as hf:
    hf.create_dataset('end', data=end_array)

np.save('words.npy', words)
np.save('indxes', indexes)

"""Pre-process Test data"""

with open('dev-v1.1.json') as json_data:
    d = json.load(json_data)

dataset = d['data']

words = {}
indexes = {}
index = 1
context_list = []
question_list = []
answer_list = []
answer_begin = []
answer_end = []
id_list = []

for article in dataset:
    for paragraph in article['paragraphs']:
        for qa in paragraph['qas']:
            id_list.append(qa['id'])
            for ans in qa['answers']:
                # append both context and questions many times for more than one question/answer
                ques = qa['question']
                if len(modify(ques)) < 100:
                    question_list.append(modify(ques))
                    cont = paragraph['context']
                    context_list.append(modify(cont))

                    an = ans['text']
                    answer_list.append(modify(an))

                    answer_begin.append(ans['answer_start'])
                    answer_end.append(ans['answer_start']+len(ans['text']))

context_array = np.zeros((len(context_list), 700), dtype=np.int)
question_array = np.zeros((len(question_list), 100), dtype=np.int)
answer_array = np.zeros((len(answer_list), 100), dtype=np.int)
begin_array = np.zeros((len(answer_begin), ), dtype=np.int)
end_array = np.zeros((len(answer_end), ), dtype=np.int)

for i in range(len(context_list)):
    for j in range(len(context_list[i])):
        context_array[i][j] = context_list[i][j]

for i in range(len(question_list)):
    for j in range(len(question_list[i])):
        question_array[i][j] = question_list[i][j]


for i in range(len(answer_list)):
    for j in range(len(answer_list[i])):
        answer_array[i][j] = answer_list[i][j]

for i in range(len(answer_begin)):
    begin_array[i] = answer_begin[i]

for i in range(len(answer_end)):
    end_array[i] = answer_end[i]

print(context_array.shape)
print(question_array.shape)
print(answer_array.shape)
print(begin_array.shape)
print(end_array.shape)

with h5py.File('context_test.h5', 'w') as hf:
    hf.create_dataset('context', data=context_array)
with h5py.File('questions_test.h5', 'w') as hf:
    hf.create_dataset('questions', data=question_array)
with h5py.File('answers_test.h5', 'w') as hf:
    hf.create_dataset('answers', data=answer_array)
with h5py.File('begin_test.h5', 'w') as hf:
    hf.create_dataset('begin', data=begin_array)
with h5py.File('end_test.h5', 'w') as hf:
    hf.create_dataset('end', data=end_array)

np.save('words_test.npy', words)
np.save('indxes_test', indexes)