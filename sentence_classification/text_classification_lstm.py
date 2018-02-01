#!/opt/Anaconda3/bin/python
# -*- coding: UTF-8 -*-

# ********************************************************
# * Author        : lyz
# * Email         : lyz038015@163.com
# * Create time   : 2018-01-31 17:50
# * Last modified : 2018-02-01 13:44
# * Filename      : text_classification_lstm.py
# * Description   : 
# *********************************************************
import csv
import sys,os
import pandas as pd
import numpy as np
import jieba
import pickle,time

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import  load_model


# Functions
#
def predict_one(s):
    """
    """
    s = np.array(doc2num(list(jieba.cut(s,cut_all=False)),maxlen))
    s = s.reshape((1,s.shape[0]))
    return model.predict_classes(s,verbose=0)[0][0]

#
def doc2num(sentence,maxlen):
    """
    """
    sentence = [i for i in sentence if i in word_set]
    sentence = sentence[:maxlen] + ['']*max(0,maxlen-len(sentence))
    return list(dict_num[sentence])


# read csv files
bx=pd.read_csv('./data/article_1.csv',header=None)
yh=pd.read_csv('./data/article_0.csv',header=None)

# label data set manually
bx['label'] = 1
yh['label'] = 0

# merge data set
merge_bx_yh = pd.concat([bx,yh],ignore_index=True)

# split words with jieba
merge_bx_yh['words'] = merge_bx_yh[0].apply(lambda s: list(jieba.cut(s,cut_all=False)))

# parameters
maxlen = 40
min_count = 1

# read all split-words objects to a list
content = []
for i in merge_bx_yh['words'][1:]:
	content.extend(i)

# count the number of each word
dict_num = pd.DataFrame(pd.Series(content).value_counts())[0]

dict_num = dict_num[dict_num >= min_count]
dict_num[:] = range(1,len(dict_num)+1)
dict_num[''] =0
word_set = list(set(dict_num.index))

#
merge_bx_yh['doc2num'] = merge_bx_yh['words'].apply(lambda s: doc2num(s,maxlen))

#
index = range(len(merge_bx_yh))
np.random.shuffle(index)
merge_bx_yh = merge_bx_yh.loc[index]

# built data set
x = np.array(list(merge_bx_yh['doc2num']))
y = np.array(list(merge_bx_yh['label']))
y = y.reshape((-1,1))

# read models or built a new model
starttime = time.time()
if os.path.exists('./models/lstm_sentence.model'):
    print ('use a trained model')
    model = load_model('./models/lstm_sentence.model')
else :
    print('Build model...')
    model = Sequential()
    model.add(Embedding(len(dict_num), 256,input_length=maxlen))
    model.add(LSTM(output_dim=128,activation='sigmoid',recurrent_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(
            loss = 'binary_crossentropy',
            optimizer = 'adam', 
            metrics = ['accuracy'])

train_num = int(0.7*len(x))

model.fit(
        x[:train_num], 
        y[:train_num], 
        batch_size=16,
        epochs=40,
        verbose=0,
        shuffle=False)
#
score = model.evaluate(x[train_num:], y[train_num:])
print ('The running time:',time.time()-starttime)
print ('Test score:', score[0])
print ('Test accuracy:', score[1])

# save models
output = open('./models/w2cid.dict','wb')
pickle.dump(dict,output)
output.close()
model.save('./models/lstm_sentence.model')








