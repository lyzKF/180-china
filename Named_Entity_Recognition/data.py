#!/opt/Anaconda3/bin/python
# -*- coding: UTF-8 -*-

# ********************************************************
# * Author        : lyz
# * Email         : lyz038015@163.com
# * Create time   : 2018-02-05 10:25
# * Last modified : 2018-02-05 10:57
# * Filename      : data.py
# * Description   : 
# *********************************************************

import sys, pickle, os, random
import numpy as np

## tags, BIO
tag2label = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-LOC": 3,
        "I-LOC": 4,
        "B-ORG": 5,
        "I-ORG": 6
        }


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    param corpus_path : 
    return: data
    """
    data = list()
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = list(), list()
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []
    return data


def vocab_build(vocab_path, corpus_path, min_count):
    """
    BUG: I forget to transform all the English characters from full-width into half-width... 
    param vocab_path  : pkl路径
    param corpus_path : 语料路径
    param min_count   : 低频词汇阈值
    return            : None
    """
    # 调用read_corpus函数读取数据
    data = read_corpus(corpus_path)
    # 遍历数据
    word2id = dict()
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    # 添加低频词
    low_freq_words = list()
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>':
            low_freq_words.append(word)
    
    # 删除word2id中的低频词
    for word in low_freq_words:
        del word2id[word]
    
    # 重新调整word的id
    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    
    # 添加起始标志
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    # 存储word2id为pkl文件
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


def sentence2id(sent, word2id):
    """
    param sent    : sentence
    param word2id : word2id
    return        : sentence2id
    """
    sentence_id = list()
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])

    return sentence_id


def read_dictionary(vocab_path):
    """
    param vocab_path : pkl文件路径
    return           : word2id
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """
    param vocab         : 
    param embedding_dim : 
    return              : embedding_matric
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """
    param sequences : sequences
    param pad_mark  : 待补充元素，默认为0
    return          : seq_len_list
    """
    # 获取sequences中最大的长度
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = list(), list()
    # 
    for seq in sequences:
        seq = list(seq)
        # 不到max_len部分，进行补pad_mark处理
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """
    param data       : 
    param batch_size : 
    param vocab      : 
    param tag2label  : 
    param shuffle    : 是否打乱数据
    return           : seqs && labels
    """
    # 打乱数据
    if shuffle:
        random.shuffle(data)
    # 
    seqs, labels = list(), list()
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels

