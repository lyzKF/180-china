# -*- coding: UTF-8 -*-
# ********************************************************
# * Author        : lyz
# * Email         : lyz038015@163.com
# * Create time   : 2018-01-19 14:57
# * Last modified : 2018-01-30 11:32
# * Filename      : wordembedding.py
# * Description   : 
# *********************************************************

from gensim.models import Word2Vec
from gensim import corpora, models
import numpy as np
import csv
import sys
import jieba
import os
from numpy import linalg as la 

sys.path.append("/home/lgl/")
from p_library import data_preprocessing
import theme_word_one

def read_mongo():
    """
    param collection: mongo数据连接对象
    return 
    """
    collection = data_preprocessing.connect_mongo(
            '192.168.3.104', 
            'theme_words', 
            'words', 
            27017)
    # 
    id2words = dict()
    result_temp = collection.find()
    if not list(result_temp):
        print("monog数据集合为空")
        sys.exit()
    #
    for index,item in enumerate(result_temp):
        
        assert len(list(item.keys())) == 2
        if list(item.keys())[0].isdigit():
            id_temp = list(item.keys())[0]
        else:
            id_temp = list(item.keys())[1]
        
        assert type(id_temp) == str
        assert type(item) == dict
        
        words_temp = item[id_temp]
        
        words = [word[0] for word in words_temp]
        
        id2words[id_temp] = words
    return id2words
        

def sentence_embedding(target_words, wv_model):
    """
    param target_words:
    return:
    """
    list_temp = list()
    unknow_words_pos = list()
    for index, word in enumerate(target_words):
        try:
            vector_temp = wv_model.wv[word][:64]
        except Exception as e:
            #print(word)
            unknow_words_pos.append(index)
            vector_temp = np.array([0] * 64)

        list_temp.append(vector_temp)
    if len(unknow_words_pos) >= 3:
        array_temp_mean = [0]*64
    else:
        array_temp = np.array(list_temp)

        for un_index in unknow_words_pos:
            array_temp[un_index,:] = np.mean(array_temp, axis = 0)
        array_temp_mean = np.mean(array_temp, axis = 0)

    return array_temp_mean


def cos_sim(vec_a,vec_b):  
    """
    param vec_a : type->array
    param vec_b : type->array
    return similarity
    """
    vec_a = np.mat(vec_a)
    vec_b = np.mat(vec_b)
    num = float(vec_a*vec_b.T)  
    denom = la.norm(vec_a)*la.norm(vec_b)
    if denom == 0:
        return 0
    else:
        return 0.5 + 0.5*(num/denom)


def test_sim(theme_sentence_vector):
    """
    param theme_sentence_vector : 所有主题词的vector，type->dict
    """
    key_vector = list(theme_sentence_vector.keys())[0]
    value_vector = theme_sentence_vector[key_vector]
    
    if len(set(value_vector.tolist())) == 1 and np.sum(set(value_vector.tolist())) == 0:
        print("输入错误")
        print(value_vector)
    else:
        print("输出相似度信息")
        print(value_vector)

        data_preprocessing.log_info("输出sim信息......")
        for vec_idx in theme_sentence_vector.keys():

            dot_temp = cos_sim(value_vector, theme_sentence_vector[vec_idx])
            dot_temp = "%.3f" % dot_temp 
            print("{0}->ID:{1} Sim:{2}".format(key_vector, vec_idx, dot_temp))


def show_sim(input_vecor, theme_sentence_vector):
    """
    param input_vector : 输入sentence的vec
    param theme_sentence_vector :
    """
    
    if len(set(input_vector)) == 1 and np.sum(set(input_vector)) == 0:
        print("输入错误")
        print(input_vector)
    else:
        print("输出相似度信息")
        
        data_preprocessing.log_info("输出sim信息......")
        for vec_idx in theme_sentence_vector.keys():

            dot_temp = cos_sim(np.array(input_vector), np.array(theme_sentence_vector[vec_idx]))
            print("ID:{0} Sim:{1}".format(vec_idx, dot_temp))

    

def run(rpc_params):
    """
    return:
    """
    data_preprocessing.log_info("获取出入文本信息......")
    assert type(rpc_params) == str
    params = eval(rpc_params)
    project_id = params['project_id']
    target_text = params['parameter_list']

    data_preprocessing.log_info("获取主题词......")
    id2words = read_mongo()
    
    data_preprocessing.log_info("加载wordembedding模型......")
    model_path = "../wv/word2vec_wx"
    wv_model_local = Word2Vec.load(model_path)
    
    data_preprocessing.log_info("获取主题词向量......")
    theme_sentence_vector = dict()
    for key in id2words:
        
        array_temp = sentence_embedding(id2words[key], wv_model_local)
        theme_sentence_vector[key] = array_temp
    
    # data_preprocessing.log_info("测试similarity......")
    # test_sim(theme_sentence_vector)
    
    data_preprocessing.log_info("获取输入信息主题词向量......")
    stop_words_path = "./stopWords.txt"
    top_k = 60
    top_n = 80
    iterations = 30
    num_topics = 5
    
    ss = theme_word_one.single_sentence(top_k, top_n, iterations, num_topics, stop_words_path)

    lda_model, doc_corpus = ss.sentence_transmit(target_sentence = target_text)

    input_words = ss.sentence_to_lda_words(lda_model, doc_corpus)

    print("目标词汇:{0}".format(input_words))

    data_preprocessing.log_info("计算sentenceembedding......")
    input_vector = sentence_embedding(target_words = input_words, wv_model = wv_model_local)


if __name__ == "__main__":
    run(rpc_params)



