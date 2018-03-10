# -*- coding: UTF-8 -*-
# ********************************************************
# * Author        : lyz
# * Email         : lyz038015@163.com
# * Create time   : 2018-01-19 14:57
# * Last modified : 2018-03-07 17:04
# * Filename      : wordcloud.py
# * Description   : 
# *********************************************************

from gensim.models import Word2Vec
from gensim import corpora, models
import numpy as np
import sys
import jieba
import os, time
import pymysql
from multiprocessing import Pool
import multiprocessing

sys.path.append("/home/lgl/")
from p_library import data_preprocessing

def read_mongo(collection, text_len = 3):
    """
    param mongo_conn : mongo数据库连接对象
    param table_name : mongo数据库表名
    param text_len   : 文本长度阈值
    return mongo数据
    """
    #
    result_temp = collection.find()
    #
    index = 0
    v_data = dict()
    while(index < 950000):
        item = result_temp[index]
        if len(item["text"]) > text_len and item:
            text_temp = item["text"]
            id_temp = item["V_id"]
            v_data[id_temp] = text_temp
        index += 1

    return v_data


class single_sentence():

    def __init__(self, top_k, top_n, iterations, num_topics, stop_words_path):
        """
        param top_k : 从输入sentence中选择最终的输出长度。eg : top_k = 4, 最后返回的list长度为4
        param top_n : 每个topic选择词的长度。eg : top_n = 4，每个topic选择权重较高的四个词
        param iterations : LDA模型的迭代次数
        param num_topic : LDA模型的主体数量
        param stop_words_path : 停用词路径
        """
        self.top_k = top_k
        self.top_n = top_n
        self.iterations = iterations
        self.num_topics = num_topics
        self.stop_words_path = stop_words_path

    def sentence_transmit(self, target_sentence):
        """
        param target_sentence : 输入的sentence
        return
        """

        target_sentence = data_preprocessing.extract_chinese(target_sentence)
        
        stop_words_iteration = data_preprocessing.read_data_from_file(self.stop_words_path)
        stop_words = [word for word in stop_words_iteration]
        
        seg_doc_temp = list(jieba.cut(target_sentence, cut_all = False, HMM = True))

        target_doc_without_swords = [word for word in seg_doc_temp if word not in stop_words]

        seg_words_temp = [target_doc_without_swords]

        word_dict = corpora.Dictionary(seg_words_temp)

        doc_corpus = [word_dict.doc2bow(text) for text in seg_words_temp]

        lda_model = models.LdaModel(
                doc_corpus, 
                num_topics = self.num_topics, 
                id2word = word_dict, 
                alpha='auto', 
                eta='auto', 
                minimum_probability = 0.01,
                iterations = self.iterations)

        del stop_words
    
        return lda_model, doc_corpus[0]

    def sentence_to_lda_words(self, lda_model, doc_corpus):
        """
        param doc_corpus : lda模型
        param lda_model  : doc语料
        return:
        """

        topic_probability_temp = lda_model.get_document_topics(
            doc_corpus,
            minimum_probability = None,
            minimum_phi_value = None,
            per_word_topics = False
            )

        theme_word_dict = dict()
        for item in topic_probability_temp:
            
            topic_to_word_temp = lda_model.print_topic(item[0], topn=self.top_n)

            tw_temp = topic_to_word_temp.strip().split("+")

            for object_temp in tw_temp:

                obj_temp = object_temp.strip().split("*")
                obj_theme_word = obj_temp[1].replace("\"", "")

                if obj_theme_word in theme_word_dict.keys():
                    theme_word_dict[obj_theme_word] += float(obj_temp[0]) * item[1]
                else:
                    theme_word_dict[obj_theme_word] = float(obj_temp[0]) * item[1]

        theme_word_dict = sorted(theme_word_dict.items(), key=lambda k: k[1], reverse=True)
        theme_word_dict = [(str(item[0]), item[1]) for item in theme_word_dict]

        target_words = theme_word_dict[:self.top_k]
        
        return target_words


def lda_single_sentence(sentence):
    """
    param : sentence 语料
    return
    """
    lda_words = dict()
    stop_words_path = "./stopWords.txt"

    top_k = 60

    top_n = 80

    iterations = 30

    num_topics = 5

    ss = single_sentence(top_k, top_n, iterations, num_topics, stop_words_path)

    lda_model, doc_corpus = ss.sentence_transmit(target_sentence = sentence)

    target_words_dict = ss.sentence_to_lda_words(lda_model, doc_corpus)

    v_data_sen = list(v_data.values())
    key_temp = list(v_data.keys())[v_data_sen.index(sentence)]

    lda_words[key_temp] = target_words_dict
    
    db_insert = data_preprocessing.connect_mongo(host, 'theme_words', 'words', port)
    # db_insert.remove() 慎用，如果不明白，不要轻易取消此行注释!!!!!!!!!!!!
    db_insert.insert(lda_words)

    return lda_words


def run():
    #
    jieba.initialize()
    # read_mongo_data
    data_preprocessing.log_info("文本主题词数据入库......")
    global host
    host = '192.168.3.104'
    # user = 'root' 
    # user_passwd = '112233'
    db_name = 'Weibo'
    table_name = 'Weibo_detail'
    global port
    port = 27017
    #
    # db_conn = data_preprocessing.connect_mongo(host, user, user_passwd, db_name, port)
    db_collection = data_preprocessing.connect_mongo(host, db_name, table_name, port)
    
    global v_data
    v_data = read_mongo(db_collection)
    v_data_sen = list(v_data.values())
    print(v_data_sen[:3])
    
    # 并行加速
    starttime = time.time()
    data_preprocessing.log_info("开启多进程,执行程序中......")
    cpu_count = multiprocessing.cpu_count()
    pool = Pool(cpu_count)
    theme_words_total = pool.map(lda_single_sentence, v_data_sen)
    pool.close()
    pool.join()
    data_preprocessing.log_info("多进程执行结束......")
    print("running time:{0}".format(time.time()-starttime))


if __name__ == "__main__":
    run()



