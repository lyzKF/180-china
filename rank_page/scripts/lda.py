# -*- coding: UTF-8 -*-

# ********************************************************
# * Author        : lyz
# * Email         : lyz038015@163.com
# * Create time   : 2018-01-17 13:16
# * Last modified : 2018-01-29 15:29
# * Filename      : theme_test.py
# * Description   : 
# *********************************************************
"""
\n类函数pre_data主要实现功能：\n\a\a1. 读取mongodb数据库，提取数据；\n\a\a2. 抽取关键词，生成用户自定义词典；
\n类函数text_segmentation主要实现功能有：\n\a\a1. 用户自定义词典的加载；\n\a\a2. 文本分词、去停用词；
\n类函数train_lda主要实现功能有：\n\a\a1. 训练LDA模型；\n\a\a2. 根据主题中的主题词提取文章的关键词；\n\a\a3. LDA模型的评测，主要包括模型的log_perplexity和Coherence；
\n函数run是执行函数，添加了多进程；
"""
print(__doc__)
import os, sys
import re
import pickle
import jieba
import jieba.analyse
import codecs
from gensim import corpora, models
import numpy as np
from gensim.models.coherencemodel import CoherenceModel
from multiprocessing import Pool
import multiprocessing
import time

sys.path.append("/home/arron/ligl/")
from p_library import data_preprocessing 


class pre_data():
    """
    function : 数据预处理
    """
    def __init__(self):
        data_preprocessing.log_info("pre_data")
        # 
        jieba.initialize()
    
    def read_mongo(self, data_mongodb_path):
        """
        param data_mongo_path : 
        return :
        """
        data_preprocessing.log_info("读取数据中.......")
        
        if os.path.exists(data_mongodb_path):
            
            data_preprocessing.log_info("读取pkl文件")
            
            data_temp = data_preprocessing.get_data_from_pkl(data_mongodb_path)
        else:
            data_preprocessing.log_info("读取数据库")
            # 具有可选参数text_len,默认为0
            data_temp = data_preprocessing.get_data_from_mongo()
            
            data_preprocessing.store_data_to_pkl(data_mongodb_path, data_temp)

        return data_temp

    def extract_keywords(self, data_temp, k, user_dict_path):
        """
        param data_temp      : 
        param topk           :
        param user_dict_path :
        """
        print("\t原始数据长度:{0}".format(len(data_temp.keys())))
        data_pre = dict()
        user_dict_writer = open(user_dict_path, 'w', encoding="utf-8")

        for index, line in data_temp.items():
            
            sentence_temp = data_preprocessing.extract_chinese(line)
            
            try:
                if index and sentence_temp:
                    
                    data_pre[index] = sentence_temp
                    
                    seg_words_temp = jieba.analyse.extract_tags(sentence_temp, topK = k)
                     
                    for item in seg_words_temp:
                        if item:
                            user_dict_writer.write(item+'\n')
            except Exception as e:
                print("Error info :{0}".format(index))
                print(e)
        user_dict_writer.close()
        print("\t处理后数据长度:{0}".format(len(data_pre.keys())))


class text_segmentation():
    """
    function : 分词
    """
    def __init__(self, data_temp):
        """
        param data_temp  :
        """
        data_preprocessing.log_info("text_segmentation")
        self.data_temp = data_temp
        
        jieba.add_word("人民")
         
        jieba.load_userdict("../data/userdict.txt")
        
    def seg_sentence(self, stop_words_temp):
        """
        param stop_words_temp :
        """
        seg_words_without_sword = list()
        i = 1
        sentence2id = dict()
        sen2id_file = open("../data/sen2id.txt", 'w')
         
        for index, sentence in self.data_temp.items():
             
            words_temp = list()
            
            cut_words_temp = jieba.cut(sentence, cut_all= False, HMM=True)
             
            cut_words_temp = [word for word in list(cut_words_temp)]
            
            for word in cut_words_temp:
                if word not in stop_words_temp:
                    words_temp.append(word)

            seg_words_without_sword.append([words_temp)
            sen2id_file.write(str(i)+"->"+str(index))
            i += 1
        sen2id_file.close()
        return seg_words_without_sword



class lda_model():

    def __init__(self, topk, topn, theme_words_file):
        data_preprocessing.log_info("lda_model")
        
        self.topk = topk
        self.topn = topn
        self.theme_words_file = theme_words_file

    def train_lda(self, seg_words_without_sword, iterations, num_topics):
        """
        param seg_words_without_sword : 
        return :
        """
        data_preprocessing.log_info("生成字典......")
        
        word_dict = corpora.Dictionary(seg_words_without_sword)
        
        corpus = [word_dict.doc2bow(text) for text in seg_words_without_sword]
        
        data_preprocessing.log_info("训练模型......")
        
        lda = models.LdaModel(corpus = corpus, id2word=word_dict, iterations = iterations, num_topics = num_topics)
        
        data_preprocessing.log_info("模型保存......")
        lda.save("../model/lda.model")
        
        return lda, corpus, word_dict
        
    def calculate_doc_keywords(self, item_corpus):
        """
        param item_corpus : 语料对象
        """
        if os.path.exists('../model/lda.model'):
            lda = models.LdaModel.load('../model/lda.model')
        else:
            print("缺少训练模型.......")
            return 0
        
        topic_probability_temp = lda.get_document_topics(
                item_corpus,                      
                minimum_probability = None,       
                minimum_phi_value = None,       
                per_word_topics = False           
                )
        
        theme_word_text = open(self.theme_words_file, 'a', encoding='utf-8')
        theme_word_dict = dict()
        for item in topic_probability_temp:
            
            topic_to_word_temp = lda.print_topic(item[0], topn = topn)
            
            tw_temp = topic_to_word_temp.strip().split('+') 
             
            for object_temp in tw_temp:
            
                obj_temp = object_temp.strip().split('*')
                obj_theme_word = obj_temp[1].replace("\"",'')
            
                if obj_theme_word in theme_word_dict.keys():
                    theme_word_dict[obj_theme_word] += float(obj_temp[0])*item[1]
                else:
                    theme_word_dict[obj_theme_word] = float(obj_temp[0])*item[1]
        
        theme_word_dict = sorted(theme_word_dict.items(), key = lambda k : k[1], reverse = True)
        
        theme_word_text.write(str(theme_word_dict[:int(self.topk)]) + '\n')
         
        theme_word_text.close()

    def evaluation_model(self, lda, corpus, word_dict, seg_words_without_sword):
        """
        param lda       : LDA模型 
        param corpus    : 语料
        param word_dict : 字典
        Details : 
        """
    
        log_perplexity = lda.log_perplexity(corpus)
        print("\a\a模型评价perplexity:{0}".format(log_perplexity))
        
        lda_coherence_umass = CoherenceModel(model = lda, corpus = corpus, dictionary = word_dict, coherence = 'u_mass')
        
        lda_coherence_cv = CoherenceModel(model = lda, texts = seg_words_without_sword, dictionary = word_dict, coherence = 'c_v')
        
        coherence_umass = lda_coherence_umass.get_coherence()
        coherence_cv = lda_coherence_cv.get_coherence()
        
        self_log.log_info("\n输出Coherence........")
        print("\a\aCoherence_umass:{0}\nCoherence_cv:{1}".format(coherence_umass, coherence_cv))
        


def run():
    """
    return:
    """
    k = 5
    user_dict_path = "../data/userdict.txt"
    data_mongodb_path = "../data/data_mongo.pkl"
    stop_words_path = "./stopWords.txt"
    seg_result_path = "../data/seg_words_without_stopwords.pkl"
    theme_words_file = "../data/theme_words_file.txt"
    iterations = 60
    num_topics = 10
    topk = 5
    topn = 10
    
    pre_d = pre_data()
    data_temp = pre_d.read_mongo(data_mongodb_path)
    data_preprocessing.log_info("抽取用户字典......")
    pre_d.extract_keywords(data_temp, k, user_dict_path)
    
    text_seg = text_segmentation(data_temp)

    lda_object = lda_model(topk, topn, theme_words_file)
    
    data_preprocessing.log_info("获取停用词表......")
    stop_words_iteration = data_preprocessing.read_data_from_file(stop_words_path)
    stop_words = [word for word in stop_words_iteration]

    data_preprocessing.log_info("分词......")
    seg_words_without_sword = text_seg.seg_sentence(stop_words)
    data_preprocessing.store_data_to_pkl(seg_result_path, seg_words_without_sword)
    
    data_preprocessing.log_info("模型训练......")
    lda, corpus, word_dict = lda_object.train_lda(seg_words_without_sword, iterations, num_topics)

    starttime = time.time()
    data_preprocessing.log_info("开启多进程，执行程序......")
    cpu_count = multiprocessing.cpu_count()
    pool = Pool(cpu_count)
    pool.map(ldamodel.calculate_doc_keywords, corpus)
    pool.close()
    pool.join()
    print("running time:{0}".format(time.time()-starttime))

if __name__ == "__main__":

    run()

