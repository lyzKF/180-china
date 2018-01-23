#!/opt/Anaconda3/bin/python
# -*- coding: UTF-8 -*-

# ********************************************************
# * Author        : lyz
# * Email         : lyz038015@163.com
# * Create time   : 2018-01-17 13:16
# * Last modified : 2018-01-23 23:12
# * Filename      : merge_origin_data.py
# * Description   : 
# *********************************************************

import csv, os
import re
import pickle
import jieba
import jieba.analyse
import codecs
from gensim import corpora, models, similarities

import pyLDAvis.gensim
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
# 导入自己的库，避免重复造轮子
import jd

class data_clean():
    #
    def __init__(self):
        print("*" * 60)
        jd.log_info("\n类函数data_clean主要实现功能有：\n\a\a1. 数据读取、及存储\n\a\a2. 数据清洗\n\a\a3. 数据合并")
        print("*" * 60)

    def extract_chinese(slef, str):
        """
        param str: 字符串对象
        return: 中文文本
        """
        line = str.strip()
        # 中文的编码范围:\u4e00到\u9fa5
        p2 = re.compile(r'[^\u4e00-\u9fa5]')
        #
        zh = " ".join(p2.split(line)).strip()
        zh = ",".join(zh.split())
        Outstr = zh
        # 返回经过相关处理后得到中文的文本
        return Outstr

    def store_data(self, pkl_path, data_object):
        """
        param pkl_path: pkl文件保存路径
        param data_object: 数据保存对象
        return: None
        """
        # 创建写入文件
        pkl_f = open(pkl_path, 'wb')
        # pickle本地保存文件
        pickle.dump(data_object, pkl_f)
        pkl_f.close()
        # 计算文件大小
        file_size_temp = os.path.getsize(pkl_path)
        print("文件存档完成，文件大小为:{0:.3f}M".format(file_size_temp/(1024.0*1024.0)))

    def get_data_from_pkl(self, pkl_path):
        """
        param pkl_path: pkl文件存储路径
        return: 本地数据
        """
        with open(pkl_path, 'rb') as pkl_f:
            data_temp = pickle.load(pkl_f)
        return data_temp

    def read_data(self, file_path):
        """
        file_path: 商品信息存储路径
        return: 存储商品信息的列表
        """
        # 商品信息列表
        content_dict = list()
        # 将商品信息添加到列表中，并返回该列表
        with open(file_path, 'r', encoding='utf-8') as f:
            # csv.Error: line contains NULL byte
            try:
                lines = csv.reader((line.replace('\0', '') for line in f))
                #
                for line in lines:
                    if line[-1]:
                        if line[-1].startswith('%'):
                            pass
                        # 文本清洗
                        line_01_temp = self.extract_chinese(str=line[-1])
                        # 商品id与商品评论append到content_list
                        content_dict.append([line[0], line_01_temp])
                    else:
                        print("\a\a用户{0}->商品{1}的评价记录不合法".format(line[1], line[0]))
            except Exception as e:
                print(e)
        return content_dict

    def merge_data(self):
        """
        function: 读取文件，做数据清洗，然后合并数据
        """
        # 本地数据文件路径
        file_path_mengniu = '../mengniu/' + 'data/product_information_mengniu.csv'
        file_path_sanyuan = '../sanyuan/' + 'data/product_information_sanyuan.csv'
        # 数据清洗，保留评论中的中文信息
        list_mengniu = self.read_data(file_path=file_path_mengniu)
        list_sanyuan = self.read_data(file_path=file_path_sanyuan)
        # 数据合并
        final_data = list_mengniu + list_sanyuan
        # 数据返回
        return final_data
    #
    def show_topic(self,perplexity, coherence):
        """
        """
        plt.figure()
        num_topics = [x for x in range(2,5,1)]
        plt.plot(perplexity, num_topics, 'b*')
        plt.plot(coherence, num_topics, 'r')
        plt.xlabel("Num_topics")
        plt.ylabel("Quanlity")
        plt.title('perplexity & coherence')
        plt.legend()
        plt.show()



class text_segmentation():
    #
    def __init__(self, stop_words_path, data_temp, word_threshold):
        """
        param stop_words_path: 停用词表存储路径
        param data_temp: 评论信息列表，每个对象是一个tuple，包含productid与comments
        param word_threshold: 低频词阈值
        """
        self.stop_words_path = stop_words_path
        self.data_temp = data_temp
        self.word_threshold = word_threshold
        # 手动初始化jieba
        jieba.initialize()
        # 添加字典
        jieba.add_word("特仑苏")
        jieba.add_word("蒙牛")
        jieba.add_word("京东")
        jieba.add_word("伊利")
        jieba.add_word("三元")

    #
    def read_stop_word(self):
        """
        return: 读取停用词表
        """
        # 停用词列表
        stop_word = list()
        with codecs.open(self.stop_words_path, 'r') as reader:
            lines = reader.readlines()
            for line in lines:
                line = line.strip()
                stop_word.append(line)
        print("\a\a停用词表大小:{0}".format(len(stop_word)))
        return stop_word

    #
    def cut_remove_word_function(self, stop_words_temp):
        """
        return: 评论信息分词
        """
        seg_words_without_sword = list()
        # 分词处理
        for sentence in self.data_temp:
            # 临时存储分词对象
            words_temp = list()
            # cut_words_temp = jieba.cut(sentence[1], cut_all= False, HMM=True)
            cut_words_temp = jieba.analyse.extract_tags(sentence[1], topK=3)
            # print '/'.join(split_word_list)
            # type : list
            cut_words_temp = [word for word in list(cut_words_temp)]
            for word in cut_words_temp:
                if word not in stop_words_temp:
                    words_temp.append(word)
            seg_words_without_sword.append(words_temp)
        return seg_words_without_sword


def run():
    """
    return:
    """
    #  定义类对象
    dc = data_clean()
    # 融合数据
    data_temp = dc.merge_data()
    pkl_file_temp = '../model/seg_words.pkl'
    if os.path.exists(pkl_file_temp):
        jd.log_info("分词已经存在，加载中......")
        seg_words_without_sword = dc.get_data_from_pkl(pkl_file_temp)
    else:
        jd.log_info("分词处理中......")
        #
        text_seg = text_segmentation(data_temp=data_temp, stop_words_path='./stopWords.txt', word_threshold=100)
        # 读取停用词
        jd.log_info("读取停用词表......")
        stop_words = text_seg.read_stop_word()
        # 分词处理、去停用词。需要把seg_words_without_sword保存下来
        jd.log_info("分词、去停用词处理......")
        seg_words_without_sword = text_seg.cut_remove_word_function(stop_words)
        #
        jd.log_info("保存分词处理后的结果......")
        dc.store_data(pkl_file_temp, seg_words_without_sword)
        # 清理内存，删除停用词、分词列表对象
        del stop_words
    # 生成字典
    jd.log_info("生成字典......")
    word_dict = corpora.Dictionary(seg_words_without_sword)
    corpus = [word_dict.doc2bow(text) for text in seg_words_without_sword]
    #
    """
    # jd.log_info("训练tf_idf模型......")
    # tfidf = models.TfidfModel(corpus)
    # corpus_tfidf = tfidf[corpus]
    """
    # 模型训练
    jd.log_info("训练LDA模型.......")
    # 
    lda = models.LdaModel(corpus = corpus, id2word=word_dict, iterations = 2,num_topics=10)
    
    # 输出每条评论属于某个主题的概率
    jd.log_info('输出第一条评论属于每个主题的概率分布')
    topic_temp = lda.get_document_topics(corpus[0], minimum_probability=None, minimum_phi_value=None, per_word_topics=False)
    print(topic_temp)
    topic_temp.sort(key = lambda k:k[1], reverse = True)
    print(topic_temp)
    print ("%s\t%0.3f\n"%(lda.print_topic(topic_temp[0][0],topn=5), topic_temp[0][1]))  
    for item in topic_temp:
        print ("%s\t%0.3f\n"%(lda.print_topic(item[0],topn=15), item[1]))  

    #查看训练集中第三个样本的主题分布
    jd.log_info('查看训练集中第1个样本的主题分布')
    test_doc=seg_words_without_sword[0]
    #文档转换成bow  
    doc_bow = word_dict.doc2bow(test_doc)     
    
    #得到新文档的主题分布  
    doc_lda = lda[doc_bow]                   
    #输出新文档的主题分布  
    print (doc_lda)  
    for topic in doc_lda: 
        print ("%s\t%0.3f\n"%(lda.print_topic(topic[0],topn=15), topic[1]))  

    #
    log_perplexity = lda.log_perplexity(corpus)
    print("\a\a模型评价perplexity:{0}".format(log_perplexity))
    """
    # corpus_tfidf: tf-idf词向量
    # id2word: a map from id to word
    # num_topics: the number of topics
    """
    # Coherence Model:Quantitative approach
    # u_mass
    # lda_coherence = CoherenceModel(model = lda, corpus = corpus, dictionary = word_dict, coherence = 'u_mass')
    # c_v
    lda_coherence = CoherenceModel(model = lda, texts = seg_words_without_sword, dictionary = word_dict, coherence = 'c_v')
    coherence = lda_coherence.get_coherence()
    print("\a\aCoherence:{0}".format(coherence))
    #
    # dc.show_topic(perplexity_temp, coherence_temp)
    # lda_vis = pyLDAvis.gensim.prepare(lda, corpus, word_dict)
    #
    # jd.log_info("保存模型")
    # lda.save("../model/lda.model")
    """
    # lda模型对象训练tf-idf词向量
    # corpus_lda = lda[corpus_tfidf]
    # print(corpus_lda)
    """
    # result = lda.get_document_topics(corpus[0], per_word_topics=True)
    # print(result)
    # result_topic = lda.get_topic_terms(topicid=2, topn= 10)
    # print("模型评价perplexity:{0}".format(lda.log_perplexity(chun)))
    # for item in result_topic:
        # print(item)
    # LDA主题展示
    # pyLDAvis.show(lda_vis)
if __name__ == "__main__":
    #
    run()

