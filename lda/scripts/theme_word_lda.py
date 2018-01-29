#!/opt/Anaconda3/bin/python
# -*- coding: UTF-8 -*-

# ********************************************************
# * Author        : lyz
# * Email         : lyz038015@163.com
# * Create time   : 2018-01-17 13:16
# * Last modified : 2018-01-25 14:56
# * Filename      : merge_origin_data.py
# * Description   : 
# *********************************************************
"""
\n类函数text_segmentation主要实现功能有：\n\a\a1. 用户自定义词典的加载\n\a\a2. 停用词表的读取\n\a\a3. 文本分词、去停用词
\n类函数train_lda主要实现功能有：\n\a\a1. 加载数据\n\a\a2. 训练LDA模型\n\a\a3. 查询主题详细信息\n\a\a4. 根据主题中的主题词提取文章的关键词\n\a\a5. LDA模型的评测，主要包括模型的log_perplexity和Coherence

"""
print(__doc__)
import csv, os,sys
import re
import pickle
import jieba
import jieba.analyse
import codecs
from gensim import corpora, models

import pyLDAvis.gensim
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing
import time
from optparse import OptionParser
# 导入自己的库，避免重复造轮子
import read_all_files
sys.path.append("/User/lyz/reposity-git/lda")
from p_library import self_log 

USAGE = "usage: python extract_tags_with_weight.py [file name] -k [topk]"

parser = OptionParser(USAGE)
parser.add_option("-k", dest="topk")
opt, args = parser.parse_args()


class data_clean():
    #
    def __init__(self):
        self_log.log_info("data_clean")

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

    def extract_tf_idf_keywords(self, user_dictionary_path, merge_data):
        """
        param user_dictionary_path : 用户自定义字典路径
        Detial : 通过tf_idf提取关键词，并将其作为用户自己的词典
        """
        with open(user_dictionary_path, 'w', encoding='utf-8') as writer:
            # 
            for sentence in merge_data:
                seg_words_temp = jieba.analyse.extract_tags(sentence[2], topK = 3)
                # 关键词写入文本文档
                for item in seg_words_temp:
                    writer.write(item+'\n')
    


class text_segmentation():
    #
    def __init__(self, stop_words_path, data_temp):
        """
        param stop_words_path: 停用词表存储路径
        param data_temp: 评论信息列表，每个对象是一个tuple，包含productid与comments
        param word_threshold: 低频词阈值
        """
        self_log.log_info("text_segmentation")
        self.stop_words_path = stop_words_path
        self.data_temp = data_temp
        # 手动初始化jieba
        jieba.initialize()
        # 添加字典
        jieba.add_word("特仑苏")
        jieba.add_word("蒙牛")
        jieba.add_word("京东")
        jieba.add_word("伊利")
        jieba.add_word("三元")
        jieba.add_word("快递小哥")
        jieba.add_word("好喝")
        jieba.add_word("挺好喝")
        jieba.add_word("味道不错")
        jieba.add_word("超快")
        jieba.add_word("超便宜")
        
        # 用户自定义词表
        jieba.load_userdict("./userdict.txt")
        

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
        # 去停用词后的分词列表对象
        seg_words_without_sword = list()
        # 分词处理
        for sentence in self.data_temp:
            # 临时存储分词对象
            words_temp = list()
            # 精准模式分词，打开HMM模型发现新词
            cut_words_temp = jieba.cut(sentence[2], cut_all= False, HMM=True)
            # print '/'.join(split_word_list)
            # type : list
            cut_words_temp = [word for word in list(cut_words_temp)]
            # 去停用词
            for word in cut_words_temp:
                if word not in stop_words_temp:
                    words_temp.append(word)
            seg_words_without_sword.append(words_temp)
        return seg_words_without_sword


class lda_model():

    def __init__(self, topk):
        self_log.log_info("lda_model")
        # 主题词topk
        self.topk = topk

    def get_dataset(self):
        """
        """
        # 定义类对象
        dc = data_clean()

        # 融合数据
        # data_temp = dc.merge_data()
        dir_path = '../product_info'
        d_file = read_all_files.dir_files(dir_path = dir_path)
        data_temp = d_file.read_files()

        # 提取tf_idf关键词
        if os.path.exists('./userdict.txt'):
            self_log.log_info("用户自定义词典已经存在")
        else:
            dc.extract_tf_idf_keywords('./userdict.txt', data_temp)

        # pkl文件存储路径
        pkl_file_temp = '../model/seg_words.pkl'

        # 查看分词文件是否存在。如果已经存在，直接加载分词结果
        if os.path.exists(pkl_file_temp):
            self_log.log_info("分词已经存在，加载中......")
            # 读取pkl文件
            seg_words_without_sword = dc.get_data_from_pkl(pkl_file_temp)
        else:
            self_log.log_info("分词处理中......")
            # 定义类对象
            text_seg = text_segmentation(data_temp=data_temp, stop_words_path='./stopWords.txt')
            """
            param data_temp : 融合后的数据集合
            param stop_words_path : 停用词表存储文档
            """
            self_log.log_info("读取停用词表......")
            # 读取停用词表，返回列表对象
            stop_words = text_seg.read_stop_word()
            
            self_log.log_info("分词、去停用词处理......")
            # 分词处理、去停用词。需要把seg_words_without_sword保存下来
            seg_words_without_sword = text_seg.cut_remove_word_function(stop_words)
            
            self_log.log_info("保存分词处理后的结果......")
            # 分词结果存储
            dc.store_data(pkl_file_temp, seg_words_without_sword)
            """
            param pkl_file_temp : 分词结果存储文档路径
            param seg_words_without_sword : 去除停用词后的分词结果
            """
            self_log.log_info("删除停用词表，清理内存......")
            # 清理内存，删除停用词、分词列表对象
            del stop_words
        return seg_words_without_sword

    def train_lda(self, seg_words_without_sword):
        """
        param seg_words_without_sword : 去停用词后的分词结果
        return : 字典、语料、LDA模型对象
        """
        self_log.log_info("生成字典......")
        # 生成字典
        word_dict = corpora.Dictionary(seg_words_without_sword)
        
        # 生成语料
        corpus = [word_dict.doc2bow(text) for text in seg_words_without_sword]
        
        # 模型加载、训练
        if os.path.exists('../model/lda.model'):
            self_log.log_info("LDA模型已经训练好，加载模型.......")
            # 加载模型
            lda = models.LdaModel.load('../model/lda.model')
        else:
            self_log.log_info("训练LDA模型.......")
            # 训练模型
            lda = models.LdaModel(corpus = corpus, id2word=word_dict, iterations = 60,num_topics=50)
            # 模型存储
            self_log.log_info("保存模型")
            lda.save("../model/lda.model")
        # 返回训练模型
        return lda, corpus, word_dict
    
    def show_topic_detial(self, corpus, lda):
        """
        param corpus : 语料
        Detials      :
        """
        # 语料长度
        corpus_len = len(corpus)

        # 输出第n条评论属于某个主题的概率
        number_corpus = input("输入文章编号........\n")
        number_corpus = int(number_corpus)

        # 判定number_corpus在合理的范围内
        assert (number_corpus < corpus_len) == True
        assert (-1 < number_corpus) == True
        
        # 输出主题概率分布
        self_log.log_info('输出第{0}条评论属于每个主题的概率分布'.format(number_corpus))
        topic_temp = lda.get_document_topics(
                corpus[number_corpus],          #
                minimum_probability = None,     #
                minimum_phi_value = None,       #
                per_word_topics = False         #
                )
        print("\a\a主题概率排序前:{0}".format(topic_temp))

        # 对主题概率排序
        topic_temp.sort(key = lambda k:k[1], reverse = True)
        print("\a\a主题概率排序后:{0}".format(topic_temp))

        # 输出主题概率最高的主题分布
        self_log.log_info("输出评分最高的主题分布")
        print ("%s\t%0.3f\n"%(lda.print_topic(topic_temp[0][0],topn=5), topic_temp[0][1]))

        # 输出每个主题的概率分布
        self_log.log_info("输出每个主题的概率分布")
        for item in topic_temp:
            print ("主题词相关度:%s\n主题概率:%0.3f\n"%(lda.print_topic(item[0],topn=5), item[1]))  
        
    def calculate_doc_keywords(self, item_corpus):
        """
        param item_corpus : 语料对象
        """
        # 选择再次加载LDA模型，主要是为了实现程序的并行化
        if os.path.exists('../model/lda.model'):
            # 加载模型
            lda = models.LdaModel.load('../model/lda.model')
        else:
            print("缺少训练模型.......")
            return 0
        # 输出item_corpus属于某个主题的概率
        topic_probability_temp = lda.get_document_topics(
                item_corpus,                      #
                minimum_probability = None,       #
                minimum_phi_value = None,         #
                per_word_topics = False           #
                )
        # 主题词字典
        theme_word_text = open('./theme_words.txt', 'a', encoding='utf-8')
        theme_word_dict = dict()
        for item in topic_probability_temp:
            # 
            topic_to_word_temp = lda.print_topic(item[0], topn =5)
            # -0.340 * "category" + 0.298 * "$M$" + 0.183 * "algebra" + -0.174 * "functor" + -0.168 * "operator"
            # 字符串分割，提取主题词对应的权重
            tw_temp = topic_to_word_temp.strip().split('+') 
            # 
            for object_temp in tw_temp:
                obj_temp = object_temp.strip().split('*')
                # 统计主题词相对文章的权重
                if obj_temp[1] in theme_word_dict.keys():
                    theme_word_dict[obj_temp[1]] += float(obj_temp[0])*item[1]
                else:
                    theme_word_dict[obj_temp[1]] = float(obj_temp[0])*item[1]
            """
            item[1] : 主题相对于文章的权重
            obj_item[0] : 主题词相对于主题的权重
            """
        # 降序排序
        theme_word_dict = sorted(theme_word_dict.items(), key = lambda k : k[1], reverse = True)
        # 写入txt文档
        theme_word_text.write(str(theme_word_dict[:int(self.topk)-1]) + '\n')
        # 
        theme_word_text.close()

    def evaluation_model(self, lda, corpus, word_dict, seg_words_without_sword):
        """
        param lda       : LDA模型 
        param corpus    : 语料
        param word_dict : 字典
        Details : 
        """
        # 语言模型的perplexity
        log_perplexity = lda.log_perplexity(corpus)
        print("\a\a模型评价perplexity:{0}".format(log_perplexity))
        
        # Coherence Model:Quantitative approach
        # u_mass
        lda_coherence_umass = CoherenceModel(model = lda, corpus = corpus, dictionary = word_dict, coherence = 'u_mass')
        # c_v
        lda_coherence_cv = CoherenceModel(model = lda, texts = seg_words_without_sword, dictionary = word_dict, coherence = 'c_v')
        # compute coherence based on u_mass and c_v
        coherence_umass = lda_coherence_umass.get_coherence()
        coherence_cv = lda_coherence_cv.get_coherence()
        # 输出模型Coherence
        self_log.log_info("\n输出Coherence........")
        print("\a\aCoherence_umass:{0}\nCoherence_cv:{1}".format(coherence_umass, coherence_cv))
        
        # pyLDAvis 
        lda_vis = pyLDAvis.gensim.prepare(lda, corpus, word_dict)
        # LDA主题展示
        pyLDAvis.show(lda_vis)




def run():
    """
    return:
    """
    # 
    if len(args) < 0:
        print(USAGE)
        sys.exit(1)
    topk =opt.topk
    #  定义类对象
    ldamodel = lda_model(topk = topk) 
    seg_words_without_sword = ldamodel.get_dataset()
    lda, corpus, word_dict = ldamodel.train_lda(seg_words_without_sword = seg_words_without_sword)
    
    # ldamodel.evaluation_model(lda, corpus, word_dict, seg_words_without_sword)
    
    # ldamodel.show_topic_detial(corpus,lda)
    
    #
    starttime = time.time()
    self_log.log_info("开启多进程,执行程序中......")
    cpu_count = multiprocessing.cpu_count()
    pool = Pool(cpu_count)
    pool.map(ldamodel.calculate_doc_keywords, corpus)
    pool.close()
    pool.join()
    print("running time:{0}".format(time.time()-starttime))
    # theme_word = ldamodel.calculate_doc_keywords(item_corpus = corpus[0], lda = lda)


#    #查看训练集中第1个样本的主题分布
#    self_log.log_info('查看训练集中第1个样本的主题分布')
#    test_doc=seg_words_without_sword[0]
#    #文档转换成bow  
#    doc_bow = word_dict.doc2bow(test_doc)     
#    
#    #得到新文档的主题分布  
#    doc_lda = lda[doc_bow]                   
#    #输出新文档的主题分布  
#    print (doc_lda)  
#    for topic in doc_lda: 
#        print ("%s\t%0.3f\n"%(lda.print_topic(topic[0],topn=15), topic[1]))  
#


if __name__ == "__main__":
    #
    run()

