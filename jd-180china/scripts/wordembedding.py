#!/opt/Anaconda3/bin/python
# -*- coding: UTF-8 -*-

# ********************************************************
# * Author        : lyz
# * Email         : lyz038015@163.com
# * Create time   : 2018-01-19 14:57
# * Last modified : 2018-01-19 14:57
# * Filename      : wordembedding.py
# * Description   : 
# *********************************************************

from gensim.models import Word2Vec
#
from merge_origin_data import data_clean, text_segmentation 
import jd
#
model_path = '../model/wv_gensim'
# 融合数据
dc = data_clean()
data_temp = dc.merge_data()
#
text_seg = text_segmentation(data_temp = data_temp, stop_words_path = './stopWords.txt', word_threshold = 2)
# 
stop_words = text_seg.read_stop_word()
seg_words_without_sword = text_seg.cut_remove_word_function(stop_words)
# word to vector 
jd.log_info("训练Word2Vec语言模型")
model = Word2Vec(seg_words_without_sword, sg=1, size=100,  window=5,  min_count=5,  negative=3, sample=0.001, hs=1)
"""
1.sg=1是skip-gram算法，对低频词敏感；默认sg=0为CBOW算法。
2.size是输出词向量的维数，值太小会导致词映射因为冲突而影响结果，值太大则会耗内存并使算法计算变慢，一般值取100~200。
3.window是句子中当前词与目标词之间的最大距离，3表示在目标词前看3-b个词，后面看b个词（b在0-3之间随机）。
4.min_count是对词进行过滤，频率小于min-count的单词则会被忽视，默认值为5。
5.negative和sample可根据训练结果进行微调，sample表示更高频率的词被随机下采样到所设置的阈值，默认值为1e-3。
6.hs=1表示层级softmax将会被使用，默认hs=0且negative不为0，则负采样将会被选择使用。
7.workers控制训练的并行，此参数只有在安装了Cpython后才有效，否则只能使用单核。
"""
# 模型保存
jd.log_info("wordembedding模型保存")
model.save(model_path)
# 模型提取
# model = wc.load(model_path)
# 相似度计算
print("\a\a词语之间的相似度:{0}".format(model.wv.similarity('牛奶', '酸奶')))  
# 输出词的向量表示 raw numpy vector of a word
print("\a\a词的向量表示:{0}".format(model.wv['好喝']))    
#输出array([-0.00449447, -0.00310097,  0.02421786, ...], dtype=float32)

