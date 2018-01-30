#!/opt/Anaconda3/bin/python
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
import codecs
import numpy as np
import csv
from optparse import OptionParser

parser = OptionParser()
parser.add_option(
        "-s",
        dest = "pos",
        help = "查询对象下标")
opt, args = parser.parse_args()
# 查询对象位置
pos = opt.pos
assert (pos > 0) == True
#
model_path = '../model/wv_gensim'
# 融合数据
def get_data(file_path):
    """
    param file_path : 文件路径名称
    """
    with open(file_path, 'r',encoding="utf-8") as reader:
        #
        lines = reader.readlines()
        #
        all_theme_words = list()
        for line in lines:
            #
            theme_words = list()
            line = line.replace("[",'').replace("]",'').replace("(",'').replace("\'",'')
            line = line.strip().split('),')
            for item in line:
                #
                item = item.strip().split(',')
                item_temp = item[0]
                theme_words.append(item_temp)
            #
            all_theme_words.append(theme_words)
    #
    return all_theme_words

#
product_theme_words = get_data(file_path = './product_theme_words.txt')
v_theme_words = get_data(file_path = './v_theme_words.txt')

#
assert len(product_theme_words[0]) == len(v_theme_words[0])

#
data_temp = product_theme_words + v_theme_words
#

# word to vector 
model = Word2Vec(
        data_temp,
        sg = 1,
        size = 100, 
        window = 5, 
        min_count = 1, 
        negative = 3, 
        sample = 0.001, 
        hs = 1)
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
# model.save(model_path)

# 获取所有词的vector，并组成矩阵
theme_words_vector = list()

for item in data_temp:
    #
    list_temp = list()
    for word in item:
        #
        vector_temp = model.wv[word]
        list_temp.append(vector_temp)
    #
    array_temp = np.array(list_temp)
    #
    array_temp_mean = np.mean(array_temp, axis = 0)
    #
    list_temp_mean = array_temp_mean.tolist()
    #
    theme_words_vector.append(array_temp_mean)    

# 计算相似度评分
for j in range(10,len(theme_words_vector)):
    dot_temp = np.dot(theme_words_vector[pos], theme_words_vector[j])
    dot_temp = "%.6f"% dot_temp
    print("{0}*{1}={2}:".format(data_temp[pos], data_temp[j],dot_temp))


# 模型提取
# model = wc.load(model_path)
# 相似度计算
#print("\a\a词语之间的相似度:{0}".format(model.wv.similarity('牛奶', '酸奶')))  
# 输出词的向量表示 raw numpy vector of a word
#print("\a\a词的向量表示:{0}".format(model.wv['好喝']))    
#输出array([-0.00449447, -0.00310097,  0.02421786, ...], dtype=float32)

