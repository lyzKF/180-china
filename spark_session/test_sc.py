# coding=utf-8
import sys
import os
reload(sys)
sys.setdefaultencoding('utf-8')
father_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(father_path)

import time
from pyspark import SparkContext
from pyspark.mllib.feature import Word2Vec
from pyspark.mllib.feature import Word2VecModel
from pyspark.sql import SQLContext
from pyspark.sql.functions import col

# sc = SparkContext('local[*]', appName='Word2Vec')


# sqlContext = SQLContext(sc)

# lookup = sqlContext.read.parquet(
#     "./package_nlp/w2c.model/data").alias("lookup")

# # lookup_bd = sc.broadcast(lookup.rdd.collectAsMap())
# # rdd = sc.parallelize([u'人民', u'酸奶'])
# # temp = rdd.map(lambda ws: lookup_bd.value.get(ws)[:6])
# # print temp.collect()

# # from pyspark import SparkContext
# # from pyspark.mllib.feature import Word2Vec
# # from pyspark.mllib.feature import Word2VecModel

# # sc = SparkContext('local', appName='Word2Vec')

# # wv_model_local = Word2VecModel.load(
# #     sc, "./package_nlp/w2c.model")

# # x = wv_model_local.transform("人民")
# # print(x)


# gensim.models.ldamodel.LdaModel(
#     corpus=None,
#     num_topics=100,
#     id2word=None,
#     distributed=False,
#     chunksize=2000,
#     passes=1,
#     update_every=1,
#     alpha='symmetric',
#     eta=None,
#     decay=0.5,
#     offset=1.0,
#     eval_every=10,
#     iterations=50,
#     gamma_threshold=0.001,
#     minimum_probability=0.01,
#     random_state=None,
#     ns_conf=None,
#     minimum_phi_value=0.01,
#     per_word_topics=False,
#     callbacks=None,
#     dtype= < type 'numpy.float32' > )

#     train(
#         rdd,
#         k=10,
#         maxIterations=20,
#         docConcentration=-1.0,
#         topicConcentration=-1.0,
#         seed=None,
#         checkpointInterval=10,
#         optimizer='em')

# 主要写的使用spark ml 中的lda算法提取文档的主题的方法思路，不牵扯到lda的 算法原理。至于算法请参照http://www.aboutyun.com/thread-20130-1-1.html 这篇文章

# 使用lda算法对中文文本聚类并提取主题，大体上需要这么几个过程：

# 1.首先采用中文分词工具对中文分词，这里采用开源的IK分词。

# 2.从分词之后的词表中去掉停用词，生成新的词表。

# 3.利用文档转向量的工具将文档转换为向量。

# 4.对向量使用lda算法运算，运算完成之后取出主题的详情，以及主题在文档中的分布详情。
# //LDA主题模型的评价指标是困惑度，困惑度越小，模型越好


from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.clustering import LDA
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal


sc = SparkContext('local[*]', appName='Word2Vec')
data = [
    [1, Vectors.dense([0.0, 1.0, 0.5])],
    [3, Vectors.dense([0.9, 1.2, 0.4])]]
rdd = sc.parallelize(data)
model = LDA.train(
    rdd,
    k=2,
    maxIterations=40,
    docConcentration=-1.0,
    topicConcentration=-1.0,
    seed=100,
    checkpointInterval=10,
    optimizer='em')
print model.vocabSize()
print model.topicsMatrix()

# topics = model.describeTopics(1)
topics = model.topicsMatrix()

for word in topics:
    print word

# topics_rdd = topics.rdd
# topics_words = topics_rdd\
#     .map(lambda row: row['termIndices'])\
#     .map(lambda idx_list: [vocab[idx] for idx in idx_list])\
#     .collect()

# for idx, topic in enumerate(topics_words):
#     print("topic: ", idx)
#     print("----------")
#     for word in topic:
#         print(word)
#     print("----------")


# classmethod load(sc, path)
# # Load the LDAModel from disk.

# # Parameters:
# # sc – SparkContext
# # path – str, path to where the model is stored.

# save(sc, path)
# # Save the LDAModel on to disk.

# # Parameters:
# # sc – SparkContext
# # path – str, path to where the model needs to be stored.

# topicsMatrix()
# # Inferred topics, where each topic is represented by a distribution over terms.

# vocabSize()
# # Vocabulary size (number of terms or terms in the vocabulary)
