# coding=utf-8
import sys
import os
reload(sys)
sys.setdefaultencoding('utf-8')
father_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(father_path)

import time
import numpy as np
from numpy import linalg as la

from pyspark.sql import SparkSession
from pyspark.mllib.feature import Word2VecModel
from pyspark.sql.functions import col
from pyspark.sql import SQLContext

from package_nlp import sentence_sim
from package_nlp import data_preprocessing
from package_nlp import sim_config
from package_nlp import words_cloud
print(sys.version)

sc = sim_config.config

num_slices = 6
host_sql = sc["host_sql"]
port_sql = sc["port_sql"]
user_sql = sc["user_sql"]
passwd_sql = sc["passwd_sql"]
db_name_sql = sc["db_name_sql"]
table_name_sql = sc["table_name_sql"]

host_mongo = sc["host_mongo"]
db_mongo_r = sc["db_mongo_r"]
port_mongo = sc["port_mongo"]

db_mongo_w = sc["db_mongo_w"]
sim_mongo_w = sc["sim_mongo_w"]

spark_temp = SparkSession.builder.master(
    "local").config("spark.some.config.option", "some-value").appName("minProject").getOrCreate()
print spark_temp

data_preprocessing.log_info("加载wordembedding模型......")
sqlContext = SQLContext(spark_temp)
lookup = sqlContext.read.parquet(
    "./data_temp/w2c.model/data").alias("lookup")
lookup_bd = spark_temp.sparkContext.broadcast(lookup.rdd.collectAsMap())


def cos_sim(vec_a, vec_b):
    """
    param vec_a : type->array
    param vec_b : type->array
    return similarity
    """
    mat_a = np.mat(vec_a)
    mat_b = np.mat(vec_b)
    num = float(mat_a * mat_b.T)
    denom = la.norm(mat_a) * la.norm(mat_b)
    if denom == 0:
        return 0
    else:
        return 0.5 + 0.5 * (num / denom)


def cal_sim(data_slice, wv_model, ss_vector):
    """
    param : data_slice  id && weight
    return:
    """
    id2words_temp = data_slice[1][:6]
    #
    array_temp = sentence_sim.sentence_embedding_weight(
        id2words_temp, wv_model)

    dot_temp = cos_sim(np.array(ss_vector), np.array(array_temp))

    return [data_slice[0], dot_temp]


class Fetch_Mongo():
    """
    """

    def __init__(self):
        self.log = {}

        self.jd_collection = data_preprocessing.connect_mongo(
            host_mongo,
            db_mongo_r,
            'seg_words_spark_jd',
            port_mongo)
        data_preprocessing.log_info("self.jd_collection")
        print self.jd_collection

        self.wb_collection = data_preprocessing.connect_mongo(
            host_mongo,
            db_mongo_r,
            'seg_words_spark_wb',
            port_mongo)
        data_preprocessing.log_info("self.wb_collection")
        print self.wb_collection

        self.data_temp = sentence_sim.read_mongo_weight(self.wb_collection)

    def _single_sentence(self, target_text):
        """

        """
        top_k = 6
        top_n = 5
        iterations = 30
        num_topics = 5
        stop_words_path = "./data_temp/stopWords.txt"

        ss = words_cloud.single_sentence(
            top_k,
            top_n,
            iterations,
            num_topics,
            stop_words_path
        )

        lda_model, doc_corpus = ss.sentence_transmit(
            target_sentence=target_text)
        print(lda_model)

        input_words = ss.sentence_to_lda_words(lda_model, doc_corpus)
        print("目标词汇:{0}".format(input_words))

        input_vector = sentence_sim.sentence_embedding_weight(
            target_words=input_words,
            wv_model=lookup_bd
        )
        return input_vector

    def proc(self, request):
        """
        param request ：
        """
        #
        start_time = time.time()
        data_temp = self.data_temp
        # print type(data_temp)

        ss_vector = self._single_sentence(request)
        print "输入语句的向量：", ss_vector
        #
        print ">>>>" * 10
        # 初始化spark
        spark_temp.sparkContext.setLogLevel("WARN")
        # RDD
        distData = spark_temp.sparkContext.parallelize(
            data_temp, numSlices=num_slices)

        distData.cache()
        # map & & reduce
        sentence_similarity = distData.map(
            lambda x: cal_sim(x, lookup_bd, ss_vector))
        # print sentence_similarity.collect()
        for item in sentence_similarity.collect():
            print "uid:{0}\tdoc_id:{1}\tsimilarity:{2}".format(item[0][0], item[0][1], item[1])
        print("Total Result:{0}".format(len(sentence_similarity.collect())))
        #
        del distData
        del sentence_similarity
        #
        spark_temp.stop()
        print "Total time:", time.time() - start_time


if __name__ == "__main__":
    fetch_mongo = Fetch_Mongo()
    fetch_mongo.proc(
        request="华为终端在巴塞罗那世界移动通信大会上发布发布了全新华为P系列智能手机——华为P10")
