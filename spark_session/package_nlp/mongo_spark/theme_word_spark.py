# -*- coding: UTF-8 -*-
from __future__ import unicode_literals
from gensim import corpora, models
import sys
reload(sys)
sys.setdefaultencoding('utf8')
sys.path.append("..")
print(sys.version)
import jieba
import time
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
import data_preprocessing
top_k = 6
top_n = 5
iterations = 30
num_topics = 5


class single_sentence():

    def __init__(self, top_k, top_n, iterations, num_topics, stop_words):
        """
        """
        self.top_k = top_k
        self.top_n = top_n
        self.iterations = iterations
        self.num_topics = num_topics
        self.stop_words = stop_words
        jieba.initialize()

    def sentence_transmit(self, target_sentence):
        """
        param target_sentence:
        return:
        """

        seg_doc_temp = list(
            jieba.cut(target_sentence, cut_all=False, HMM=True))

        target_doc_without_swords = [
            word for word in seg_doc_temp if word not in self.stop_words]

        if target_doc_without_swords:
            seg_words_temp = [target_doc_without_swords]

            word_dict = corpora.Dictionary(seg_words_temp)

            doc_corpus = [word_dict.doc2bow(text) for text in seg_words_temp]

            lda_model = models.LdaModel(
                doc_corpus,
                num_topics=self.num_topics,
                id2word=word_dict,
                alpha='auto',
                eta='auto',
                minimum_probability=0.01,
                iterations=self.iterations)

            return lda_model, doc_corpus[0]
        else:
            return [], []

    def sentence_to_lda_words(self, lda_model, doc_corpus):
        """
        param doc_corpus : lda模型
        param lda_model  : doc语料
        return:
        """

        topic_probability_temp = lda_model.get_document_topics(
            doc_corpus,
            minimum_probability=None,
            minimum_phi_value=None,
            per_word_topics=False
        )

        theme_word_dict = dict()
        for item in topic_probability_temp:

            topic_to_word_temp = lda_model.print_topic(
                item[0], topn=self.top_n)

            tw_temp = topic_to_word_temp.strip().split("+")

            for object_temp in tw_temp:

                obj_temp = object_temp.strip().split("*")
                obj_theme_word = obj_temp[1].replace("\"", "")

                if obj_theme_word in theme_word_dict.keys():
                    theme_word_dict[obj_theme_word] += float(
                        obj_temp[0]) * item[1]
                else:
                    theme_word_dict[obj_theme_word] = float(
                        obj_temp[0]) * item[1]

        theme_word_dict = sorted(
            theme_word_dict.items(), key=lambda k: k[1], reverse=True)
        theme_word_dict = [(item[0], item[1]) for item in theme_word_dict]

        target_words = theme_word_dict[:self.top_k]

        return target_words


def lda_single_sentence(sentence, ss):
    """
    param : sentence 语料
    return:
    """
    lda_model, doc_corpus = ss.sentence_transmit(target_sentence=sentence[-1])
    if doc_corpus:

        target_words_dict = ss.sentence_to_lda_words(lda_model, doc_corpus)

        if len(target_words_dict) >= 3:

            return [sentence[0], sentence[1], target_words_dict]


def run(host_mongo, port_mongo, func_mongo, *args, **kwargs):
    """
    param host_mongo :
    param port_mongo :
    param func_mongo :
    param *args
    param **kwargs   :
    """
    start_time = time.time()
    spark_temp = SparkSession.builder.master(
        "local").config("spark.some.config.option", "some-value").appName("readMongo_jd").getOrCreate()
    sqlContext = SQLContext(spark_temp)

    db_collection = data_preprocessing.connect_mongo(
        host_mongo, kwargs['db_fetch'], kwargs['table_fetch'], port_mongo)
    data_preprocessing.log_info("链接{0}数据库:{1}\n".format(
        kwargs['db_fetch'], db_collection))

    v_data = func_mongo(db_collection, *args)
    print v_data[0][-1]

    db_insert = data_preprocessing.connect_mongo(
        kwargs['host_write'], kwargs['db_write'], kwargs['table_write'], kwargs['port_write'])
    data_preprocessing.log_info(
        "链接{0}数据库:{1}\n".format(kwargs['db_write'], db_insert))

    stop_words_iteration = data_preprocessing.read_data_from_file(
        "/home/lgl/repos_rpc/spark_session/data_temp/stopWords.txt")
    stop_words = [unicode(word, "utf-8") for word in stop_words_iteration]

    ss = single_sentence(top_k, top_n, iterations, num_topics, stop_words)

    # 初始化spark
    spark_temp.sparkContext.setLogLevel("WARN")
    # RDD
    distData = spark_temp.sparkContext.parallelize(
        v_data, numSlices=3)

    distData.cache()
    # map & & reduce
    theme_words_temp = distData.map(
        lambda x: lda_single_sentence(x, ss))
    #
    for item in theme_words_temp.collect():
        data_dict_temp = dict()
        if item:
            # import chardet
            data_dict_temp["id"] = [item[0], item[1]]
            data_list_temp = list()
            for _ in item[-1]:

                data_list_temp.append([_[0], _[1]])
            data_dict_temp["seg_word"] = data_list_temp

            db_insert.insert(data_dict_temp)

    # print("Total Result:{0}".format(len(theme_words_temp.collect())))
    #
    del distData
    print "Total time:{0}".format(time.time() - start_time)
    spark_temp.stop()


if __name__ == "__main__":
    host_mongo = '192.168.3.207'
    port_mongo = 27017

    # import theme_word_jd as jd
    # run(
    #     host_mongo,
    #     port_mongo,
    #     jd.read_mongo,
    #     *(3, 800000),
    #     db_fetch="JDKB",
    #     table_fetch="JDKB_detail",
    #     host_write="192.168.3.207",
    #     db_write="theme_words",
    #     table_write="test_jd",
    #     port_write=27017
    # )

    import theme_word_wb as wb
    run(
        host_mongo,
        port_mongo,
        wb.read_mongo,
        *(3, 800000),
        db_fetch="Weibo",
        table_fetch="Weibo_detail",
        host_write="192.168.3.207",
        db_write="theme_words",
        table_write="test_wb",
        port_write=27017
    )
