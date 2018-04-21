# _*_ coding:utf-8 -*-
import json
import sys
print(sys.version)
from gensim.models import Word2Vec
import package_nlp.sentence_sim as sentence_sim
import package_nlp.words_cloud as words_cloud
import package_nlp.data_preprocessing as data_preprocessing
from package_nlp import sim_config

import ctypes
SYS_gettid = 186

print(sys.version)
sc = sim_config.config
# config
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
stop_words_path = sc["stop_words_path"]


def _calculate_sim(wv_model_local, r_collection, w_collection, project_id, target_text, source_id=1):
    """
    param : project_id   项目ID
    param : target_text  词云文本
    return:
    """
    data_preprocessing.log_info("开始执行similarity的计算>>>>>>>>>>>>>>>>>>")
    libc = ctypes.cdll.LoadLibrary('libc.so.6')
    tid = libc.syscall(SYS_gettid)
    print("当前线程id{0}".format(tid))

    top_k = 6
    top_n = 5
    iterations = 30
    num_topics = 5

    data_preprocessing.log_info("获取主题词......")
    id2words = sentence_sim.read_mongo_weight(r_collection)

    data_preprocessing.log_info("获取主题词向量......")
    theme_sentence_vector = dict()
    for key in id2words:
        if len(id2words[key]) > 6:
            id2words_temp = id2words[key][:6]
        else:
            id2words_temp = id2words[key]
        array_temp = sentence_sim.sentence_embedding_weight(
            id2words_temp, wv_model_local)
        theme_sentence_vector[key] = array_temp
    del id2words

    data_preprocessing.log_info("获取输入信息主题词向量......")
    ss = words_cloud.single_sentence(
        top_k, top_n, iterations, num_topics, stop_words_path)
    lda_model, doc_corpus = ss.sentence_transmit(
        target_sentence=target_text)
    print(lda_model)

    input_words = ss.sentence_to_lda_words(lda_model, doc_corpus)
    print("目标词汇:{0}".format(input_words))

    data_preprocessing.log_info("计算sentenceembedding......")

    input_vector = sentence_sim.sentence_embedding_weight(
        target_words=input_words, wv_model=wv_model_local)
    print input_vector

    sentence_sim.show_sim(
        project_id,
        input_vector,
        theme_sentence_vector,
        w_collection,
        str(source_id)
    )
    data_preprocessing.log_info("线程{0}文本similarity是计算结束。".format(tid))


class EgModuleWordCloud:
    """
    """

    def __generate_wordcloud(self, project_id, target_text):
        """
        param : project_id   项目ID
        param : target_text  词云文本
        return:
        """
        sql_conn = words_cloud.connect_mysql(
            host_sql,
            user_sql,
            passwd_sql,
            db_name_sql,
            port_sql)
        data_preprocessing.log_info("self.sql_conn")
        print sql_conn

        top_k = 60
        top_n = 80
        iterations = 30
        num_topics = 5

        data_preprocessing.log_info("获取输入信息主题词向量......")
        ss = words_cloud.single_sentence(
            top_k, top_n, iterations, num_topics, stop_words_path)
        lda_model, doc_corpus = ss.sentence_transmit(
            target_sentence=target_text)
        target_words = ss.sentence_to_lda_words(lda_model, doc_corpus)

        target_words_dict = dict()
        for item in target_words:
            target_words_dict[item[0]] = int(item[1] * 1000)

        data_preprocessing.log_info("文本主题词数据入库......")
        words_cloud.insert_mysql(sql_conn, table_name_sql, json.dumps(
            target_words_dict, encoding="utf8", ensure_ascii=False), project_id)

    def __init__(self):
        self.log = {}

        data_preprocessing.log_info("加载wordembedding模型......")
        self.wv_model_local = Word2Vec.load("./package_nlp/wv/word2vec_wx")

        self.jd_collection = data_preprocessing.connect_mongo(
            host_mongo,
            db_mongo_r,
            'words_jd_10',
            port_mongo)
        data_preprocessing.log_info("self.jd_collection")
        print self.jd_collection

        self.wb_collection = data_preprocessing.connect_mongo(
            host_mongo,
            db_mongo_r,
            'words_wb_10',
            port_mongo)
        data_preprocessing.log_info("self.wb_collection")
        print self.wb_collection

        self.w_collection = data_preprocessing.connect_mongo(
            host_mongo,
            db_mongo_w,
            sim_mongo_w,
            port_mongo)
        data_preprocessing.log_info("self.w_collection")
        print self.w_collection

    def proc(self, request):
        """
        param request ：
        """
        #
        params = json.loads(request)
        # print type(params)
        # print params.keys()
        # print params

        project_id = params['project_id']
        print ">>>>>>>>>>>>>>>>>>>>>project_id:{0}".format(project_id)
        target_text = params['parameter_list']
        print target_text
        # baike_text = target_text['project_info']
        # print baike_text
        data_preprocessing.log_info("生成词云...........................")
        self.__generate_wordcloud(project_id, target_text)

        data_preprocessing.log_info("计算相似度...........................")
        _calculate_sim(self.wv_model_local, self.wb_collection, self.w_collection,
                       project_id, target_text, source_id=1)

        _calculate_sim(self.wv_model_local, self.jd_collection, self.w_collection,
                       project_id, target_text, source_id=2)
#


if __name__ == "__main__":

    request_temp = {
        "project_id": "42",
        "project_start_time": 1519920000,
        "project_end_time": 1521388800,
        ""
        "parameter_list":
        {
            "brand_product": "华为 手机",
            "project_info": "北京时间2017年2月26日，华为终端在巴塞罗那世界移动通信大会2017（MWC）上发布发布了全新华为P系列智能手机——华为P10&P10Plus.P系列是华为终端产品中定位时尚精致的高端旗舰系列。华为P10&P10Plus延续了P系列的经典基因，还加入了顶尖技术，其中摄影功能迎来了更高的突破，外观ID设计进行了重大的革新，并辅以特殊工艺，视觉层面更注重艺术细节美。同时，在科技体验方面，华为P10&P10Plus也较上一代有重大突破，在继续发扬核心性能优势的基础上，聚焦系统流畅性、续航和摄影等消费者痛点进行了全方位的改进和优化。[1]华为P10华为P10&P10Plus于2017年2月26日在巴塞罗那世界移动通信大会2017（MWC）正式发布。华为P10&P10Plus产品定位时尚精致的高端旗舰系列，产品主打功能点为“人像摄影艺术”。"
        }
    }
    tt = EgModuleWordCloud()
    tt.proc(request=json.dumps(request_temp))

    # # similarity测试
    # params = json.loads(json.dumps(request_temp))
    # project_id = params['project_id']
    # target_text = params['parameter_list']
    # baike_text = target_text['project_info']

    # cs = cal_sim()
    # cs.calculate_sim(project_id, baike_text, sid_temp=2)
