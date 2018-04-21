# _*_ coding:utf-8 -*-

from gensim import corpora, models
import sys
import os
reload(sys)
sys.setdefaultencoding('utf8')
import jieba
import os
import json
import pymysql

current_path = os.path.abspath(sys.argv[0])
father_path = os.path.abspath(
    os.path.dirname(current_path) + os.path.sep + ".")
sys.path.append(father_path)
from package_nlp import data_preprocessing


def connect_mysql(host_sql, user, user_passwd, db_name, port_sql):
    """
    param user : 用户名
    param user_passwd : 密码
    param db_name : 数据库名
    return : 
    """
    # 打开数据库连接
    db_conn = pymysql.connect(
        host=host_sql,
        database=db_name,
        user=user,
        password=user_passwd,
        port=port_sql,
        charset='utf8'
    )
    return db_conn

#


def insert_mysql(db_conn, table_sql, insert_object, project_id):
    """
    param db_conn : mysql数据库的链接对象
    param insert_object : 插入数据对象
    return None
    """
    try:
        with db_conn.cursor() as cursor:
            # Create a new record
            cursor.execute("SELECT * FROM %s WHERE  project_id = '%d'" %
                           (table_sql, int(project_id)))
            db_conn.commit()
            content_temp = cursor.fetchone()

            if not content_temp:
                cursor.execute("INSERT INTO %s (project_id, topic_words) VALUES ('%d', '%s')" % (
                    table_sql, int(project_id), insert_object))
                db_conn.commit()
            else:
                # cursor.execute("INSERT INTO %s (project_id, topic_words) VALUES ('%d', '%s')" % (
                #     table_sql, int(project_id), insert_object))
                cursor.execute("UPDATE %s SET topic_words = '%s' WHERE project_id = '%d'" % (
                    table_sql, insert_object, int(project_id)))
                db_conn.commit()
        # db_conn is not autocommit by default. So you must commit to save
        db_conn.commit()
    except Exception as e:
        print(e)
        db_conn.rollback()
        print("Fialed")
    finally:
        db_conn.close()


class single_sentence():

    def __init__(self, top_k, top_n, iterations, num_topics, stop_words_path):
        """

        """
        jieba.initialize()
        self.top_k = top_k                      # 最终返回前top_k个词
        self.top_n = top_n                      # 每个主题选取多少个词
        self.iterations = iterations            # lda模型迭代次数
        self.num_topics = num_topics            # 主题个数
        self.stop_words_path = stop_words_path  # 停用词路径

    def sentence_transmit(self, target_sentence):
        """
        param target_sentence:
        return:
        """

        stop_words_iteration = data_preprocessing.read_data_from_file(
            self.stop_words_path)
        stop_words = [word for word in stop_words_iteration]

        seg_doc_temp = list(
            jieba.cut(target_sentence, cut_all=False, HMM=True))

        target_doc_without_swords = [
            word for word in seg_doc_temp if word not in stop_words]

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
