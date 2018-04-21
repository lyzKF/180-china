# -*- coding: UTF-8 -*-
import numpy as np
import sys
import os
reload(sys)
sys.setdefaultencoding('utf-8')
import time
from numpy import linalg as la

current_path = os.path.abspath(sys.argv[0])
father_path = os.path.abspath(
    os.path.dirname(current_path) + os.path.sep + ".")
sys.path.append(father_path)
from package_nlp import data_preprocessing


def read_mongo(collection):
    """
    param collection: mongo数据连接对象
    return 
    """
    #
    id2words = dict()
    result_temp = collection.find({}, {"_id": 0})
    #
    for index, item in enumerate(result_temp):

        id_list_temp = [id_temp for id_temp in list(
            item.keys()) if id_temp.isdigit()]
        assert len(id_list_temp) == 1
        id_temp = id_list_temp[0]

        uid_temp = item["uid"]

        words_temp = item[id_temp]

        words = [word[0] for word in words_temp]

        id2words[id_temp, uid_temp] = words
    return id2words


def read_mongo_weight(collection):
    """
    param collection: mongo数据连接对象
    return 
    """
    #
    id2words = dict()
    result_temp = collection.find({}, {"_id": 0})
    #
    for index, item in enumerate(result_temp):

        id_list_temp = [id_temp for id_temp in list(
            item.keys()) if id_temp.isdigit()]
        assert len(id_list_temp) == 1
        id_temp = id_list_temp[0]

        uid_temp = item["uid"]
        words_temp = item[id_temp]

        id2words[id_temp, uid_temp] = words_temp

    return id2words


def sentence_embedding(target_words, wv_model):
    """
    param target_words:
    return: sentence embedding 如果unknow words超过40 %，则舍弃。参考每个words对整个sentence的贡献度相同
    """
    list_temp, nknow_words_pos = list(), list()

    total_len = float(len(target_words))
    for index, word in enumerate(target_words):
        word = word[0]
        try:
            vector_temp = wv_model.wv[word][:64]
        except Exception as e:
            # print(word)
            unknow_words_pos.append(index)
            vector_temp = np.array([0] * 64)

        list_temp.append(vector_temp)
    if len(unknow_words_pos) / total_len >= 0.4:
        array_temp_mean = [0] * 64
    else:
        array_temp = np.array(list_temp)

        for un_index in unknow_words_pos:
            array_temp[un_index, :] = np.mean(array_temp, axis=0)
        array_temp_mean = np.mean(array_temp, axis=0)

    return array_temp_mean


def sentence_embedding_weight(target_words, wv_model):
    """
    param target_words:
    return: sentence embedding, 依据Latent Dirichlet Allocation的结果，
            参考每个词weight的不同，计算该word对sentence的贡献度
    """
    list_temp, weight_temp, unknow_words_pos = list(), list(), list()

    total_len = float(len(target_words))

    for index, words in enumerate(target_words):
        word = words[0]
        word_weight = words[1]

        try:
            vector_temp = wv_model.wv[word][:64]
        except Exception as e:
            # print(word)
            unknow_words_pos.append(index)
            vector_temp = np.array([0] * 64)

        list_temp.append(vector_temp)
        weight_temp.append(word_weight)
    if len(unknow_words_pos) / total_len >= 0.4:
        array_temp_mean = [0] * 64
    else:
        array_temp = np.array(list_temp)
        # weight normal
        try:
            normal_weight = [(item - min(weight_temp)) /
                             float(max(weight_temp) - min(weight_temp)) for item in weight_temp]
        except Exception as e:
            len_temp = len(weight_temp)
            normal_weight = [1 / float(len_temp)] * len_temp
        for un_index in unknow_words_pos:
            array_temp[un_index, :] = np.mean(array_temp, axis=0)
        # 每行赋权重
        for i in range(array_temp.shape[0]):
            array_temp[i, :] = array_temp[i, :] * normal_weight[i]
        array_temp_mean = [0] * 64
        # 对应元素相加
        for item in array_temp:
            array_temp_mean += item
        array_temp_mean = array_temp_mean.tolist()
    return array_temp_mean


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


def show_sim(project_id, input_vector, theme_sentence_vector, w_collection, source_id):
    """
    param project_id : 项目id
    param input_vector : 输入sentence的vec
    param theme_sentence_vector :
    """
    time_temp = int(time.time())
    timeArray = time.localtime(time_temp)
    stamp_temp = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)

    if len(set(input_vector)) == 1 and np.sum(set(input_vector)) == 0:
        print("输入错误")
        print(input_vector)
    else:
        print("输出相似度信息")
        w_collection.remove(
            {"project_id": project_id, "source_id": source_id})

        print("当前数据总量:{0}".format(w_collection.find(
            {"project_id": project_id, "source_id": source_id}).count()))

        data_preprocessing.log_info("数据入库操作，执行中......")
        index = 0
        for vec_idx in theme_sentence_vector.keys():

            id2sim_temp = dict()
            dot_temp = cos_sim(np.array(input_vector), np.array(
                theme_sentence_vector[vec_idx]))

            id2sim_temp["doc_id"] = vec_idx[0]
            id2sim_temp["project_id"] = project_id
            id2sim_temp["uid"] = vec_idx[1]
            id2sim_temp["source_id"] = source_id
            id2sim_temp["similarity"] = dot_temp
            id2sim_temp["update_time"] = stamp_temp

            w_collection.insert(id2sim_temp)
            index += 1
            if index % 1000 == 0:
                print("》》》》》》》》》》》数据已经更新{0}条".format(index))

        total_number = w_collection.find({}).count()
        print("插入数据后，数据总量:{0}".format(total_number))
