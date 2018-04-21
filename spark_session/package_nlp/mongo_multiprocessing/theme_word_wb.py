# -*- coding: UTF-8 -*-

import sys
import data_preprocessing


def read_mongo(collection, *args):
    """
    param *args : list arguments
    collection   : 数据库连接对象
    args[0] -> text_len     : 数据库中每条记录长度
    args[1] -> fetch_num    : 数据库读取记录个数
    collection, text_len = 3, fetch_num = 100000
    """

    result_temp = collection.find(
        {}, {"V_id": 1, "uid": 1, "text": 1, "_id": 0})

    v_data = list()
    for index, item in enumerate(result_temp):

        if len(item["text"]) > args[0] and item:

            id_temp = item["V_id"]
            uid_temp = item["uid"]

            text_temp = data_preprocessing.extract_chinese(
                item["text"])

            v_data.append([id_temp, uid_temp, text_temp])

        if index % int(args[1] / 10.0) == 0:
            print("已经读取{}条数据".format(index))
        if index >= args[1]:
            return v_data


def read_mongo_base_keyword(collection, *args):
    """
    param *args : list arguments
    collection   : 数据库连接对象
    args[0] -> text_len     : 数据库中每条记录长度
    args[1] -> keyword      : 关键词搜索
    collection, text_len = 3, keyword = "酸奶"
    """

    result_temp = collection.find({'text': {'$regex': args[1]}})

    v_data = dict()
    for index, item in enumerate(result_temp):

        if len(item["text"]) > args[0] and item:

            id_temp = item["V_id"]
            uid_temp = item["uid"]

            text_temp = data_preprocessing.extract_chinese(item["text"])
            v_data[id_temp, uid_temp] = text_temp

        if index % 10 == 0:
            print("已经读取{}条数据".format(index))
    return v_data
