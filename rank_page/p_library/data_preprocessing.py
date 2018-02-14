#!/opt/Anaconda3/bin/python
# -*- coding: UTF-8 -*-

# ********************************************************
# * Author        : lyz
# * Email         : lyz038015@163.com
# * Create time   : 2018-01-29 11:41
# * Last modified : 2018-02-14 13:53
# * Filename      : data_preprocessing.py
# * Description   : 
# *********************************************************
import os
import re
import codecs
import time
import pickle
from pymongo import MongoClient


def log_info(msg):
    """
    param msg : 文字消息
    function  : 输出带有时间的文字消息
    """
    print(r"%s: %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg))

def extract_chinese(str_temp):
    """
    param str_temp : 字符串对象
    return         : 中文文本
    """
    line = str_temp.strip()
    #
    re_temp = re.compile(r'[^\u4e00-\u9fa5]')
    #
    zh = " ".join(re_temp.split(line)).strip()
    #
    zh = ",".join(zh.split())
    Outstr = zh
    # 保留中文文本
    return Outstr


class read_data_from_file(object):
    """
    functions   : __iter__读取文件内容
    """
    def __init__(self, dirname):
        """
        param dirname       : 文件路径名
        """
        self.dirname = dirname

    def __iter__(self):
        """
        param None
        """
        with codecs.open(self.dirname, 'rb', encoding="utf-8") as reader:
            lines = reader.readlines()
            for line in lines:
                yield line.strip()


def store_data_to_pkl(pkl_path, data_object):
    """
    param pkl_path    : pkl文件存储路径
    param data_object : 数据存储对象
    function          : 数据持久化
    """
    #
    pkl_f = open(pkl_path, 'wb')
    pickle.dump(data_object, pkl_f)
    pkl_f.close()
    #
    file_size_temp = os.path.getsize(pkl_path)
    print("文件存档完成，文件大小为：{0:.3f}M".format(file_size_temp/(1024.0 * 1024.0)))


def get_data_from_pkl(pkl_path):
    """
    param pkl_path  : pkl文件存储路径
    param data_temp : dict对象
    function        : 读取本地pkl数据
    """
    data_temp = dict()
    with open(pkl_path, 'rb') as pkl_reader:
        data_temp = pickle.load(pkl_reader)
    return data_temp


def get_data_from_mongo(test_len = 0):
    """

    """
    # 链接mongodb信息
    client = MongoClient("192.168.3.104", 27017)
    # 绑定数据库
    db_temp = client.Weibo
    # 链接数据库的表
    collection = db_temp.V_detail
    #
    result_temp = collection.find()
    #
    v_data = dict()
    for item in result_temp:
        if len(item["text"]) > text_len:
            v_data[item["V_id"]] = item["text"]
    return v_data



