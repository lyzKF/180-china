# _*_ coding:utf-8 -*-
import sys
import os
import re
import pymysql
import time
import pickle
from pymongo import MongoClient


def log_info(msg):
    """
    func : show messages with time information
    """
    print(r"%s: %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg))


def extract_chinese(str_temp):
    """
    func : fetch chinese text
    """
    line = str_temp.strip()
    #
    re_temp = re.compile(r'[^\u4e00-\u9fa5]')
    #
    zh = " ".join(re_temp.split(line)).strip()
    #
    zh = ",".join(zh.split())
    Outstr = zh

    return Outstr


class read_data_from_file(object):
    """
    func : read file
    """

    def __init__(self, dirname):
        """
        """
        self.dirname = dirname

    def __iter__(self):
        """
        """
        with open(self.dirname, 'r') as reader:
            lines = reader.readlines()
            for line in lines:
                yield line.strip()


def connect_mongo(host, db_name, table_name, port):
    """
    func : connect to mongo 
    """
    client = MongoClient(host, port, connect=False)

    db_temp = client[db_name]

    collection_mongo = db_temp[table_name]

    return collection_mongo


def connect_mysql(host, user, user_passwd, db_name, port):
    """
    func : connect to mysql
    """
    db_conn = pymysql.connect(
        host=host,
        database=db_name,
        user=user,
        password=user_passwd,
        port=port,
        charset='utf8'
    )
    return db_conn
