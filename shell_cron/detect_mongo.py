# -*- coding: UTF-8 -*-
# ********************************************************
# * Author        : lyz
# * Email         : lyz038015@163.com
# * Create time   : 2018-03-04 14:04
# * Last modified : 2018-03-04 16:20
# * Filename      : detect_mongo.py
# * Description   : 
# *********************************************************
import sys, os
import pymysql
from pymongo import MongoClient
import pandas as pd

os.system("/bin/echo 启动mysql数据监测")
os.system("/bin/echo $(date +%F%n%T)")


def connect_mongo(db_name, table_name, text_len = 0, text_num = 100000):
    """
    param db_name : 数据库名称
    param table_name : 表名
    param text_len : 每条文本的长度
    param text_num : 文本数量
    return 
    """
    # 链接mongodb信息
    client = MongoClient("192.168.3.104", 27017)
    # 绑定数据库
    db_temp = client[db_name]
    # 链接数据库的表
    collection_mongo = db_temp[table_name]
    #
    return collection_mongo

def fetch_mongo(start_time, end_time, key_field, value_field, collection_mongo):
    """
    param start_time : 数据读取时间-01
    param end_time : 数据读取时间-02
    param key_temp : 表字段-01
    param value_temp : 表字段-02
    param db_data : 返回的mongo数据列表对象
    return 
    """
    # 查询mongo数据库
    db_data = collection_mongo.find({"crawl_date":{'$gte':start_time,'$lte':end_time}})
    # db_data = collection_mongo.find()
    # 
    data_dict = dict()
    for item in db_data:
        # key_field = "V_id"
        # value_field = "text"
        key_temp = item[key_field]
        # key_temp = item["V_id"]
        value_temp = item[value_field]
        
        data_dict[key_temp] = value_temp

    return data_dict


def connect_mysql(user,user_passwd, db_name):
    """
    param user : 用户名
    param user_passwd : 密码
    param db_name : 数据库名
    param training_threshold : 数据阈值，用于决定是否训练LDA模型
    return : Mysql数据库的检测
    """
    print(u"logging information for mysql")
    # 打开数据库连接
    db_conn = pymysql.connect(
            host = "192.168.3.207",
            database = db_name,
            user = user,
            password = user_passwd,
            port = 3306,
            charset = 'utf8'
    )
    return db_conn


def search_mysql(db_conn, training_threshold):
    # db_conn : mysql数据库的链接对象
    """
    log_id：日志id：默认自增
    data_table_path：爬虫爬取的数据在MongoDB中存储的库和表的路径
    start_time：日志对应的第一条数据的存入时间
    end_time：日志对应的最后一条数据的存入时间
    data_count：日志对应的数据更新（插入和更新）条数
    data_source：日志对应的数据的数据来源（如微博、ＪＤ快报等）
    data_status：日志对应的数据是否被处理（0=未处理；1=已处理）
    """
    # sql查询语句
    # UNIX_TIMESTAMP(crawler_log_tbl.start_time)获取start_time的时间戳
    # crawler_log_tbl是表名
    sql_cmd = "select UNIX_TIMESTAMP(crawler_log_tbl.start_time), UNIX_TIMESTAMP(crawler_log_tbl.end_time), data_count, data_status, data_table_path from crawler_log_tbl"

    #利用pandas 模块导入mysql数据
    df_log = pd.read_sql(sql_cmd, db_conn)
    # df_log是pandas中的DataFrame类型，其中.values()返回该类型的np.ndarray(),
    value_temp = df_log.values[0].tolist()
    # 0->start_time, 1->end_time, 2->data_count, 3->data_status, 4->data_table_path
    # data_count达到特定阈值，开启LDA模型的训练
    if value_temp [2] >= training_threshold and value_temp[3] == 0:
        print(u"Start to train LDA")
        return value_temp
    else:
        for item in value_temp:
            print(item)
    print("\n")
    # 关闭数据库连接
    db_conn.close()

def update_mysql(db_conn):
    """
    param db_conn : mysql数据库的链接对象
    return None
    """
    # 使用cursor()方法获取操作游标
    cur = db_conn.cursor()

    sql_update ="update crawler_log_tbl set data_status = '%d'"

    try:
        # 向sql语句传递参数
        cur.execute(sql_update % (1))  
        #提交
        db_conn.commit()
    except Exception as e:
        #错误回滚
        db_conn.rollback()
    finally:
        db_conn.close()
        

def run():
    """
    mysql数据库的操作
    """
    # 用户名
    user = "root"
    # 用户密码
    user_passwd = "112233"
    # 数据库名
    db_name = "engine"
    # 训练阈值
    train_thd = 100
    # 链接mysql
    db_conn = connect_mysql(user, user_passwd, db_name)
    sql_det_res = search_mysql(db_conn, train_thd)
    # 
    if sql_det_res:
        """
        mongo数据库的操作
        """
        info_for_mongo = sql_det_res[-1].strip().split(".")
        # 开始时间
        st_temp = sql_det_res[0]
        # 结束时间
        et_temp = sql_det_res[1]
        # 数据库名
        db_name_temp = info_for_mongo[0]
        # 数据库中表名
        table_name_temp = info_for_mongo[1]
        # 日志状态，是否已经处理
        status_temp = sql_det_res[3]
        # 链接mongo
        db_mongo = connect_mongo(db_name_temp, table_name_temp)
        # 表中字段
        col_field_01 = "V_id"
        col_field_02 = "text"
        # 从mongo中提取数据
        data_mongo = fetch_mongo(st_temp, et_temp, col_field_01, col_field_02, db_mongo)
        # print(list(data_mongo.items())[:3])
        # 
        update_mysql(db_conn)

    else:
        pass


if __name__=="__main__":

    run()


