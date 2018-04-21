# -*- coding:utf8 -*-
from pymongo import *
import time
import pymysql
import json
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import chardet


class EgModuleBaseStat:

    def __init__(self):
        self.ip = "10.180.0.30"
        self.prot = 27017
        self.client =MongoClient(self.ip, self.prot)

    def proc(self, requests):
        requests = json.loads(requests)
        brandlist = requests['parameter_list'].split(" ")
        requests['brand_name'] = brandlist[0]
        requests['title_name'] = brandlist[1]
        project_id = requests['project_id']
        self.commodity(requests, db="Tmall", project_id=project_id, source_id=1)
        self.media(requests, "Weibo", project_id=project_id, source_id=2)
        self.commodity(requests, db="JD", project_id=project_id, source_id=3)
        self.media(requests, "JDKB", project_id=project_id, source_id=4)
        self.keysword(requests)

    def commodity(self, requests, db='', project_id=0, source_id=0):
        """
        电商类统计
        :param requests: 用户传来的信息请求
        :param db: Mongodb库名
        :param collection: 表的名称
        :param subtabulation: 子表名称
        :return:
        """
        uniq = {}
        collection = "{}_detail".format(db)
        subtabulation = "{}_comment".format(db)
        db = self.client[db]
        collection = db[collection]
        subtabulation = db[subtabulation]
        for count in collection.find({"brandName": {"$regex": requests['brand_name']},
                                      "title": {"$regex": requests['title_name']}
                                      },
                                     {"goodID": 1, "_id": 0}):
            for i in subtabulation.find({"goodID": count['goodID'],
                                         "comment_date": {
                                            "$gt": requests['project_start_time'],
                                            "$lte": int(time.time())}},
                                        {"comment_date": 1, "_id": 0}):
                timeArray = time.localtime(i['comment_date'])
                otherStyleTime = time.strftime("%Y-%m-%d", timeArray)
                if uniq.get(otherStyleTime):
                    uniq[otherStyleTime] += 1
                else:
                    uniq[otherStyleTime] = 1

        number = uniq.keys()
        number_ = []
        for i in number:
            project = {}
            project['timestamp'] = i
            project['project_id'] = int(project_id)
            project['source_id'] = source_id
            project['count'] = uniq[i]
            number_.append(project)
            self.Mysql(project)
            break

    def media(self, requests, db, project_id=0, source_id=0):
        """
        社交媒体表中数据查询
        :param requests:
        :param db:
        :return:
        """
        contact = {}
        collection = "{}_detail".format(db)
        db = self.client[db]
        collection = db[collection]
        content = collection.find({"text": {"$regex": requests['brand_name'] + requests['title_name']},
                                   "created_date": {"$gt": requests['project_start_time'],
                                                  "$lte": int(time.time())}},)
        for i in content:
            timeArray = time.localtime(i['created_date'])
            otherStyleTime = time.strftime("%Y-%m-%d", timeArray)
            if contact.get(otherStyleTime):
                contact[otherStyleTime] += 1
            else:
                contact[otherStyleTime] = 1

        number = contact.keys()
        number_ = []
        for i in number:
            project = {}
            project['timestamp'] = i
            project['project_id'] = int(project_id)
            project['source_id'] = source_id
            project['count'] = contact[i]
            number_.append(project)
            self.Mysql(project)

    def Mysql(self, project, word=False, words=''):
        conn = pymysql.connect(host='10.180.0.24',
                               port=3307,
                               user='appai',
                               passwd='xee9Jeis5roa',
                               db='social_listening',
                               charset='utf8')
        cursor = conn.cursor()

        if word:
            sql = "INSERT INTO tags_tbl( project_id, tags) VALUES ('%d','%s')" % (int(project['project_id']), json.dumps(words, ensure_ascii=False))
            select_sql = "SELECT * FROM tags_tbl WHERE  project_id = '%d'" % (int(project['project_id']))
            cursor.execute(select_sql)
            content = cursor.fetchone()
            if content:
                UpdateSql = "UPDATE tags_tbl SET tags = '%s' WHERE project_id = '%d'" % (json.dumps(words, ensure_ascii=False), int(project['project_id']))
                cursor.execute(UpdateSql)
                conn.commit()
                conn.close()
                print "情感关键词更新成功"
                return
            else:
                # 执行sql语句
                cursor.execute(sql)
                # 提交到数据库执行
                conn.commit()
                conn.close()
                print "情感关键词插入成功"
                # except:
                #     # 发生错误时回滚
                #     conn.rollback()
                return

        sql = "SELECT * FROM sound_tbl \
               WHERE  sound__timestamp = '%s' AND project_id = '%d' AND sound_source_id = '%d'" % (project['timestamp'], project['project_id'], project['source_id'])

        cursor.execute(sql)
        number = cursor.fetchone()
        if number:
            sound_id = number[0]
            UpdateSql = "UPDATE sound_tbl SET sound_count = %d WHERE sound_id = '%d'" % (project['count'], sound_id)
            cursor.execute(UpdateSql)
            conn.commit()
            conn.close()
            print "声量统计更新成功 平台id为 {}".format(project['source_id'])
        else:
            sql = "INSERT INTO sound_tbl(project_id, \
                   sound_source_id, sound__timestamp, sound_count,sound_attitude) \
                   VALUES ('%d', '%d', '%s', '%d' , %d)" % \
                  (int(project['project_id']), int(project['source_id']), project['timestamp'], int(project['count']), 2)
            try:
                # 执行sql语句
                cursor.execute(sql)
                # 提交到数据库执行
                conn.commit()
                print "声量统计插入成功 平台id{}".format(project['source_id'])
                conn.close()
            except:
                # 发生错误时回滚
                conn.rollback()
        return ''

    def keysword(self, requests):
        words = {}
        key_words = {}
        keys = []
        db_JD = self.client.JD
        db_Tmall = self.client.Tmall
        collection_JD = db_JD.JD_detail
        collection_Tmall = db_Tmall.Tmall_detail
        for JD in collection_JD.find({"brandName": {"$regex": requests['brand_name']},
                                      "title": {"$regex": requests['title_name']}},
                                     {"keyword": 1, "_id": 0}):
            if JD['keyword'].keys():
                keys.append(JD)

        for Tmall in collection_Tmall.find({"brandName": {"$regex": requests['brand_name']},
                                            "title": {"$regex": requests['title_name']}},
                                     {"keyword": 1, "_id": 0}):
            try:
                if Tmall['keyword'].keys():
                    keys.append(Tmall)
            except:
                continue

        for content in keys:
            for number in content['keyword'].keys():
                if key_words.get(number):
                    key_words[number.encode("utf-8")] += content['keyword'][number]
                else:
                    key_words[number.encode("utf-8")] = content['keyword'][number]

        top_twenty = sorted(key_words.items(), key=lambda item: item[1], reverse=True)[0:20]
        for i in top_twenty:
            words[i[0]] = i[1]
        self.Mysql(requests, word=True, words=words)


if __name__ == "__main__":
    # for i in range(100):
    a = time.time()
    print "程序开始{}".format(a)
    number = EgModuleBaseStat()
    new = number.proc(json.dumps({"project_id": "39", "project_start_time": 1519833600, "parameter_list": "华为 P8",}))
    print new
    print "程序结束{}".format(time.time() - a)
    # print "循环了第{}次".format(str(i))
