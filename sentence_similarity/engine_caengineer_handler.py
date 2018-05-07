# _*_ coding:utf-8 -*-

import sys
import time
reload(sys)
sys.setdefaultencoding("utf-8")
sys.path.append('gen-py')
from enginee import CAEnginee
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from thrift.server import TNonblockingServer
import engine_module_wordscloud
import engine_module_KOLRecom
import engine_module_baseStat
from concurrent.futures import ThreadPoolExecutor
import threading
import MySQLdb
import json
from datetime import datetime
import logging
import time
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='my.log', level=logging.DEBUG, format=LOG_FORMAT)
engine_mysql_ip = '10.180.0.23'
engine_mysql_port = 3307
engine_mysql_db = 'engine'
engine_mysql_user = 'appai'
engine_mysql_pwd = 'xee9Jeis5roa'

import signal


class CAEngineeHandler:
    def __init__(self):
        self.executor = ThreadPoolExecutor(100)
        self.log = {}
        self.wordsClouder = engine_module_wordscloud.EgModuleWordCloud()
        self.kolRecomer = engine_module_KOLRecom.EgModuleKOLRe()
        self.baseStater = engine_module_baseStat.EgModuleBaseStat()
        self.taskListLock = threading.Lock()
        self.managerThread = None
        self.serverThread = None
        self.enable = True
        self.server = None
        self.serverTP = None
        self.tfactory = TTransport.TBufferedTransportFactory()
        self.pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    def ping(self):
        return 1

    def serverStop(self):
        self.enable = False
        time.sleep(1)
        """
        if self.server !=None:
            # self.serverTP.close()
            self.server.stop()
            print 'stop server'
            self.server=None
        """

    def serverStart(self, port1):
        self.managerThread = threading.Thread(target=self.managerProc)
        self.managerThread.setDaemon(True)
        self.managerThread.start()

        processor = CAEnginee.Processor(self)
        self.serverTP = TSocket.TServerSocket(port=port1)
        self.server = TServer.TThreadedServer(
            processor, self.serverTP, self.tfactory, self.pfactory)

    def managerProc(self):
        inited = False
        while self.enable:
            currt = datetime.now()
            sql = ""
            if inited:
                sql = 'select * from task_tbl where task_status=0 and \'{}\'>project_start_time and \'{}\'<project_end_time order by task_create_time desc'.format(
                    currt, currt)
            else:
                sql = 'select * from task_tbl where \'{}\'>project_start_time and \'{}\'<project_end_time order by task_create_time desc'.format(
                    currt, currt)
                inited = True

            db = MySQLdb.connect(host=engine_mysql_ip, port=engine_mysql_port,
                                 user=engine_mysql_user, passwd=engine_mysql_pwd, db="engine")
            # db = MySQLdb.connect("localhost", "root", "112233", "engine")
            print sql
            # 使用cursor()方法获取操作游标
            cursor = db.cursor()
            insert_sql = ""
            try:
                # 执行sql语句
                cursor.execute(sql)
                # 提交到数据库执行
                db.commit()
                tasks = cursor.fetchall()
                for task in tasks:
                    task_id = task[0]
                    task_type = int(task[2])
                    task_status = task[4]
                    task_paras = task[3]
                    insert_sql = 'update task_tbl SET task_status= 1 where task_id={}'.format(
                        task_id)
                    cursor.execute(insert_sql)
                    db.commit()
                    # 增加新的人物到工作线程
                    # print "submit task,taskid={},task_type={},task_paras={}".format(task_id, task_type, task_paras)
                    self.executor.submit(
                        self.ThreadProc, task_id, task_type, task_paras)
            except:
                # 发生错误时回滚
                logging.error(
                    "function managerProc(),执行 insert_sql={}".format(insert_sql))
                db.rollback()
            db.close()
            time.sleep(5)
        self.executor.shutdown(True)

    def call(self, type, request):
        res = '0'
        """
        if type==2:
            re_str=self.baseStater.proc(request)
            print re_str
            return  re_str

        """

        # 解析请求，获得参数
        prequest = json.loads(request)
        p_id = prequest['project_id']
        p_start_time = float(prequest['project_start_time'])
        p_end_time = float(prequest['project_end_time'])
        parameter_list = request
        d = datetime.fromtimestamp(p_start_time)
        data1 = d.strftime("%Y-%m-%d %H:%M:%S.%f")
        d = datetime.fromtimestamp(p_end_time)
        data2 = d.strftime("%Y-%m-%d %H:%M:%S.%f")

        checksql = "select * from task_tbl where project_id={} and task_type='{}'".format(
            p_id, type)
        sql = "insert into task_tbl(project_id,task_type,parameter_list,task_status,project_start_time,project_end_time) values({},'{}','{}',0,'{}','{}')".format(
            p_id, type, parameter_list, data1, data2)
        # 任务处理，插入数据库

        # db = MySQLdb.connect("localhost", "root", "112233", "engine")
        db = MySQLdb.connect(host=engine_mysql_ip, port=engine_mysql_port,
                             user=engine_mysql_user, passwd=engine_mysql_pwd, db="engine")
        # 使用cursor()方法获取操作游标
        try:
            cursor = db.cursor()
            cursor.execute(checksql)
            # 提交到数据库执行
            db.commit()
            if cursor.rowcount >= 1:
                res = '1'
            else:

                # 执行sql语句
                cursor.execute(sql)
                # 提交到数据库执行
                db.commit()
                res = '1'
        except:
            # 发生错误时回滚
            db.rollback()
            logging.error("function call(),执行 sql={}".format(sql))
            res = '0'
        db.close()
        return res
    
    def ThreadProc(self, id, type, content):
        if type == 0:
            # print 'ThreadProc--type={} content={}'.format(type, content)
            print 'call words cloud function'
            self.wordsClouder.proc(content)
            print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>call words end !!!!!"
        elif type == 1:
            print 'call KOL index function'
            self.kolRecomer.proc(content)
            print "call KOL end !!!!!"
        elif type == 2:
            print 'call static started'
            self.baseStater.proc(content)
            print 'call static end!!!>>>'
        else:
            print 'unimplemented function'
#
        time.sleep(3)

        # if type== 1:
        db = MySQLdb.connect(host=engine_mysql_ip, port=engine_mysql_port,
                             user=engine_mysql_user, passwd=engine_mysql_pwd, db="engine")

        # db = MySQLdb.connect("localhost", "root", "112233", "engine")
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()
        sql = ""
        try:
            # 执行sql语句
            sql = 'update task_tbl SET task_status=0  where task_id={}'.format(
                id)
            cursor.execute(sql)
            # 提交到数据库执行
            db.commit()
        except:
            # 发生错误时回滚
            db.rollback()
            logging.error("function call(),执行 sql={}".format(sql))
