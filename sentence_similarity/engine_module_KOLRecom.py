# _*_ coding:utf-8 -*-
import os
import sys
import json
import numpy as np

from gensim.models import Word2Vec

import pymysql
from pymongo import *

import networkx as nx

raw_monogo_ip='10.180.0.30'
raw_monogo_port=27017

ripe_monogo_ip='10.180.0.24'
ripe_monogo_port=27017


def connect_mysql(user,user_passwd, db_name):
    """
    param user : 用户名
    param user_passwd : 密码
    param db_name : 数据库名
    return : 
    """
    # 打开数据库连接
    db_conn = pymysql.connect(
            host = "10.180.0.24",
            database = db_name,
            user = user,
            password = user_passwd,
            port = 3307,
            charset = 'utf8'
            )
    if db_conn:
	print('APP 数据库连接成功')
    return db_conn

def insert_mysql(db_conn, top_rank_list, project_id, network_id ):
    """
    param db_conn : mysql数据库的链接对象
    param insert_object : 插入数据对象
    return None
    """
    try:
        with db_conn.cursor() as cursor:
            sql = "SELECT rank_tbl.rank_id FROM rank_tbl, vip_tbl WHERE rank_tbl.vip_id = vip_tbl.vip_id AND rank_tbl.project_id = " + project_id + " AND vip_tbl.vip_network_id = " + network_id
            print(sql)
	    cursor.execute(sql)
            results = cursor.fetchall()
            res_list = [ i[0] for i in results ]
            count = -1
	    if len(results) == 0:
                print("INSERT")
                for line in top_rank_list:
		    print(line)
		    if line["vip_doc_url"] == None:
                    	sql = "INSERT INTO rank_tbl (project_id, vip_id, rank_score ) VALUES ("+ project_id+","+ line["vip_id"]+","+str(line["rank_score"])+")"
                    else:
			sql = "INSERT INTO rank_tbl (project_id, vip_id, rank_score, vip_doc_url ) VALUES ("+ project_id+","+ line["vip_id"]+","+str(line["rank_score"])+",'"+line["vip_doc_url"]+"')"
                    print(sql)
                    cursor.execute(sql)
                # db_conn is not autocommit by default. So you must commit to save
                db_conn.commit()
            else:
                print("UPDATE")
                count = 0
                for line in top_rank_list:
		    if line["vip_doc_url"] == None:
                    	sql = "UPDATE rank_tbl SET vip_id = "+ line["vip_id"]+", rank_score =  "+str(line["rank_score"])+", vip_doc_url ='' WHERE rank_id = " + str(res_list[count])
                    else:
                    	sql = "UPDATE rank_tbl SET vip_id = "+ line["vip_id"]+", rank_score =  "+str(line["rank_score"])+", vip_doc_url ='"+line["vip_doc_url"]+"' WHERE rank_id = " + str(res_list[count])
		    cursor.execute(sql)
                    count = count + 1
                # db_conn is not autocommit by default. So you must commit to save
                db_conn.commit()
    except Exception as e:
        print(e)
	print (count)
	print (res_list)
        print(sql)
        db_conn.rollback()
        print("Fialed")
    finally:
        db_conn.close()

def readWeight(project_id):
    client = MongoClient(ripe_monogo_ip,int(ripe_monogo_port))
    db = client.Ripe
    collection = db.sim_jd_wb
    dict = {}
    edge_weights = {}
    for line in collection.find({"project_id":project_id,"source_id":"1"}):
#        print (line)
        edge = int(line["doc_id"])
        weight = float(line["similarity"])
        dict[edge] = weight
    minW = min(dict.items(),key=lambda x : x[1])[1] 
    maxW = max(dict.items(),key=lambda x : x[1])[1]
#    print (minW)
#    print (maxW)
    a = maxW - minW
    for key in dict:
        w = dict.get(key)
        dict[key] = (w - minW)/a
#    print ( dict)
    return dict

def createGraph(weight_dict):
    G = nx.DiGraph()
    client = MongoClient(raw_monogo_ip,int(raw_monogo_port))
    db = client.Weibo
    collection = db.Weibo_detail
    print("连接MongoDB微博表成功")
    for line in collection.find():
        transmiters = line['transmit_max'].keys()
        uid = line['uid']
        if transmiters :
            for vid in transmiters:        
                weight = weight_dict.get(int(line['V_id']),0.0)
#                print (str(line['V_id'])+":"+str(weight))
                if weight > 0.86:
                    if G.has_edge(vid,uid):
#                       print (strlist[w_index])
                        w = G.get_edge_data(vid,uid).get('weight')
                #print(weight)
                #print(w)
                        weight = weight + w
#                        print("aaaaaa")
#                print(strlist[0]+"\t"+strlist[1]+"\t"+strlist[w_index]+":"+str(w)+","+str(weight))
                    G.add_weighted_edges_from([(vid, uid, weight)])
    print("读入"+str(G.number_of_edges())+"条边")
    return G

def lookupDoc(uid):
    if uid :
    	client = MongoClient(ripe_monogo_ip, int(ripe_monogo_port))
    	db = client.Ripe
    	collection = db.sim_jd_wb
    	for line in collection.find({"uid":str(uid)}).sort([("similarity",-1)]).limit(1):
        	print("lookup"+uid)
        	print (line)
		if line :
			return str(line["doc_id"])
		else:
			return ''
    else:
	return ''

def readSimTop(project_id, users):
    client = MongoClient(ripe_monogo_ip, int(ripe_monogo_port))
    db = client.Ripe
    collection = db.sim_jd_wb
    dict = {}
    edge_weights = {}
    match = {'source_id':'2', 'project_id': project_id}
    groupby = 'uid'
    group = {'_id':"$%s" % (groupby if groupby else None),'sim':{'$max':"$similarity"}}
    for line in collection.aggregate([{'$match': match},{'$group' : group}]):
	#print ("Line"+str(line))
	edge = str(line["_id"])
       	weight = float(line["sim"])
	dict[edge] = float(weight)
#    for uid in users:
#	print(uid)
#        weight = 0.0;
#        count = 0
#        for line in collection.find({"uid":str(uid),"project_id":project_id,"source_id":"2"}).sort([("similarity",-1)]).limit(5):
#            #print ("Line"+str(line))
#            count = count + 1
#            edge = str(line["uid"])
#            weight = weight + float(line["similarity"])
#        if count > 0:
#            dict[edge] = float(weight/count)
#            print(edge)
#            print(weight)
    
    if dict == {} :
	return dict

    minW = min(dict.items(),key=lambda x : x[1])[1] 
    maxW = max(dict.items(),key=lambda x : x[1])[1]
    print ("read sim top min " + str(minW))
    print ("read sim top max " + str(maxW))
    a = maxW - minW
    for key in dict:
        w = dict.get(key)
        dict[key] = (w - minW)/a
#    print ( dict)
    return dict

def readFansCount():
    client = MongoClient(raw_monogo_ip, int(raw_monogo_port))
    db = client.JDKB1
    collection = db.jdkb_user
    dict = {}
    edge_weights = {}
    for line in collection.find():
#        print (line)
        edge = str(line["userId"])
        weight = int(line["fans"])
        dict[edge] = weight
    minW = min(dict.items(),key=lambda x : x[1])[1] 
    maxW = max(dict.items(),key=lambda x : x[1])[1]
    print ("read Fans Count " + str(minW))
    print ("read Fans Count " + str(maxW))
    a = maxW - minW
    for key in dict:
        w = dict.get(key)
        dict[key] = (w - minW)/a
#    print ( dict)
    return dict

def lookupDocJD(uid,project_id):
    client = MongoClient(ripe_monogo_ip, int(ripe_monogo_port))
    db = client.Ripe
    collection = db.sim_jd_wb
    doc_id =''
    for line in collection.find({"uid": str(uid),"project_id":project_id,"source_id":"2"}).sort([("similarity",-1)]).limit(1):
	doc_id = line["doc_id"]
    if doc_id != '':
	client = MongoClient(raw_monogo_ip, int(raw_monogo_port))
    	db = client.JDKB1
    	collection = db.JDKB_detail
    	for line in collection.find({"article_id": str(doc_id)}):
#       print("lookup"+uid)
#       print (line)
            return line["article_link"]
    else:
	return doc_id

class EgModuleKOLRe:

    def __init__(self):
        self.log = {}


    def proc(self,request):
      
        params = json.loads(request)

        project_id = str(params['project_id'])
        print("KOL starting "+ project_id)

        dict = readWeight(project_id)
        G = createGraph( dict )
        
        pr = nx.pagerank(G,alpha=0.8)
        s = [ (k,pr[k]) for k in sorted (pr,key=pr.get, reverse=True) ]
    
        top_rank_list = list()
        for line in s[0:10]:
	    print(line)
            dict = {}
	    tmp_str=lookupDoc(list(line)[0])
	    print(tmp_str)
	    if tmp_str:
            	doc_url = "https://m.weibo.cn/status/"+tmp_str
	    else:
		doc_url = ""
            dict["vip_id"] = list(line)[0]
            dict["rank_score"] = float(list(line)[1])
            dict["vip_doc_url"] = doc_url
            top_rank_list.append(dict)
    
        user = 'appai'
        user_passwd = 'xee9Jeis5roa'
        db_name = 'social_listening'
    
        db_conn = connect_mysql(user, user_passwd, db_name)
        insert_mysql(db_conn, top_rank_list, project_id, "1")
        
	print("KOL for weibo Finished.")
	
        vipFansCount = readFansCount()
	
        simDict = readSimTop(project_id, vipFansCount) 
        scoreDict = {}
        for uid in simDict:
            if float(simDict[uid]) > 0.86:
		if vipFansCount.get(uid):
                	scoreDict[uid] = 0.7*simDict[uid]+0.3*vipFansCount[uid]
		else:
                	scoreDict[uid] = 0.7*simDict[uid]
            else:
		if vipFansCount.get(uid):
                	scoreDict[uid] = 0.3*vipFansCount[uid]
		else:
                	scoreDict[uid] = 0.0

        pr = scoreDict
        s = [ (k,pr[k]) for k in sorted (pr,key=pr.get, reverse=True) ]
        
        top_rank_list = list()
        for line in s[0:10]:
            dict = {}
            doc_url = lookupDocJD(list(line)[0], project_id)
            dict["vip_id"] = list(line)[0]
            dict["rank_score"] = float(list(line)[1])
            dict["vip_doc_url"] = doc_url
            top_rank_list.append(dict)
#    	print(dict)
        db_conn = connect_mysql(user, user_passwd, db_name)
        insert_mysql(db_conn, top_rank_list, project_id, "2")
       
	print("KOL for JDKB  Finished.")
        return "xxxx"
if __name__ == '__main__':
    a = EgModuleKOLRe()
    a.proc(json.dumps({"project_id": "33"}))

