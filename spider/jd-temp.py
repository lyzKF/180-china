# Created by lyz on 2017/12/18
# Email: lyz038015@163.com
# GitHub: https://github.com/lyzKF
# Functions: 京东产品信息爬虫程序
# nohup python get-info-JD.py > log.txt 2>&1 &
###############################################

import requests
import csv
from multiprocessing import Pool
import multiprocessing
import time
import psutil
import sys

class JD():

    def __init__(self, url_file, csv_file):
        """
        param url_file: 商品url链接存放文件
        param csv_file: 商品评论等信息存放文件
        """
        self.url_file = url_file
        self.csv_file = csv_file
        print("CPU数量:{0}\n运行状态:{1}".format(multiprocessing.cpu_count(), psutil.cpu_stats()))
        print("内存信息:{0}".format(psutil.virtual_memory()))
        self.log_info("程序开始运行")

    def key_exit(self, item, str_temp):
        # 判断关键字是否存在
        if str_temp in item.keys():
            str_temp_value = item[str_temp]
            str_temp_value = str(str_temp_value)
        else:
            str_temp_value = u''
        return str_temp_value

    # 写入产品信息
    def comments_infor(self, productid, product_price):
        """
        param productid: 京东商品唯一标识码
        return: 返回商品的productCommentSummary和hotCommentTagStatistics信息
        """
        print("ProductID:{0}".format(productid))
        # 创建CSV文件，并写入字段名
        csv_object = open(self.csv_file, "a", encoding='utf-8', newline='')
        product_text = open('../mengniu/data2/' + str(productid) + '.txt', 'w')
        product_text.write(product_price+'\n')
        writer = csv.writer(csv_object)
        tag = True
        # 爬取JD接口，获取产品相关信息
        while tag:
            # 创建一个session
            with requests.session() as s:
                # 评论API接口url
                url = 'https://club.jd.com/comment/productPageComments.action'
                # 表单数据，其中productId与page很关键
                data = {
                    'productId': productid,
                    'score': 0,
                    'sortType': 5,
                    'pageSize': 10,
                    'isShadowSku': 0,
                    'page': 0
                }
                requests.adapters.DEFAULT_RETRIES = 5
                s.keep_live = False
                try:
                    r = s.get(url, params=data).json()
                    if r is None:
                        print("comments not exit!!!")
                        return 0
                    # 写入产品评价等信息
                    comment_temp = r['comments']
                    for item in comment_temp:
                        userexpvalue_temp = self.key_exit(item=item, str_temp='userExpValue')
                        id_temp = self.key_exit(item=item, str_temp='id')
                        nickname_temp = self.key_exit(item, 'nickname')
                        topped_temp = self.key_exit(item, 'topped')
                        creationTime_temp = self.key_exit(item, 'creationTime')
                        isTop_temp = self.key_exit(item, 'isTop')
                        replyCount_temp = self.key_exit(item, 'replyCount')
                        score_temp = self.key_exit(item, 'score')
                        usefulVoteCount_temp = self.key_exit(item, 'usefulVoteCount')
                        uselessVoteCount_temp = self.key_exit(item, 'uselessVoteCount')
                        userLevelId_temp = self.key_exit(item, 'userLevelId')
                        viewCount_temp = self.key_exit(item, 'viewCount')
                        isReplyGrade_temp = self.key_exit(item, 'isReplyGrade')
                        userClient_temp = self.key_exit(item, 'userClient')
                        userLevelName_temp = self.key_exit(item, 'userLevelName')
                        plusAvailable_temp = self.key_exit(item, 'plusAvailable')
                        recommend_temp = self.key_exit(item, 'recommend')
                        content_temp = self.key_exit(item, 'content')

                        writer.writerow([productid, id_temp, nickname_temp, topped_temp, creationTime_temp, isTop_temp,
                                         replyCount_temp, score_temp, usefulVoteCount_temp, uselessVoteCount_temp,
                                         userLevelId_temp, viewCount_temp, isReplyGrade_temp, userClient_temp, userLevelName_temp,
                                         plusAvailable_temp, userexpvalue_temp, recommend_temp, content_temp])
                except Exception as e:
                    # print("id:{0}\aerror:{1}".format(productid, e))
                    data['page'] = 0
                    tag = False
                    pass
                data['page'] += 1
        # 对于每个产品，将产品部分信息写入以其productId命名的文本文档中
        if r['productCommentSummary']:
            for key in r['productCommentSummary'].keys():
                content_temp = str(key) + ':' + str(r['productCommentSummary'][key])
                product_text.write(content_temp + '\n')
        if r['hotCommentTagStatistics']:
            for item in r['hotCommentTagStatistics']:
                item_temp = str(item['name']) + ':' + str(item['count'])
                product_text.write(item_temp + '\n')
        # 文件系统关闭       
        product_text.close()
        csv_object.close()

    # 从url_file文件中读取产品的url信息、价格信息，然后写入列表
    def url_list(self):
        '''
        return: 返回一个列表，列表对象是商品url与商品价格的二元组
        '''
        self.log_info("读取产品信息文件，将产品URL与产品价格写入列表")
        urls = list()
        with open(self.url_file, 'r',encoding='utf-8') as reader:
            lines = csv.reader(reader)
            for line in lines:
                if len(line) > 0:
                    url_temp = line[0].replace('(', '').replace(')', '').replace('\'', '').replace('\'', '').replace(',', '')
                    price_temp = str(line[1])
                    urls.append([url_temp, price_temp])
        return urls
    
    # 程序主要入口
    def write_info(self, product_info):
        """
        param product_info: 由产品id与产品价格构成
        return: None
        function: 每个产品信息写入以productId命名的文本文档，产品评价信息写入"product_information.csv"
        """
        # 去除特殊字符，获取产品的URL
        url_temp = product_info[0].replace('(', '').replace(')', '').replace('\'', '').replace('\'', '').replace(',', '')
        # 获取产品价格信息
        price_temp = 'price:' + str(product_info[1])
        if url_temp.startswith('//'):
            # 打印产品的URL字符串
            # 获取产品ID
            productid_temp = url_temp.replace('//item.jd.com/', ' ').replace('.html', ' ')
            # 调用方法comments_info
            self.comments_infor(productid=int(productid_temp), product_price = price_temp)
            # time.sleep(1)

    # 自定义log函数，主要是加上时间
    def log_info(self, msg):
        """
        param msg: 输出消息
        return: None
        """
        print(u'%s: %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), msg))

    # 多进程
    def run(self, ids):
        """
        param ids: 商品url与商品价格的二元组
        return: None
        """
        startTime = time.time()
        self.log_info("开启多进程爬虫程序")
        cpu_count = multiprocessing.cpu_count()
        pool = Pool(cpu_count)
        pool.map(self.write_info, ids)
        pool.close()
        pool.join()
        self.log_info("多进程任务结束")
        print("Running Time:{0}".format(time.time() - startTime))
        self.log_info("爬虫程序终止")

if __name__ == "__main__":
    csv_file = '../mengniu/data2/product_information_mengniu.csv'
    url_file = '../mengniu/url/mengniu_url.csv'
    jd = JD(url_file=url_file, csv_file=csv_file)
    urls = jd.url_list()
    # 是否多进程爬取数据
    multiprocess = sys.argv[1]
    if multiprocess == 0:
        for item in urls:
            jd.write_info(product_info = item)
    else:
        jd.run(ids = urls)
