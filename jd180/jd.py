# Created by lyz on 2017/12/18
# Email: lyz038015@163.com
# GitHub: https://github.com/lyzKF
# Functions: 京东产品信息爬虫程序
###############################################
import requests
import csv
from multiprocessing import Pool
import multiprocessing
import time
import psutil
import sys
import os


# 自定义log函数，主要是加上时间
def log_info(msg):
    """
    param msg: 输出消息
    return: None
    """
    print(u'%s: %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), msg))

class JD():

    def __init__(self, url_file, csv_file, data_path):
        """
        param url_file: 商品url链接存放文件
        param csv_file: 商品评论等信息存放文件
        """
        self.url_file = url_file
        self.csv_file = csv_file
        self.data_path = data_path
        print("CPU数量:{0}\n运行状态:{1}".format(multiprocessing.cpu_count(), psutil.cpu_stats()))
        print("内存信息:{0}".format(psutil.virtual_memory()))
        log_info("程序开始运行")

    def key_exit(self, item, str_temp):
        # 判断关键字是否存在
        if str_temp in item.keys():
            str_temp_value = item[str_temp]
            str_temp_value = str(str_temp_value)
        else:
            str_temp_value = u''
        return str_temp_value

    # 写入评论信息
    def comments_infor(self, productid):
        """
        param productid: 京东商品唯一标识码
        return: 返回商品的productCommentSummary和hotCommentTagStatistics信息
        """
        print('#' * 50)
        # 创建CSV文件，并写入字段名
        csv_object = open(self.csv_file, "a", encoding='utf-8', newline='')
        writer = csv.writer(csv_object)
        # 判断标志
        tag, r_is_exit = True, False
        # 评论API接口url
        url = 'https://club.jd.com/comment/productPageComments.action'
        # 爬取JD接口，获取产品相关信息
        data = {}
        pagenumber = 0
        print("开始页码:{0}".format(pagenumber))
        while tag:
            # 表单数据，其中productId与page很关键
            data['productId'] = productid
            data['score'] = 0
            data['sortType'] = 5
            data['pageSize'] = 10
            data['isShadowSku'] = 0
            data['page'] = pagenumber
            requests.adapters.DEFAULT_RETRIES = 100
            try:
                r = requests.get(url, params=data).json()
                r_is_exit = True
            except Exception as e:
                r_is_exit = False
                print("Request data:{0}".format(data))
                print('Error_productid:{0}\n:{1}'.format(productid, e))
            # 判断r是否存在
            if r_is_exit:
                # 判断评论信息是否存在
                if r['comments']:
                    # add the number of page
                    pagenumber += 1
                    # 写入产品评价等信息
                    for item in r['comments']:
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
                        #
                        writer.writerow([productid, id_temp, nickname_temp, topped_temp, creationTime_temp, isTop_temp,
                                         replyCount_temp, score_temp, usefulVoteCount_temp, uselessVoteCount_temp,
                                         userLevelId_temp, viewCount_temp, isReplyGrade_temp, userClient_temp,
                                         userLevelName_temp,
                                         plusAvailable_temp, userexpvalue_temp, recommend_temp, content_temp])
                else:
                    # 如果不存在，说明已经读取到最后一页
                    # 文件系统关闭
                    csv_object.close()
                    print("最终页码:{0}".format(data['page']))
                    print("商品{0}的评论信息读写完成！！！".format(productid))
                    tag = False
            else:
                # 修改tag，退出循环
                tag = False
                print("r is None")

        # 获取当前文件大小
        file_size_temp = os.path.getsize(self.csv_file)
        print("The size of current file:{0}".format(file_size_temp/1024.0))
        
   # 写入产品信息
    def title_infor(self, productid, product_price):
        """
        param productid: 京东商品唯一标识码
        """
        # 创建txt文件，并写入字段名
        product_text = open(self.data_path + 'data/' + str(productid) + '.txt', 'w')
        product_text.write(product_price+'\n')
        # 评论API接口url
        url = 'https://club.jd.com/comment/productPageComments.action'
        # 创建一个session
        with requests.session() as s:
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
            except Exception as e:
                print(productid)
                pass   
            
    def url_list(self):
        '''
        return: 返回一个列表，列表对象是商品url与商品价格的二元组
        '''
        self.log_info("读取产品信息文件，将产品URL与产品价格写入列表")
        urls = list()
        with open(self.url_file, 'r', encoding='utf-8') as reader:
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
        # price_temp = 'price:' + str(product_info[1])
        if url_temp.startswith('//'):
            # 打印产品的URL字符串
            # 获取产品ID
            productid_temp = url_temp.replace('//item.jd.com/', ' ').replace('.html', ' ')
            # 调用方法comments_info
            self.comments_infor(productid=int(productid_temp))
            #self.title_infor(productid=int(productid_temp), product_price = price_temp)

    # 多进程
    def run(self, ids):
        """
        param ids: 商品url与商品价格的二元组
        return: None
        """
        startTime = time.time()
        log_info("开启多进程爬虫程序")
        cpu_count = multiprocessing.cpu_count()
        pool = Pool(cpu_count)
        pool.map(self.write_info, ids)
        pool.close()
        pool.join()
        log_info("多进程任务结束")
        print("Running Time:{0}".format(time.time() - startTime))
        log_info("爬虫程序终止")

if __name__ == "__main__":
    # 
    #data_path = sys.argv[2]
    data_path = '../mengniu/'
    # 评论信息存储路径
    csv_file = data_path + 'data/product_information_mn.csv'
    # 产品url存储路径
    url_file = data_path + 'url/product_url.csv'
    # 定义类对象 
    jd = JD(url_file=url_file, csv_file=csv_file, data_path = data_path)
    # 读取url，返回url列表
    urls = jd.url_list()
    # 是否多进程爬取数据
    # multiprocess = sys.argv[1]
    multiprocess = 'T'
    if multiprocess == 'F':
        for item in urls:
            jd.write_info(product_info = item)
            # time.sleep(2)
    elif multiprocess == 'T':
        jd.run(ids = urls)
