#!/opt/Anaconda3/bin/python
# -*- coding: UTF-8 -*-

# ********************************************************
# * Author        : lyz
# * Email         : lyz038015@163.com
# * Create time   : 2018-01-02 20:50
# * Last modified : 2018-01-02 20:50
# * Filename      : social_science_funds.py
# * Description   : 国家社科基金数据库 
# *********************************************************

import requests
import csv
import re
import time

"""
国家社科基金数据库
"""

def run():
    # 定义csv文件
    file_path = './data/social_science_funds.csv'
    f = open(file_path, 'a', newline='')
    writer = csv.writer(f)
    writer.writerow(['项目批准号', '项目类别', '学科分类', '项目名称', '立项时间', '项目负责人', '专业职务',
                        '工作单位', '单位类别', '所在省区市', '所属系统', '成果名称', '成果形式', '成果等级',
                        '结项时间', '结项证书号', '出版社', '出版时间', '作者', '获奖情况'])
    #
    i = 0
    
    while i< 3154:
        # 头URL
        start_url = 'http://fz.people.com.cn/skygb/sk/?&p={0}'.format(i)

        # 获取源码
        r = requests.get(start_url)
        result_temp = r.text

        # re匹配内容
        re_ruler = r'<td width=.*?>.*?<span title.*?>(.*?)</span>.*?</td>'
        re_temp = re.compile(re_ruler, re.S)
        re_result = re.findall(re_temp, result_temp)
        
        # 写入数据
        for j in range(0, len(re_result), 19):
            writer.writerow([re_result[j], re_result[j+1], re_result[j+2], re_result[j+3], re_result[j+4], re_result[j+5], re_result[j+6],
                            re_result[j+7], re_result[j+8], re_result[j+9], re_result[j+10], re_result[j+11], re_result[j+12], re_result[j+13], 
                            re_result[j+14], re_result[j+15], re_result[j+16], re_result[j+17], re_result[j+18], ''])

        # 查看进度
        if i%100 == 0:
            print("爬去进度:{0}".format(i/3154))
        i += 1
        time.sleep(2)
    writer.close()

if __name__ =="__main__":
    run()
