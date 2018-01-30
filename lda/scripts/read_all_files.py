#!/opt/Anaconda3/bin/python
# -*- coding: UTF-8 -*-

# ********************************************************
# * Author        : lyz
# * Email         : lyz038015@163.com
# * Create time   : 2018-01-28 21:59
# * Last modified : 2018-01-28 21:59
# * Filename      : read_all_files.py
# * Description   : 
# *********************************************************
"""
读取文件夹下所有文本信息
"""
print(__doc__)
import os,sys
import codecs
import numpy as np

sys.path.append('/home/arron/ligl/lda/')
from p_library import clean_data

class dir_files():
    """
    Function : read all texts in the folder
    """
    def __init__(self,dir_path):
        """
        dir_path: 目标路径
        """
        self.dir_path = dir_path
    
    def read_files(self):
        """
        param  : none
        return : a list(每个对象包含一个三元组)
        """
        # 获取文件夹下的所有文件名称
        files= os.listdir(self.dir_path)

        # 文本内容列表对象
        file_content_list = list()
        #
        id2sentence = dict()
        # 遍历文件夹
        for index, file_temp in enumerate(files): 
            # 判断是否为文件、文件后缀为.txt
            if not os.path.isdir(file_temp) and file_temp.endswith(".txt"):
                reader = codecs.open(self.dir_path + "/" + file_temp, 'rb', encoding='utf-8')
                # 创建迭代器
                lines = reader.readlines()
                str_temp = ""
                # 遍历文件，读取文本
                for line in lines:
                    str_temp = str_temp + line
                str_temp = clean_data.extract_chinese(str_temp)
                # 2元组：
                # product name, product info
                file_content_list.append([file_temp, str_temp])
                # 语句与序号的对应关系
                id2sentence[index] = file_temp
                # 关闭文件
                reader.close()
        #
        return file_content_list, id2sentence

    def store_senNum(self, id2sentence,senNum_name):
        """

        """
        with open(senNum_name, 'w', encoding="utf-8") as writer:
            for key in id2sentence.keys():
                writer.write(str(key) + ':' + id2sentence[key] + '\n')

if __name__ == "__main__":
    dir_path = '../product_info'
    dfiles = dir_files(dir_path = dir_path)
    files_contents, id2sentence = dfiles.read_files()
    dfiles.store_senNum(id2sentence = id2sentence, senNum_name = './product_info.txt')
    #
    print(len(files_contents))
