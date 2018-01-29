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

class dir_files():
    """
    Function : read all texts in the folder
    """
    def __init__(self,dir_path):
        """
        dir_path: 目标路径
        """
        self.dir_path = dir_path
    
    def read_all_files(self):
        """
        param  : none
        return : a list(每个对象包含一个三元组)
        """
        # 获取文件夹下的所有文件名称
        files= os.listdir(self.dir_path)

        # 文本内容列表对象
        file_content_list = list()

        # 遍历文件夹
        for index, file_temp in enumerate(files): 
            # 判断是否为文件、文件后缀为.txt
            if not os.path.isdir(file_temp) and file_temp.endswith(".txt"):
                reader = codecs.open(dir_path + "/" + file_temp, 'rb', encoding='utf-8')
                # 创建迭代器
                lines = reader.readlines()
                str_temp = ""
                # 遍历文件，读取文本
                for line in lines:
                    str_temp = str_temp + line
                # 三元组：
                # index, product name, product info
                file_content_list.append([index, file_temp, str_temp])
                # 关闭文件
                reader.close()
        #
        return file_content_list


if __name__ == "__main__":
    dir_path = './product_info'
    dfiles = dir_files(dir_path = dir_path)
    files_contents = dfiles.read_all_files()
    #
    print(len(files_contents))
    print(np.array(files_contents)[:,1])
