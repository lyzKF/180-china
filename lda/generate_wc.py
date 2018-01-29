# coding:utf-8
"""
wordcloud生成中文词云
"""
print(__doc__)
from wordcloud import WordCloud
import re
import jieba
from scipy.misc import imread
import matplotlib.pyplot as plt
from optparse import OptionParser
import sys

USAGE = 'usage: python filename.py -p [file_path] -n [pic_name]'

parser = OptionParser()
parser.add_option(
        "-p",
        "--file path",
        dest = "file_path",
        help="指定文件路径名称")
parser.add_option(
        "-n",
        "--picture name",
        dest = "pic_name",
        help="生成词云图片的名称")
(opt,args) = parser.parse_args()

#
if len(args) < 0:
    print(USAGE)
    sys.exit(1)

#
file_path = opt.file_path
pic_name = opt.pic_name

def extract_chinese(str_temp):
    """
    """
    line = str_temp.strip()
    
    p2 = re.compile(r'[^\u4e00-\u9fa5]')

    zh = " ".join(p2.split(line)).strip()
    zh = ",".join(zh.split())
    Outstr = zh
    return Outstr

# 绘制词云
def draw_wordcloud():
    #读入一个txt文件
    comment_text = open(file_path,'r',encoding='utf-8').read()
    comment_text = extract_chinese(comment_text)
    
    #结巴分词，生成字符串，如果不通过分词，无法直接生成正确的中文词云
    cut_text = " ".join(jieba.cut(comment_text))
    
    color_mask = imread("dogs.jpg") # 读取背景图片
    
    font_ch = '/System/Library/Fonts/STHeiti Light.ttc'
    
    cloud = WordCloud(
            # 设置字体
            font_path= font_ch,
            # 画布宽度 default = 400
            width = 400,
            # 画布高度 default = 200
            height = 200,
            # 词语水平方向排版出现的频率 default = 0.9
            # preper_horizontal = 0.8,
            # 如果为空，则使用二维遮罩，非空时设置的width和height将会
            # 被忽略，遮罩形状被mask取代。
            # mask = color_mask,
            # 比例放大画布
            scale = 3,
            # 背景颜色
            background_color = "white",
            # 最大词汇
            max_words = 300,
            # 最大号字体
            max_font_size = 80,
            # 最小号字体，default = 4
            min_font_size = 4,
            # 如果为空，则使用内置的停用词，
            stopwords = None,
            # 给单词随机分配颜色，default = viridis
            colormap = 'viridis',
            # 是否包括两个词的搭配，default = True
            collocations = True,
            # 词频和字体大小的关联性 default = 0.5
            relative_scaling = 0.5,
            # 颜色模式 default = 'RGB' (GRBA)
            mode = 'RGBA'
            )
    # 生成词云
    word_cloud = cloud.generate(cut_text) 
    # 保存图片
    word_cloud.to_file(pic_name) 
    #  显示词云图片
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.show()



if __name__ == '__main__':
    draw_wordcloud()
