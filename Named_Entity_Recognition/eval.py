#!/opt/Anaconda3/bin/python
# -*- coding: UTF-8 -*-

# ********************************************************
# * Author        : lyz
# * Email         : lyz038015@163.com
# * Create time   : 2018-02-05 11:00
# * Last modified : 2018-02-05 11:00
# * Filename      : eval.py
# * Description   : 
# *********************************************************
import os

def conlleval(label_predict, label_path, metric_path):
    """
    param label_predict : 
    param label_path    :
    param metric_path   :
    return              :
    """

    eval_perl = "./conlleval_rev.pl"
    with open(label_path, "w") as fw:
        line = list()
        for sent_result in label_predict:
            for char, tag, tag_ in sent_result:
                tag = '0' if tag == 'O' else tag
                char = char.encode("utf-8")
                line.append("{} {} {}\n".format(char, tag, tag_))
            line.append("\n")
        fw.writelines(line)
    os.system("perl {} < {} > {}".format(eval_perl, label_path, metric_path))
    with open(metric_path) as fr:
        metrics = [line.strip() for line in fr]
    return metrics
    
