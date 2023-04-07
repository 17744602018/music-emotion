#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2023/03/15 17:03:56
@Author  :   yiyizhang 
@Version :   1.0
@Desc    :   帮助类
'''
import csv 
class Utils:

    """
    返回CSV文件的表头和
    字典对象列表
    """
    @staticmethod
    def get_csv_dic_list(path):
        file = open(path,mode="r",encoding="utf-8")
        csv_dic = csv.DictReader(file)
        fieldnames = csv_dic.fieldnames
        data = []
        for dic in csv_dic:
            data.append(dict(dic))
        file.close()
        return fieldnames,data 

