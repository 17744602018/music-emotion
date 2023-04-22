#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   bert.py
@Time    :   2023/04/12 08:39:54
@Author  :   yiyizhang 
@Version :   1.0
@Desc    :   将歌词分为多个部分，然后放入TF-IDF中做特征提取后返回
'''


import pandas as pd
import csv
import numpy as np
from .tfidf import lyric_tf_idf

labels = []
datas = []


annotation = "/Users/yiyizhang/Desktop/PMEmo2019/newannotations/thayer_static_annotations.csv"
arf = open(annotation,mode="r",encoding="utf-8")
annotation_reader = list(csv.DictReader(arf))

# 添加标注
for a in annotation_reader:
    music_id= int(a["musicId"])
    X = lyric_tf_idf(music_id)
    if X is not None:
        lable = a["Thayer"].strip("]").strip("[").split(",")
        lable[0] = int(lable[0])
        lable[1] = int(lable[1])

        if lable[0] == 1 and lable[1] == 1:
            lable = 0
        elif lable[0] == -1 and lable[1] == 1:
            lable = 1
        elif lable[0] == -1 and lable[1] == -1:
            lable = 2
        else:
            lable = 3
        for item in X:
            item.astype(float)
            item = np.pad(item,(0,100-len(item)),'constant')
            datas.append(item)
            labels.append(lable)
datas = np.array(datas)
labels = np.array(labels)
arf.close()


