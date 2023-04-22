#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   gaussianNB.py
@Time    :   2023/04/14 10:56:34
@Author  :   yiyizhang 
@Version :   1.0
'''



import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from utils import Utils
# from dataload.loaddata import datas, labels
from dataload.randompickd import datas, labels
# from dataload.fusiondata import * 
print("GaussianNB")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(datas, labels, test_size=0.2, random_state=42)

# 定义GaussianNB分类器
gnb = GaussianNB(var_smoothing=1e-9)

# 训练模型
gnb.fit(X_train, y_train)

# 在测试集上进行预测和评估
y_pred = gnb.predict(X_test)
print(classification_report(y_test, y_pred))
print("auc",Utils().multiclass_roc_auc_score(y_test,y_pred))
