#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   forest.py
@Time    :   2023/04/14 11:09:44
@Author  :   yiyizhang 
@Version :   1.0
'''

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from utils import Utils
# from dataload.loaddata import datas, labels
# from dataload.randompickd import datas, labels
from dataload.fusiondata import * 
print("Forest")
# 定义评估函数
def evaluate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

# 定义K折交叉验证
kf = KFold(n_splits=2, shuffle=True)

# 定义随机森林分类器
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# 定义保存评估结果的列表
rf_accuracy = []

# K折交叉验证
for train_index, test_index in kf.split(datas):
    # 分割训练集和测试集
    X_train, X_test = datas[train_index], datas[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    # 训练模型并预测
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # 评估模型并保存结果
    acc = evaluate(y_test, y_pred)
    rf_accuracy.append(acc)
    print(classification_report(y_test,y_pred))
    print("auc",Utils().multiclass_roc_auc_score(y_test,y_pred))

# 打印平均准确率
print('随机森林平均准确率：', np.mean(rf_accuracy))
