#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   svm.py
@Time    :   2023/04/14 09:14:51
@Author  :   yiyizhang 
@Version :   1.0
'''


import csv
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import classification_report
from utils import Utils
# from dataload.loaddata import datas, labels
# from dataload.randompickd import datas, labels
from dataload.fusiondata import * 

print("SVM")
filename = "svm_voice"
# datas = []
# labels = []
def load_local_data():
    
    with open(filename+"data.csv",mode="r",encoding="utf-8") as rf:
        csv_reader = csv.reader(rf)
        for item in csv_reader:
            temp = []
            for d in item:
                temp.append(float(d))
            datas.append(temp)
            
    
    with open(filename+"label.csv",mode="r",encoding="utf-8") as rf:
        csv_reader = csv.reader(rf)
        for item in csv_reader:
            for l in item:
                labels.append(int(l))

    
# load_local_data()
# datas = np.array(datas)
# labels = np.array(labels)


# 最大最小值归一化
for data in datas:
    # 沿着第二个轴计算每个特征维度上的最大值和最小值
    max_vals = np.max(data, axis=0)
    min_vals = np.min(data, axis=0)
    # 对每个特征维度上的特征值进行归一化处理
    data = (data - min_vals) / (max_vals - min_vals)
    

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(datas, labels, test_size=0.2, random_state=42)

# 定义超参数范围
#C正则化参数，用于控制分类器的惩罚系数。C越小，表示容忍误差的程度越高，可能导致欠拟合；C越大，表示容忍误差的程度越低，可能导致过拟合。
#kernel：核函数类型，用于将输入数据映射到高维空间中。常用的核函数有线性核函数、多项式核函数、径向基函数等。
#gamma：径向基函数的参数，用于控制径向基函数的影响范围。gamma越大，表示影响范围越小，可能导致过拟合；gamma越小，表示影响范围越大，可能导致欠拟合。
# param_grid = {'C': [ 1,2,3,4,5,6,7,9, 10, 11,12,20,30],
#               'gamma': [ 1,2,3,4,5,6,7,9, 10, 11,12,20,30],
#               'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
param_grid = {'C': [ 8],
              'gamma': [8],
              'kernel': ['poly']}

# svm_model = svm.SVC()

# 定义网格搜索对象
# grid_search = GridSearchCV(svm_model, param_grid, cv=5)

# 进行交叉验证和参数调整
# grid_search.fit(X_train, y_train)

# 输出最佳参数组合
# print("Best parameters: {}".format(grid_search.best_params_))

# 使用最佳参数组合训练模型
# best_svm_model = svm.SVC(**grid_search.best_params_)
# best_svm_model = svm.SVC(C=8,gamma=8,kernel='poly')
best_svm_model = svm.SVC(C=0.4)
best_svm_model.fit(X_train, y_train)

# 在测试集上进行预测和评估
y_pred = best_svm_model.predict(X_test)
print(classification_report(y_test, y_pred))

print("auc",Utils().multiclass_roc_auc_score(y_test,y_pred))

Utils().save_data_label(datas,labels,filename)
