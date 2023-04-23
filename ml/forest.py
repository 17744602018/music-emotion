#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   forest.py
@Time    :   2023/04/14 11:09:44
@Author  :   yiyizhang 
@Version :   1.0
'''
from sklearn.model_selection import GridSearchCV,train_test_split
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



# 定义保存评估结果的列表
rf_accuracy = []
X_train, X_test, y_train, y_test = train_test_split(datas, labels, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [150],
    'max_depth': [15],
    'min_samples_split': [4],
    'max_features':[len(datas[0])]
}
# 定义随机森林分类器
# forest_model = RandomForestClassifier(random_state=42)


# 定义网格搜索对象
# grid_search = GridSearchCV(forest_model, param_grid, cv=5)

# # 进行交叉验证和参数调整
# grid_search.fit(X_train, y_train)

# # 输出最佳参数组合
# print("Best parameters: {}".format(grid_search.best_params_))

# 使用最佳参数组合训练模型 融合的max_depth = 15
best_forest_model = RandomForestClassifier(max_depth=15,max_features=100,min_samples_split=4,n_estimators=150,random_state=42)
best_forest_model.fit(X_train, y_train)

y_pred = best_forest_model.predict(X_test)
# 评估模型并保存结果
acc = evaluate(y_test, y_pred)
print(y_test[:20])
print(y_pred[:20])
rf_accuracy.append(acc)
print(classification_report(y_test,y_pred))
print("auc",Utils().multiclass_roc_auc_score(y_test,y_pred))

