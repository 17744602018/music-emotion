#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   logistic.py
@Time    :   2023/04/14 11:04:32
@Author  :   yiyizhang 
@Version :   1.0
'''

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from utils import Utils

# from dataload.loaddata import datas, labels
# from dataload.randompickd import datas, labels
from dataload.fusiondata import * 
print("Logistic")
def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    return {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1}


kf = KFold(n_splits=5, shuffle=True)
results = []
for train_idx, test_idx in kf.split(datas):
    X_train, y_train = datas[train_idx], labels[train_idx]
    X_test, y_test = datas[test_idx], labels[test_idx]

    # Define the logistic regression model with regularization penalty set to L2
    # Note: You can adjust the hyperparameters here to optimize the performance of the model
    clf = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, solver='liblinear', random_state=42)
    clf.fit(X_train, y_train)

    # Predict the test data using the trained model
    y_pred = clf.predict(X_test)

    # Evaluate the model using various metrics
    results.append(evaluate(y_test, y_pred))
    print(classification_report(y_test,y_pred))
    print("auc",Utils().multiclass_roc_auc_score(y_test,y_pred))

# Print the average of the evaluation results across the 5 folds
avg_results = {k: np.mean([r[k] for r in results]) for k in results[0]}
# print(avg_results)