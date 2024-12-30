# -*- coding: utf-8 -*-
# @Time    : 2024/12/30 21:26
# @Author  : sjx_alo！！
# @FileName: main.py
# @Algorithm ：
# @Description:


from OOF import geneData, spe
import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, precision_score, confusion_matrix

path = './dataset/'
data_list = []
res = {'acu': [], 'f1': [], 'gmean': []}

folder_list = os.listdir(path)
for folder in folder_list:

    acc = []
    f_score = []
    gmean_list = []
    for n_time in range(10):
        for i in range(5):
            train_name = path + folder + '/5-' + str(i + 1) + 'tra.dat'
            test_name = path + folder + '/5-' + str(i + 1) + 'tst.dat'

            train_data = pd.read_csv(train_name, header=None)
            test_data = pd.read_csv(test_name, header=None)

            train_data_x = train_data.iloc[:, :-1]
            train_data_class = train_data.iloc[:, -1]
            test_data_x = test_data.iloc[:, :-1]
            test_data_class = test_data.iloc[:, -1]
            train_data_y = [1 if 'positive' in data else 0 for data in train_data_class]
            test_data_y = [1 if 'positive' in data else 0 for data in test_data_class]
            train_data_x = np.array(train_data_x)
            train_data_y = np.array(train_data_y)
            test_data_x = np.array(test_data_x)
            test_data_y = np.array(test_data_y)

            X_samp, y_samp = geneData(train_data_x, train_data_y)

            svc = svm.SVC()
            svc.fit(X_samp, y_samp)
            y_pre = svc.predict(test_data_x)

            precision = precision_score(test_data_y, y_pre)
            specificity = spe(test_data_y, y_pre, 1)[0]
            gmean_list.append(np.sqrt(precision * specificity))
            fpr, tpr, threshold = roc_curve(test_data_y, y_pre)

            acc.append(roc_auc_score(test_data_y, y_pre))
            f_score.append(f1_score(test_data_y, y_pre)) 
    res['acu'].append(np.mean(acc))
    res['f1'].append(np.mean(f_score)) 
    res['gmean'].append(np.mean(gmean_list))
    print(res)
