from sklearn import svm,linear_model
from smote_variants import ADASYN, CBSO, SMOTE, SMOTE_PSO, SMOTE_PSOBAT, SMOTE_D, Borderline_SMOTE1, Safe_Level_SMOTE, Random_SMOTE
import pandas as pd
import os
import numpy as np
import sklearn.datasets as datasets
from sklearn.model_selection import KFold,train_test_split   #交叉验证
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score,roc_curve,auc,f1_score, precision_score,confusion_matrix
from sklearn.metrics import precision_recall_curve
import numpy as np
from sklearn import svm
import PSO,SSA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


def spe(Y_test,Y_pred,n):
    
    spe = []
    con_mat = confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        fp = np.sum(con_mat[:,i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)
    
    return spe


def findX(test_other):
    global new_data_list
    
    cost = 0
    #test_other  表示寻优的x的数据
    new_data = test_other
    
    #计算生成的数据与0类数据的距离
    dist_list_0 = []
    for ind in class_0:
        dist = np.linalg.norm(test_other - scaler_x[ind])
        dist_list_0.append(dist)
    dist_0 = np.mean(dist_list_0)
    #print(len(dist_list_0))
    #print(dist_0)
    
    #计算生成的数据与1类数据的距离
    dist_list_1 = []
    for ind in class_1:
        dist = np.linalg.norm(test_other - scaler_x[ind])
        dist_list_1.append(dist)
    dist_1 = np.mean(dist_list_1)
    #print(len(dist_list_1))
    #print(dist_1)
    
    #计算生成的数据与0类数据的相似度
    sim_list_0 = []
    vec1 = np.array(test_other).reshape(1,-1)
    for ind in class_0:
        vec2 = np.array(scaler_x[ind]).reshape(1,-1)
        sim = cosine_similarity(vec1 , vec2)
        sim_list_0.append(sim[0][0])
    sim_0 = np.mean(sim_list_0)
    #cos_sim = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))
    #print(sim_0)

    #计算生成的数据与0类数据的
    sim_list_1 = []
    for ind in class_1:
        vec2 = np.array(scaler_x[ind]).reshape(1,-1)
        sim = cosine_similarity(vec1 ,vec2)
        sim_list_1.append(sim[0][0])
    sim_1 = np.mean(sim_list_1)
    #cos_sim = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))
    #print(sim_1)
    
    if dist_1 - dist_0 >= 0.01 and sim_0 - sim_1 >=0.01:
        if len(new_data_list) == 0:
            new_data_list.append(test_other) 
            
            new_dis.append(dist_0 - dist_1)
            new_sim.append(sim_0 - sim_1)
            
        else:
            #计算新产生的数据跟原数据之间的差距
            dist_new_data = []
            for data in new_data_list:
                dist = np.linalg.norm(test_other - data)
                dist_new_data.append(1/dist)
            if np.mean(dist_new_data)<=10:
                new_data_list.append(test_other)
            else:
                cost += np.mean(dist_new_data)
    
    if len(new_data_list) > len(class_1) - len(class_0):
        new_data_list = np.array(new_data_list)
        print(len(new_data_list),len(class_1) - len(class_0))
        list_ind = np.argsort(new_dis)
        new_data_list = new_data_list[list_ind[:len(class_1) - len(class_0)]]
    cost += dist_1+sim_1-(dist_0+sim_0)
    return cost

def geneData(x,y):
    # x: 输入对应的x
    # y: 输入对应的类别
    scaler = MinMaxScaler()
    global scaler_x,data_class, class_0,class_1,new_dis,new_sim,new_data_list
    new_data_list = []
    new_dis = []
    new_sim = []
    scaler_x = scaler.fit_transform(x)
    data_class = y
    if len(np.where(data_class == 0)[0]) > len(np.where(data_class == 1)[0]):
        less = 1
        more = 0
    else:
        less = 0
        more = 1
    class_0 = np.where(data_class == less)[0]
    class_1 = np.where(data_class == more)[0]
    maxgen = 10
    pso = SSA.SSA(findX, [0.00000000001]*len(x[0]),[1]*len(x[0]),
                     len(x[0]), 50, maxgen)
    print(np.array(new_data_list).shape)
    #findX(pso.bestFoodPositions[-1])
    new_data= scaler.inverse_transform(new_data_list)
    new_clss = [0 for _ in range(len(new_data))]
    all_data = np.concatenate((x, new_data),axis=0)
    all_class = np.concatenate((data_class, new_clss))
    print(all_data.shape, all_class.shape)
    return all_data,all_class

