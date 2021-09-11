# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 10:42:46 2020

@author: Administrator
"""

import numpy as np
import xlrd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
import itertools
from sklearn import metrics
import time

from sklearn.externals import joblib

def Find_max_min(data):
    d_max = 0
    d_min = 1
    for i in range(data.shape[0]):
        if(data[i]>d_max):
            d_max = data[i]
        if(data[i]<d_min):
            d_min = data[i]
    return d_max,d_min

def normalization(data):
    #找每一列最大最小值
    d_max = np.zeros((data.shape[1],1))
    d_min = np.zeros((data.shape[1],1))
    for i in range(data.shape[1]):
        d_max[i],d_min[i] = Find_max_min(data[:,i])
    #归一化
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j] = (data[i][j]-d_min[j])/d_max[j]
    return data

def get_method1_label(path):
#    path = "F:/004Vaa3d/002Data/001/6.27/802标签.xlsx"
    #读excel文件
    data = xlrd.open_workbook(path)
    #打开sheet1
    table = data.sheet_by_index(0)
    method1_lable = np.zeros((0))
    name = np.array((0))
#    #取第六列数据,manual_label
#    for i in range(len(table.col_values(6))):
#        #前两行是注释，从第三行开始写入
#        if(i>1):
#            manual_lable = np.append(manual_lable,table.col_values(6)[i])
    #取第八列数据,method1_lable
    for i in range(len(table.col_values(8))):
        #前两行是注释，从第三行开始写入
        if(i>1):
            if(table.col_values(8)[i] !=0 and table.col_values(8)[i] !=1 and table.col_values(8)[i] !=2 and table.col_values(8)[i] !=-1 ):
                break
            method1_lable = np.append(method1_lable,table.col_values(8)[i]) 
            #取第一列数据,neuron_name
            name = np.append(name,table.col_values(1)[i])
    name = np.delete(name,0)
    
    return name.reshape(len(name),1),method1_lable.reshape((len(method1_lable),1))

def get_method2_label(path):
#    path = "F:/004Vaa3d/002Data/001/6.27/802标签.xlsx"
    #读excel文件
    data = xlrd.open_workbook(path)
    #打开sheet1
    table = data.sheet_by_index(0)
    method2_lable = np.zeros((0))
    name = np.array((0))
    
    #取第一列数据,neuron_name 
    #取第九列数据,method2_lable
    for i in range(len(table.col_values(9))):
        #前两行是注释，从第三行开始写入
        if(i>1):
            if(table.col_values(9)[i] !=0 and table.col_values(9)[i] !=1 and table.col_values(9)[i] !=2 and table.col_values(9)[i] !=-1 ):
                break
            if(table.col_values(9)[i]>=0):
                method2_lable = np.append(method2_lable,table.col_values(9)[i])   
                name = np.append(name,table.col_values(1)[i])
    name = np.delete(name,0)
    
    return name.reshape(len(name),1),method2_lable.reshape((len(method2_lable),1))

def read_distance(path):
    #读文本数据
    f = open(path) 
    ftextlist = f.readlines()
    #删除注释行
    ftextlist.pop(0)
    line = len(ftextlist[0].split('\t'))-1
#    print(line)
    data_y = np.zeros((0,len(ftextlist)))
    data_x = np.array(0)
    for s in ftextlist:
        str = ''.join(s)
        data_x = np.append(data_x,(str.split('\t')[1]))
    data_x = np.delete(data_x,0)
    for i in range(2,line):
        tem = np.zeros(0)
        for s in ftextlist:
            str = ''.join(s)
            tem = np.append(tem,float(str.split('\t')[i]))
        tem = tem.reshape(1,len(tem))
        data_y = np.append(data_y,tem,axis=0)
    return data_x.reshape(len(data_x),1),data_y.T

def Find_max(data,max_value,max_num):
    l = len(data)
    for i in range(l):
        if(data[i]>max_value):
            max_value = data[i]
            max_num = i
    return max_value,max_num

def Find_min(data,min_value,min_num):
    l = len(data)
    for i in range(l):
        if(data[i]<min_value):
            min_value = data[i]
            min_num = i
    return min_value,min_num

def save_combinations(data):
    name = 'combinations'+'.txt'
    path = os.getcwd().replace('\\','/')
    path = path + '/'+name
    print("保存文件的路径：",path)
    f = open(path,mode='w')
    f.write('#### 序号    组合 \n')
    for i in range(len(data)):
        f.writelines([str(i),'\t',str(data[i]),'\t','\n'])
    
def classification(data,label):
    if(data.shape[0] != label.shape[0]):
        print("data shape mismatch label shape")
        print("-------------------------------")
        return
    
    #记录标签有几类
    label_variaty = len(set(label.flatten().tolist()))
    print("label_variaty:",label_variaty)
    #生成组合序列，总共127种
#    list1 = [0,1,2,3,4,5,6]
    list1 = [0,1,2,3,4,5,6,7]
    list2 = []
    for i in range(1,len(list1)+1):
#        print("i:",i)
        iter1 = itertools.combinations(list1,i)
        for j in iter1:
            list2.append(list(j))
#    save_combinations(list2)

#    取每一种组合的数据
#    re = zeros((0))
    re = []
    for i in range(len(list2)):
        tem = []
        for j in range(len(list2[i])):
#            np.c_(re,data[:,list2[i][j]])
            tem.append(data[:,list2[i][j]])
        re.append(tem)
    #转置(n,802)->(802,n)
    for i in range(len(re)):
        re[i] = list(np.array(re[i]).T)      

    #用分类器进行分类
#    label_num = 3
    classification_name = ["Logistic","SVM","NN1","NN2"]
    score_max = np.zeros((len(classification_name)))
    score_min = np.ones((len(classification_name)))
    score_max_num = np.zeros((len(classification_name)))
    score_min_num = np.zeros((len(classification_name)))
    score = np.zeros((len(re),len(classification_name)))
    
    con_matrix = np.zeros((len(classification_name),len(re),label_variaty,label_variaty))

    #取ave_n次平均
    print("start classify...")
    ave_n = 5
#    score_max = np.zeros(4)
#    score_min = np.ones(4)
#    for m in range(label_num):
    clf1 = LogisticRegression(random_state=0,penalty='l2')
    clf3 = MLPClassifier(random_state=0,learning_rate='adaptive',max_iter=3000,alpha=0.001)
    for n in range(ave_n):
        for i in range(0,len(re)):
#            print("complete:%d/%d" %(i+n*len(re)+1,len(re)*ave_n))
#            x_train,x_test,y_train,y_test = train_test_split(re[i],label,test_size=0.3,random_state=0)
            x_train,x_test,y_train,y_test = train_test_split(re[i],label,test_size=0.3)
            clf1.fit(x_train, y_train)
#            clf2 = SVC(random_state=0).fit(x_train, y_train)
            clf3.fit(x_train, y_train)
            clf4 = MLPClassifier(random_state=0,learning_rate='adaptive',max_iter=3000,hidden_layer_sizes=(10,30,60,30),alpha=0.001).fit(x_train, y_train)
            
            #混淆矩阵
            con_matrix[0][i] = con_matrix[0][i] + metrics.confusion_matrix(y_test,clf1.predict(x_test))/ave_n
#            con_matrix[1][i] = con_matrix[1][i] + metrics.confusion_matrix(y_test,clf2.predict(x_test))/ave_n
            con_matrix[2][i] = con_matrix[2][i] + metrics.confusion_matrix(y_test,clf3.predict(x_test))/ave_n
            con_matrix[3][i] = con_matrix[3][i] + metrics.confusion_matrix(y_test,clf4.predict(x_test))/ave_n
            
            #正确率
            score[i][0] = score[i][0] + clf1.score(x_test,y_test)/ave_n
#            score[i][1] = score[i][1] + clf2.score(x_test,y_test)/ave_n
#            score[i][1] = score[i][1] + clf4.score(x_train,y_train)/ave_n
            score[i][2] = score[i][2] + clf3.score(x_test,y_test)/ave_n
#            score[i][3] = score[i][3] + clf4.score(x_test,y_test)/ave_n
            score[i][3] = score[i][3] + clf4.score(x_test,y_test)/ave_n
            
#            print(score)
            #保存模型
            if(n == (ave_n-1)):
                model_name = 'F:\\004Vaa3d\\005label_classification_model\\1115\\method2_1to2_5\\'+'MLP_clf4_'+ str(i) + '.pkl'
                joblib.dump(clf4, model_name)
            
        for i in range(len(classification_name)):
            score_max[i],score_max_num[i] = Find_max(score[:,i],score_max[i],score_max_num[i])
            score_min[i],score_min_num[i] = Find_min(score[:,i],score_max[i],score_max_num[i])
            
#    print(score)      
    print("score_max:",score_max,"score_min:",score_min)
    print("score_max_num:",score_max_num,"score_min_num:",score_min_num)
#    for m in range(label_num):
    for i in range(len(classification_name)):
        print("label %s confusion_matrix:"%(classification_name[i]),con_matrix[i][int(score_max_num[i])])    
#def save_model(path,):
    print("complete classify")
    print("----------------")      
    
def three_classification(data1,label1,data2,label2):
    #manual_label作为标签学习
#    classification(data1,label1[:,0])       
    #method1_label作为标签学习
    classification(data1,label1)  
#    #method2_label作为标签学习
#    classification(data2,label2)  

    
def two_1to2_classification(data1,label1,data2,label2):
    for i in range(label1.shape[0]):
        for j in range(label1.shape[1]):
            if(label1[i][j] == 1):
                label1[i][j] = 2
    for i in range(label2.shape[0]):
        for j in range(label2.shape[1]):
            if(label2[i][j] == 1):
                label2[i][j] = 2
                
    #manual_label作为标签学习
#    classification(data1,label1)       
    #method1_label作为标签学习
    classification(data1,label1)  
#    #method2_label作为标签学习
    classification(data2,label2)                  

def two_1to0_classification(data1,label1,data2,label2):
    for i in range(label1.shape[0]):
        for j in range(label1.shape[1]):
            if(label1[i][j] == 1):
                label1[i][j] = 0
    for i in range(label2.shape[0]):
        for j in range(label2.shape[1]):
            if(label2[i][j] == 1):
                label2[i][j] = 0
    #manual_label作为标签学习
#    classification(data1,label1)       
    #method1_label作为标签学习
    classification(data1,label1)  
#    #method2_label作为标签学习
    classification(data2,label2)    

def save_error_type(name,predict_label,y,true_label):
#    print(name.shape,predict_label.shape,y.shape)
    save_name = 'error_type'+'.txt'
    path = os.getcwd().replace('\\','/')
    path = path + '/'+save_name
    print("保存文件的路径：",path)
    f = open(path,mode='w')

    f.write('#### n    name   predict   true   flag\n')
    for i in range(name.shape[0]):
        if(predict_label[i] != true_label[i]):
            f.writelines([str(i),'\t',str(name[i][0]),'\t',str(predict_label[i]),'\t',str(y[i][0]),'\n'])
#        f.writelines([str(i),'\t',str(name[i][0]),'\t',str(predict_label[i]),'\t',str(y[i][0]),'\n'])
    f.close()
    
def error_type_analyze(model_path,name,x,y):
    label = np.zeros(0)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if(y[i][j] == 0):
                label = np.append(label,0)
            else:
                label = np.append(label,2)
    label = label.reshape((len(label),1))  
    
    #选特征
    x = x[:,[0, 1, 2, 5]]
#        #导入模型
    clf = joblib.load(model_path)
    predict_label = clf.predict(x)
    print(clf.score(x,label))
    save_error_type(name,predict_label,y,label)
                
def main():
    starttime = time.time()
    label_path = "F:\\004Vaa3d\\003label\\001_latest_label_1113.xlsx"
#    distance1_path = "F:\\004Vaa3d\\004feature\\neuron_distance\\nd_lm_method1.txt"
#    distance2_path = "F:\\004Vaa3d\\004feature\\neuron_distance\\nd_lm_method2.txt"
    distance1_path = "F:\\004Vaa3d\\004feature\\nd_lm_tn_method1_905.txt"
    distance2_path = "F:\\004Vaa3d\\004feature\\nd_lm_tn_method2_905.txt"   
    
    print("load label:")
    label1_name,label1_data = get_method1_label(label_path)
    label2_name,label2_data = get_method2_label(label_path)
    print(label1_data.shape)
    print(label2_data.shape)   
    #取distance文件的第一列作为distance_name，取[2,3,5,6,7,9,10]列作为distance_data
    distance1_name,distance1_data = read_distance(distance1_path)
    distance2_name,distance2_data = read_distance(distance2_path)
    
    print("load neuron_distance:")
    #取distance2打标签的前label2_name.shape[0]项
    distance1_name = distance1_name[:label1_name.shape[0]]
    distance1_data = distance1_data[:label1_name.shape[0]]
    distance2_name = distance2_name[:label2_name.shape[0]]
    distance2_data = distance2_data[:label2_name.shape[0]]
    print(distance1_data.shape,distance1_name.shape)
    print(distance2_data.shape)
    
#    print(distance1_data[0])
#    print("start nomalization...")
#    distance1_data = normalization(distance1_data)
#    distance2_data = normalization(distance2_data)
#    print(distance1_data[0])
    #三分类
#    three_classification(distance1_data,label1_data,distance2_data,label2_data)

#    #二分类
    print('start 1to2...')
 #    
    two_1to2_classification(distance1_data,label1_data,distance2_data,label2_data)

    print('start 1to0...')
#    #二分类
    two_1to0_classification(distance1_data,label1_data,distance2_data,label2_data)  
    
#    #错误情况分析            
#    print("start error type analyze...")
#    model_path = "F:\\004Vaa3d\\005label_classification_model\\814\\method2_1to2_1\\MLP_clf4_65.pkl"
#
#    error_type_analyze(model_path,distance2_name,distance2_data,label2_data)
    
     
    endtime = time.time()
    print('总共的时间为:', (endtime - starttime),'secs')

    
main()
 








