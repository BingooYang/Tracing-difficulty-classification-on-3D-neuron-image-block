# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:06:36 2020

@author: Administrator
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import xlrd
import os

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


def save_label_txt(neuron_name,data):
    name = 'method2_1to0_classification_label_1115_clf176'+'.txt'
    path = os.getcwd().replace('\\','/')
    path = path + '/'+name
    print("保存文件的路径：",path)
    f = open(path,mode='w')
    f.write('#### 序号    name    label \n')
    for i in range(data.shape[0]):
        f.writelines([str(i),'\t',str(neuron_name[i][0]),'\t',str(data[i]),'\n'])

def save_error_true_label_txt(predict_name,manual_name,data):
    name = 'only_error_label'+'.txt'
    path = os.getcwd().replace('\\','/')
    path = path + '/'+name
    print("保存文件的路径：",path)
    f = open(path,mode='w')
    f.write('#### 序号    predict_name    manual_predict    predict_label    manual_label    flag \n')
    for i in range(data.shape[0]):
        flag = -1
        if(data[i][0] == data[i][1] or (data[i][0]==2 and data[i][1]==1)):
            flag = 0
        else:
            flag = 1
            f.writelines([str(i),'\t',str(predict_name[i][0]),'\t',str(manual_name[i][0]),'\t'
                          ,str(data[i][0]),'\t',str(data[i][1]),'\t',str(flag),'\t\n'])

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

def main():
#    label2_path = "F:/004Vaa3d/002Data/001/6.27/neuron_distance_analyse/dis_score_method2.txt"
#    model_path = "F:/004Vaa3d/002Data/001/6.27/neuron_distance_analyse/model/two_classification_1to2_method2_label/MLP_clf4_47.pkl"
    label2_path = "F:\\004Vaa3d\\004feature\\nd_lm_tn_method2_905.txt"
    model_path = "F:\\004Vaa3d\\005label_classification_model\\1113\\method2_1to2_5\\MLP_clf4_176.pkl"
    
#    label_path = "F:\\004Vaa3d\\003label\\001_latest_label_904.xlsx"
    #读数据
    print("start read data...")
    name,data = read_distance(label2_path)
    print(data.shape)
    print("read data complete...")
    print("---------------------")
    #取特征[1, 3, 4]
    feasure =[0, 1, 3, 5, 7]
    data_x = np.zeros((data.shape[0],0))

    for i in range(len(feasure)):
        data_x = np.column_stack((data_x,data[:,feasure[i]]))
    
    #导入模型
    clf = joblib.load(model_path)
# 
#    manual_label_name,manual_label_data = get_method2_label(label_path)
    
    predict_label = clf.predict(data_x)
    
    print(predict_label.shape)
    
    print("start save auto_label txt...")
    save_label_txt(name,predict_label)
    print("complete save")
    print("-------------")

    

main()    

