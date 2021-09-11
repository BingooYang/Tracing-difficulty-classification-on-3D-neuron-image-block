# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 22:12:12 2020

@author: Administrator
"""

'''
path:路径
    path = "F:\\004Vaa3d\\002Data\\001\\6.27\\802标签.xlsx"
line:取哪些列
    line=[1,3,4],代表取802标签.xlsx文件的第1,3,4列数据
'''
import numpy as np
import os
import xlrd

def get_label(path):
    #读excel文件
    data = xlrd.open_workbook(path)
    #打开sheet1
    table = data.sheet_by_index(0)
    manual_lable = np.zeros((0))
    method1_lable = np.zeros((0))
    method2_lable = np.zeros((0))
    name = np.array((0))
    #取第一列数据,neuron_name
    for i in range(len(table.col_values(1))):
        #前两行是注释，从第三行开始写入
        if(i>1):
            name = np.append(name,table.col_values(1)[i])
    name = np.delete(name,0)
    #取第六列数据,manual_label
    for i in range(len(table.col_values(6))):
        #前两行是注释，从第三行开始写入
        if(i>1):
            manual_lable = np.append(manual_lable,table.col_values(6)[i])
    #取第八列数据,method1_lable
    for i in range(len(table.col_values(8))):
        #前两行是注释，从第三行开始写入
        if(i>1):
            method1_lable = np.append(method1_lable,table.col_values(8)[i])          
    #取第九列数据,method2_lable
    for i in range(len(table.col_values(9))):
        #前两行是注释，从第三行开始写入
        if(i>1):
            method2_lable = np.append(method2_lable,table.col_values(9)[i])   
    
    data = np.hstack((manual_lable.reshape(len(manual_lable),1),method1_lable.reshape(len(method1_lable),1)))
    data = np.hstack((data,method2_lable.reshape(len(method2_lable),1)))
#    print(data.shape)
    
    return name.reshape(len(name),1),data

def get_method2_label_name(path):
#    path = "F:/004Vaa3d/002Data/001/6.27/802标签.xlsx"
    #读excel文件
    data = xlrd.open_workbook(path)
    #打开sheet1
    table = data.sheet_by_index(0)
    name = np.array((0))
    #取第一列数据,neuron_name 
    for i in range(len(table.col_values(9))):
        #前两行是注释，从第三行开始写入
        if(i>1):
            if(table.col_values(9)[i] !=0 and table.col_values(9)[i] !=1 and table.col_values(9)[i] !=2 and table.col_values(9)[i] !=-1 ):
                break
            if(table.col_values(9)[i]>=0):
                name = np.append(name,table.col_values(1)[i])
    name = np.delete(name,0)
    
    return name.reshape(len(name),1)

def get_method1_label_name(path):
#    path = "F:/004Vaa3d/002Data/001/6.27/802标签.xlsx"
    #读excel文件
    data = xlrd.open_workbook(path)
    #打开sheet1
    table = data.sheet_by_index(0)
    name = np.array((0))
    #取第一列数据,neuron_name 
    for i in range(len(table.col_values(8))):
        #前两行是注释，从第三行开始写入
        if(i>1):
            if(table.col_values(8)[i] !=0 and table.col_values(8)[i] !=1 and table.col_values(8)[i] !=2 and table.col_values(8)[i] !=-1 ):
                break
            if(table.col_values(8)[i]>=0):
                name = np.append(name,table.col_values(1)[i])
    name = np.delete(name,0)
    
    return name.reshape(len(name),1)

def read_distance(path):
    #读文本数据
    f = open(path) 
    ftextlist = f.readlines()
    #删除注释行
    ftextlist.pop(0)
    line = len(ftextlist[0].split('\t'))
    print(line)
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
            tem = np.append(tem,(float(str.split('\t')[i])))
        tem = tem.reshape(1,len(tem))
        data_y = np.append(data_y,tem,axis=0)
    return data_x.reshape(len(data_x),1),data_y.T

#path: 文件路径
#select: [2,3,5,6,7]表示取第2,3,4.5,6,7列数据
#pre: 取前pre行数据
def read_distance_select(path,select,pre):
    #读文本数据
    f = open(path) 
    ftextlist = f.readlines()
    #删除注释行
    ftextlist.pop(0)
#    ftextlist = ftextlist[:pre]
    
#    print(line)
    data_y = np.zeros((0,len(ftextlist)))
    data_x = np.array(0)
    for s in ftextlist:
        str = ''.join(s)
        data_x = np.append(data_x,(str.split('\t')[1]))
    data_x = np.delete(data_x,0)
#    for i in range(2,line):
    for m in select:
        tem = np.zeros(0)
        for s in ftextlist:
            str = ''.join(s)
            tem = np.append(tem,(float(str.split('\t')[m])))
            
        tem = tem.reshape(1,len(tem))
        data_y = np.append(data_y,tem,axis=0)
    return data_x.reshape(len(data_x),1),data_y.T

def read_nd_lm(path):
    #读文本数据
    f = open(path) 
    ftextlist = f.readlines()
    #删除注释行
    ftextlist.pop(0)
    data_y = np.zeros((0,len(ftextlist)))
    data_x = np.array(0)
    for s in ftextlist:
        str = ''.join(s)
        data_x = np.append(data_x,(str.split('\t')[1]))
    data_x = np.delete(data_x,0)
    for i in range(2,11):
        tem = np.zeros(0)
        for s in ftextlist:
            str = ''.join(s)
            tem = np.append(tem,(float(str.split('\t')[i])))
        tem = tem.reshape(1,len(tem))
        data_y = np.append(data_y,tem,axis=0)
    return data_x.reshape(len(data_x),1),data_y.T

def read_tree_num_apart(path,label_name):
    #读文本数据
    f = open(path) 
    ftextlist = f.readlines()
    #删除注释行
    ftextlist.pop(0)
    data_y = np.zeros((0,label_name.shape[0]))
    name = np.array(0)
    for s in ftextlist:
        str = ''.join(s)
        name = np.append(name,(str.split('\t')[1]))
    name = np.delete(name,0)
    
    for m in range(2,4):
        tem = np.zeros(0)
        record=0
        for i in range(label_name.shape[0]):
            for j in range(record,len(name)):
                #找相同的名字序号
                if(name[j] == label_name[i]):
                    record = j
                    break
                if(j == len(name)):
                    print("not find same LM name")
    
            s = ftextlist[record]
            str_ = ''.join(s)
            tem = np.append(tem,(float(str_.split('\t')[m])))  
        tem = tem.reshape(1,len(tem))
        data_y = np.append(data_y,tem,axis=0)
    
    return name.reshape(len(name),1),data_y.T

def read_tree_num_all(path):
    #读文本数据
    f = open(path) 
    ftextlist = f.readlines()
    #删除注释行
    ftextlist.pop(0)
    name = np.array(0)
    for s in ftextlist:
        str = ''.join(s)
        name = np.append(name,(str.split('\t')[1]))
    name = np.delete(name, 0)
    data_y = np.zeros((0,len(name)))
    for m in range(2,4):
        tem = np.zeros(0)
        for s in ftextlist:
            str_ = ''.join(s)
            tem = np.append(tem,(float(str_.split('\t')[m])))  
        tem = tem.reshape(1,len(tem))
        data_y = np.append(data_y,tem,axis=0)
    
    return name.reshape(len(name),1),data_y.T

def save_combination_lm_nd(path,data,name,note):
    print("save_conmbination_lm_nd...")
    f = open(path,mode='w')

#写第一行注释行
    for i in note:
        f.write(i+'\t')
    #换行
    f.write('\n')
#写数据
    for i in range(data.shape[0]):
        f.write(str(i)+'\t')
        f.write(name[i][0]+'\t')
        for j in range(data.shape[1]):
            f.write(str(data[i][j])+'\t')
        #换行，除最后一行
        if(i != (data.shape[0]-1)):
            f.write('\n')
    f.close()

def save_neuron_name(data):
    name = 'neuron_name2'+'.txt'
    path = os.getcwd().replace('\\','/')
    path = path + '/'+name
    print("保存文件的路径：",path)
    f = open(path,mode='w')
    f.write('#### 序号    组合 \n')
    for i in range(len(data)):
        f.writelines([str(i),'\t',str(data[i]),'\t','\n'])
    f.close()

def read_dir_save(path):
    
    dir_list = np.array(0)
    for dirs in os.listdir(path):
        dir_list = np.append(dir_list,dirs)
    dir_list = np.delete(dir_list,0)
    print(dir_list.shape)
    print(dir_list[1])
    save_neuron_name(dir_list)

#def take_apart_LM(path):
#    #读文本数据
#    f = open(path) 
#    ftextlist = f.readlines()
#    #删除注释行
#    data = np.zeros((0,len(ftextlist)))
#    name = np.array((0))
#    for s in ftextlist:
#        str_ = ''.join(s)
#        tem = str_.split('\t')[0].split('\\')
#        name = np.append(name,tem[len(tem)-1])
#    name = np.delete(name,0)
#
#    #取第3个和第5个数据
#    tem = np.zeros(0)
#    for s in ftextlist:
#        str_ = ''.join(s)
#        tem = np.append(tem,(int(str_.split('\t')[2])))
#    tem = tem.reshape(1,len(tem))
#    data = np.append(data,tem,axis=0)
#    
#    tem = np.zeros(0)
#    for s in ftextlist:
#        str_ = ''.join(s)
#        str_ = str_.split('\t')[4]
#        #去掉括号
#        str_ = str_.strip('(').strip(')')
#        tem = np.append(tem,int(str_))
#    tem = tem.reshape(1,len(tem))
#    data = np.append(data,tem,axis=0)    
#    f.close()
#    return name.reshape(len(name),1),data.T

def take_apart_LM(path,label_name):
    #读文本数据
    f = open(path) 
    ftextlist = f.readlines()
    #删除注释行
#    data = np.zeros((0,len(ftextlist)))
    name = np.array((0))
    for s in ftextlist:
        str_ = ''.join(s)
        tem = str_.split('\t')[0].split('\\')
        name = np.append(name,tem[len(tem)-1])
    name = np.delete(name,0)
    
    data1 = np.zeros(0)
    data2 = np.zeros(0)
    record=0
    for i in range(label_name.shape[0]):
        for j in range(record,len(name)):
            #找相同的名字序号
            if(name[j] == label_name[i]):
                record = j
                break
            if(j == len(name)):
                print("not find same LM name")

        s = ftextlist[record]
        str_ = ''.join(s)
        #取第3列数据
        data1 = np.append(data1,(int(str_.split('\t')[2])))    
        #取第5列数据
        str_2 = str_.split('\t')[4]
        #去掉括号
        str_2 = str_2.strip('(').strip(')')
        data2 = np.append(data2,int(str_2))
#    print(data1.shape,data2.shape)
    data1 =data1.reshape((len(data1),1))      
    data2 =data2.reshape((len(data2),1)) 
    data1 = np.append(data1,data2,axis=1)
    f.close()
#    print(data1.shape)
    return label_name,data1

def take_LM(path):
    #读文本数据
    f = open(path) 
    ftextlist = f.readlines()

    name = np.array((0))
    for s in ftextlist:
        str_ = ''.join(s)
        tem = str_.split('\t')[0].split('\\')
        name = np.append(name,tem[len(tem)-1])
    name = np.delete(name,0)

    data1 = np.zeros(0)
    data2 = np.zeros(0)

    for s in ftextlist:
        str_ = ''.join(s)
        #取第3列数据
        str_1 = str_.split('\t')
        data1 = np.append(data1,(int(str_1[2])))    
        #取第5列数据
        str_2 = str_.split('\t')[4]
        #去掉括号
        str_2 = str_2.strip('(').strip(')')
        data2 = np.append(data2,int(str_2))

#    print(data1.shape,data2.shape)
    data1 =data1.reshape((len(data1),1))      
    data2 =data2.reshape((len(data2),1)) 
    data1 = np.append(data1,data2,axis=1)
    f.close()
#    print(data1.shape)
    return name.reshape(len(name),1),data1    
if __name__ == '__main__':
    
    path = "F:\\004Vaa3d\\002Data\\001block_10\\method1_auto_many_block"
#    path = "F:\\004Vaa3d\\002Data\\001\\6.27\\L_measure\\manual_block_LM\\branch_manual_806.txt"
    read_dir_save(path)
#    name_lm,data_lm = take_apart_LM(path)
#    print(name.shape,data.shape)
    
    print(path)
    