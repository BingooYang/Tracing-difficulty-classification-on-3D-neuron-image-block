# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 15:28:50 2021

@author: Bingoo
"""
import numpy as np
from sklearn.preprocessing import normalize
import os
import time
"""
path: L-measure文件路径(txt)
f_num: 特征个数

"""
def read_nd_lm(path,f_num):
    #读文本数据
    f = open(path)
    ftextlist = f.readlines()

    tem_data = np.zeros(f_num)

    dic = {}
    count = 0
    for s in ftextlist:
        sr = ''.join(s)
        if(count%f_num !=0 ):
            tem_data[count%f_num] = float(sr.split('\t')[2])
        else:
            tem_data = np.zeros(f_num)
            tem_data[0] = float(sr.split('\t')[2])

        if((count+1)%f_num==0):
            tem = sr.split('\t')[0].split('\\')
            # print(tem[len(tem)-1].split('l')[0])
            dic[tem[len(tem)-1].split('l')[0]] = tem_data

        count += 1

    return dic

def read_nd_lm_normalization(path,f_num):
    #读文本数据
    f = open(path)
    ftextlist = f.readlines()

    tem_data = np.zeros(f_num)

    dic = {}
    count = 0
    for s in ftextlist:
        sr = ''.join(s)
        if(count%f_num !=0 ):
            tem_data[count%f_num] = float(sr.split('\t')[2])
        else:
            tem_data = np.zeros(f_num)
            tem_data[0] = float(sr.split('\t')[2])

        if((count+1)%f_num==0):
            tem = sr.split('\t')[0].split('\\')
            # print(tem[len(tem)-1].split('l')[0])
            dic[tem[len(tem)-1].split('l')[0]] = tem_data

        count += 1

    my_list = [elem for elem in dic.values()]
    my_list = normalize(my_list, axis=0, norm='max')
    for i,name in enumerate(dic):
        dic[name] = my_list[i]

    return dic

def test():
    dic = read_nd_lm('/home/zhang/disk2/001yangbin/001vaa3d/003_label_result/method2_auto_norag_nofew_block_nosoma_Lmeasure_label_0109.txt',32)
    print(dic['001_0000_x_16202_y_15473_z_3047_'])


if __name__ == '__main__':
    print('start...')
    starttime = time.time()
    test()
    endtime = time.time()
    print('总共的时间为:', (endtime - starttime),'secs')
    
    
    
    