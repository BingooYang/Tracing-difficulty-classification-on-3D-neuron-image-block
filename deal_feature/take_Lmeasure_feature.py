# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 15:28:50 2021

@author: Bingoo
"""
import numpy as np
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
#    #删除注释行
#    ftextlist.pop(0)
#    print(len(ftextlist))
    data = np.zeros((int(len(ftextlist)/f_num),f_num))
#    print(data.shape)
    name = np.array(0)    

    count = 0
    for s in ftextlist:
        sr = ''.join(s)
#        print(s)
        if(count%f_num==0):
            name = np.append(name,(sr.split('\t')[0].split('\\')[6]))
        data[int(count/f_num)][count%f_num] = float(sr.split('\t')[2])
#        print(float(sr.split('\t')[2]))
        count += 1
        

    #删除第一行0
    name = np.delete(name,0)        
    return name,data


def test():
    name,data = read_nd_lm('F:\\004Vaa3d\\004feature\\LM\\method2_auto_norag_nofew_block_Lmeasure_0104.txt',7)
    print(name.shape,data.shape)
if __name__ == '__main__':
    print('start...')
    starttime = time.time()
    test()
    endtime = time.time()
    print('总共的时间为:', (endtime - starttime),'secs')
    
    
    
    