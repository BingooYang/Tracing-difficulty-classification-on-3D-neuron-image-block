# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 13:23:40 2020

@author: Administrator
"""


import itertools
import numpy as np

def save_(path,data):

    print("保存文件的路径：",path)
    f = open(path,mode='w')
    f.write('#### 序号    组合 \n')
    for i in range(len(data)):
        f.writelines([str(i),'\t',str(data[i]),'\t','\n'])
    f.close()

if __name__ == '__main__':
    path = "F:\\004Vaa3d\\Code_python\\deal_feature\\featrre_combination_8_904.txt"
    #生成组合序列，总共127种
    #    list1 = [0,1,2,3,4,5,6]
    list1 = [0,1,2,3,4,5,6,7]
    list2 = []
    for i in range(1,len(list1)+1):
    #        print("i:",i)
        iter1 = itertools.combinations(list1,i)
        for j in iter1:
            list2.append(list(j))
        
    save_(path,list2)
    
    
    