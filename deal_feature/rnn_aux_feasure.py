# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 09:43:38 2021

@author: Bingoo
"""

import numpy as np
# from read_file_data_deal_feature import take_LM
import time


def take_LM(path):
    # 读文本数据
    f = open(path)
    ftextlist = f.readlines()

    name = np.array((0))
    for s in ftextlist:
        str_ = ''.join(s)
        tem = str_.split('\t')[0].split('\\')
        name = np.append(name, tem[len(tem) - 1])
    name.pop(0)

    data1 = np.zeros(0)
    data2 = np.zeros(0)

    for s in ftextlist:
        str_ = ''.join(s)
        # 取第3列数据
        str_1 = str_.split('\t')
        data1 = np.append(data1, (int(str_1[2])))
        # 取第5列数据
        str_2 = str_.split('\t')[4]
        # 去掉括号
        str_2 = str_2.strip('(').strip(')')
        data2 = np.append(data2, int(str_2))

    #    print(data1.shape,data2.shape)
    data1 = data1.reshape((len(data1), 1))
    data2 = data2.reshape((len(data2), 1))
    data1 = np.append(data1, data2, axis=1)
    f.close()
    #    print(data1.shape)
    return name.reshape(len(name), 1), data1

def get_feature():
    path_lm_auto_method2 = "F:\\004Vaa3d\\004feature\\LM\\branch_method2_auto_norag_nofew_block_827.txt"
    
    name_lm_auto_method2,data_lm_auto_method2 = take_LM(path_lm_auto_method2)
    
    print('dd')

if __name__ == '__main__':
    
    get_feature()


