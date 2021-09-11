# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 15:08:01 2020

@author: Administrator
"""

import numpy as np
from read_file_data_deal_feature import take_LM,take_apart_LM,read_distance_select,get_method2_label_name,get_method1_label_name,save_combination_lm_nd,read_tree_num_all    
import time
def read_17545_sample_name(path):
    f = open(path)
    ftextlist = f.readlines()
    re = np.array(0)
    for name in ftextlist:
        re = np.append(re,name)
    re = np.delete(re, 0)

    return re.reshape(len(re),1)

if __name__ == '__main__':
    starttime = time.time()

    path_lm_auto_method2 = "F:\\004Vaa3d\\004feature\\17545\LM\\LM_branch_method2_auto_norag_nofew_block_414.txt"
    path_lm_manual_method2 = "F:\\004Vaa3d\\004feature\\17545\LM\\LM_branch_method2_manual_norag_nofew_block_414.txt"
    
    path_nd_method2 ="F:\\004Vaa3d\\004feature\\17545\\Neuron_distance\\17545_dis_score_method2_4.14.txt"

    path_tree_num_method2 = "F:\\004Vaa3d\\004feature\\17545\\Tree_num\\neuron_tree_num_414.txt"
    
    method2_label_path = "F:\\004Vaa3d\\004feature\\17545\\17545_all_sample_name.txt"
#####################################################################
########           加载数据
####################################################################    
    print("load label name:")
    method2_label_name = read_17545_sample_name(method2_label_path)
    print(method2_label_name.shape)
#    print(method2_label_name[0],method2_label_name[1])
    
    #加载LM数据
    print("load LM data:")

    name_lm_auto_method2,data_lm_auto_method2 = take_LM(path_lm_auto_method2)
    name_lm_manual_method2,data_lm_manual_method2 = take_LM(path_lm_manual_method2)   

    print(name_lm_auto_method2.shape,data_lm_auto_method2.shape)
    print(name_lm_manual_method2.shape,data_lm_manual_method2.shape)    
#    #加载neuron_distance数据，其中[2,3,5,6,7]表示取2,3,5,6,7列数据
    print("load neuron_distance data:")
    name_nd_method2,data_nd_method2 = read_distance_select(path_nd_method2,[2,3,5,6,7],method2_label_name.shape[0])

    print(name_nd_method2.shape,data_nd_method2.shape)
    
    #加载block_tree_num
    print("load block_tree_num data:")

    name_tree_num_method2,data_tree_num_method2 = read_tree_num_all(path_tree_num_method2)

    print(name_tree_num_method2.shape,data_tree_num_method2.shape)

#####################################################################
########           预处理数据
####################################################################      
    #预处理LM数据,分枝数做差，神经元结点个数做差

    branch_minus2 = np.zeros((data_lm_auto_method2.shape[0],1))
    neuron_num_minus2 = np.zeros((data_lm_auto_method2.shape[0],1))
#    mag = 10
    for i in range(data_lm_auto_method2.shape[0]):
        branch_minus2[i][0] = abs(data_lm_auto_method2[i][0] - data_lm_manual_method2[i][0] )
        neuron_num_minus2[i][0] = abs(data_lm_auto_method2[i][1]+data_lm_auto_method2[i][0] - data_lm_manual_method2[i][1]- data_lm_manual_method2[i][0] )

    #预处理block_tree_num，树的个数差
    tree_num_minus_method2 = np.zeros((data_tree_num_method2.shape[0],1))
    for i in range(data_tree_num_method2.shape[0]):
        tree_num_minus_method2[i][0] = abs(data_tree_num_method2[i][1] - data_tree_num_method2[i][0] )
        
        
#####################################################################
########           融合数据
####################################################################          
    #融合lm和nd到一个数组
    data_method2 = np.append(data_nd_method2,branch_minus2,axis=1)
    data_method2 = np.append(data_method2,neuron_num_minus2,axis=1)
    data_method2 = np.append(data_method2,tree_num_minus_method2,axis=1)

    print(data_method2.shape)

#####################################################################
########           保存数据
####################################################################
    save_data2 = "F:\\004Vaa3d\\004feature\\17545\\17545_nd_5_lm_3_tn_method2_414.txt"
    #选择的特征
    note = ['###n','neuron_name','dist_12_allnodes','dist_12_allnodes','percent_21_apartnodes'
    ,'percent_21_apartnodes','percent_12_apartnodes','branch_minus','neuron_num_minus','block_tree_num_minus']

#    note = ['###n','neuron_name','dist_12_allnodes','dist_21_allnodes','dist_apartnodes','percent_12_apartnodes'
#        ,'percent_12_apartnodes','percent_21_apartnodes','branch_minus','neuron_num_minus','block_tree_num_minus']
    save_combination_lm_nd(save_data2,data_method2,name_nd_method2,note)
    
    endtime = time.time()
    print('总共的时间为:', (endtime - starttime),'secs')
#    print(name_nd_method1[1562][0])
#    print(name_nd_method2[1562][0])



