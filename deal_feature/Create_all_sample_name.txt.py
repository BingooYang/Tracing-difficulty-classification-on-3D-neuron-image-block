import numpy as np
import os
import sys
import  xlrd

def read_test_label(path):
    data = xlrd.open_workbook(path)
    # 打开sheet1
    table = data.sheet_by_index(0)
    method2_lable = np.zeros((0))
    name = np.array((0))

    for i in range(len(table.col_values(9))):
        # 前两行是注释，从第三行开始写入
        if (i > 1):
            if (table.col_values(9)[i] != 0 and table.col_values(9)[i] != 1 and table.col_values(9)[i] != 2 and
                    table.col_values(9)[i] != -1):
                break
            if (table.col_values(9)[i] >= 0):
                method2_lable = np.append(method2_lable, table.col_values(9)[i])
                name = np.append(name, table.col_values(1)[i])

    name = np.delete(name, 0)
    return name.reshape((len(name), 1)), method2_lable.reshape(len(method2_lable), 1)


def read_train_label(path):
    # 读文本数据
    f = open(path, encoding='GB2312')
    ftextlist = f.readlines()
    # 删除注释行
    ftextlist.pop(0)
    label_name = np.array(0)
    label_data = np.zeros(0)
    for s in ftextlist:
        str = ''.join(s)
        label_name = np.append(label_name, (str.split('\t')[1]))
    label_name = np.delete(label_name, 0)
    for s in ftextlist:
        str = ''.join(s)
        label_data = np.append(label_data, (int(str.split('\t')[2][0])))
    #        data_x = data_x.reshape(len(data_x),1)

    return label_name.reshape((len(label_name), 1)), label_data.reshape((len(label_data), 1))

def all_sample_no_label():
    path_good_block = 'F:/004Vaa3d/002Data/001block_10/method2_manual_many_block'
    path_bad_block = 'F:/004Vaa3d/002Data/001block_10/method2_manual_few_block'
    list_good_block = os.listdir(path_good_block)
    list_bad_block = os.listdir(path_bad_block)
    list_block = list_good_block + list_bad_block
    list_block.sort()

    save_path = 'F:\\004Vaa3d\\004feature\\all_samples_name\\all_samples_swc_name.txt'
    f = open(save_path, mode='w')

    for i in range(len(list_block)):
        f.writelines([str(i),'\t',list_block[i],'\t','\n'])

def all_sample_have_label():
    ##加载标签数据
    path_train_label = "F:/004Vaa3d/Code_python/machine_label/method2_1to0_classification_label_1115_clf176.txt"
    path_test_label = "F:/004Vaa3d/003label/001_latest_label_1113.xlsx"

    label_train_name,label_train_data = read_train_label(path_train_label)
    label_test_name,label_test_data = read_test_label(path_test_label)

    print(label_train_data.shape,label_test_data.shape)
    #除去人工标签部分
    label_train_name = label_train_name[label_test_name.shape[0]:]
    label_all_name = np.concatenate((label_test_name,label_train_name),axis=0)
    label_train_data = label_train_data[label_test_data.shape[0]:]
    label_all_data = np.concatenate((label_test_data,label_train_data),axis=0)

    #加载swc文件名列表
    path_good_block = 'F:/004Vaa3d/002Data/001block_10/method2_manual_many_block'
    path_bad_block = 'F:/004Vaa3d/002Data/001block_10/method2_manual_few_block'
    list_good_block = os.listdir(path_good_block)
    list_good_block.sort()
    for i in range(len(list_good_block)):
        for j in range(len(label_all_name)):
            if(list_good_block[i] == label_all_name[j]):
                list_good_block[i] = list_good_block[i].split('.')[0] + '_l' + str(label_all_data[j][0]) + '.swc'

    list_bad_block = os.listdir(path_bad_block)
    for i in range(len(list_bad_block)):
        list_bad_block[i] = list_bad_block[i].split('.')[0] + '_l3.0.swc'
    list_block = list_good_block + list_bad_block
    list_block.sort()

    ##保存文件名列表
    save_path = 'F:\\004Vaa3d\\004feature\\all_samples_name\\all_samples_swc_name_have_label_1.29.txt'
    f = open(save_path, mode='w')

    for i in range(len(list_block)):
        f.writelines([str(i),'\t',list_block[i],'\t','\n'])

def create_filelist_name():
    path = 'F:\\004Vaa3d\\002Data\\17545_delete_rag_few_seg\\method2_manual_norag_nofew_block\\'
    file_list = os.listdir(path)
    # file_list = file_list.sort()

    save_path = 'F:\\004Vaa3d\\004feature\\17545\\all_sample_name.txt'
    f = open(save_path,mode='w')
    for name in file_list:
        f.write(str(name)+'\n')

def main():
    print('start...')
    # create_filelist_name()
    all_sample_have_label()
    print('end...')

if __name__=='__main__':
    main()

