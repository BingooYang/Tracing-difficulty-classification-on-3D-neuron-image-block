import numpy as np
import os

def all_sample_txt():
    path = 'F:\\004Vaa3d\\002Data\\001block_10\\001no_rag_few_seg\\method2_auto_norag_nofew_block_swc_104\\'
    save_path = 'F:\\004Vaa3d\\004feature\\all_samples_name\\all_samples_name.txt'
    file_list = os.listdir(path)

    f = open(save_path, mode='w')
    count = 0
    for name in file_list:
        count += 1
        f.writelines([str(count),'\t',name,'\t',name.split('.')[0].split('l')[1],'\n'])


def read_name(path):
    # 读文本数据
    f = open(path)
    ftextlist = f.readlines()
    name = np.array(0)

    count = 0
    for text in ftextlist:
        if(text.split('\t')[0]!='1'):
            count += 1
        else:
            break
    #删除非数据行
    del ftextlist[0:count]
    for text in ftextlist:
        name = np.append(name, text.split('\t')[1].split('z')[0])

    # 删除第一行0
    name = np.delete(name, 0)
    return name

def sta_label_true():
    path = 'F:\\004Vaa3d\\001note\\错误样本分析\\rnn_resnet\\para_2021-01-21 13_05_08_end.txt'
    # 读文本数据
    f = open(path)
    ftextlist = f.readlines()

    true_num = 0
    count = 0
    for text in ftextlist:
        label = int(text.split('\t')[1].split('.')[0].split('l')[1])
        if(label==0):
            true_num += 1
    print(true_num)

def read_name_tree_spe(path,tree_num=1):
    # 读文本数据
    f = open(path)
    ftextlist = f.readlines()
    name = np.array(0)

    count = 0
    for text in ftextlist:
        if(text.split('\t')[0]!='1'):
            count += 1
        else:
            break
    #删除非数据行
    del ftextlist[0:count]
    for text in ftextlist:
        tem = text.split('\t')
        tem = tem[len(tem)-1].split('\n')[0]
        if(int(tem) == tree_num):
            name = np.append(name, text.split('\t')[1].split('z')[0])

    # 删除第一行0
    name = np.delete(name, 0)
    return name

def common_samples(name1,name2):
    count = 0
    for n1 in name1:
        for n2 in name2:
            if(n1 == n2):
                count+=1
    return count

def sta_error_samples():
    file1 = 'F:/004Vaa3d/001note/错误样本分析/resnet_fcn/para_2020-12-21 21_23_42_end.txt'
    file2 = 'F:\\004Vaa3d\\001note\\错误样本分析\\rnn_resnet\\para_2021-01-21 13_05_08_end.txt'

    file3 = 'F:/004Vaa3d/004feature/all_samples_name/all_samples_name.txt'
    file4 = 'F:\\004Vaa3d\\001note\\错误样本分析\\resnet_fcn\\001stem_para_2021-01-09 15_42_56_end.txt'

    all_sample_name = read_name(file3)
    test_sample_name = all_sample_name[:2000]

    file1_name = read_name(file1)
    file2_name = read_name(file2)
    file4_name = read_name(file4)
    # tree_num_name = read_name_tree_spe(file2,tree_num=2)
    num = common_samples(file4_name,file2_name)
    print('file1_name:',file4_name.shape,'file2_name:',file2_name.shape)
    print('common:',num)


def main():
    print('start...')
    # sta_error_samples()
    sta_label_true()
    print('end...')

if __name__=='__main__':
    main()

