import numpy as np
import os

def diff_label():
    # path = 'F:\\004Vaa3d\\Code_python\\machine_label\\method2_1to0_classification_label_1115_clf176.txt'
    path = 'F:\\004Vaa3d\\004feature\\all_samples_name\\all_samples_name.txt'

    # 读文本数据
    f = open(path)
    ftextlist = f.readlines()
    name = np.array(0)

    # #删除注释行
    # ftextlist.pop(0)
    num_label0 = 0
    num_label2 = 0

    old = -1
    for text in ftextlist:
        tem = text.split('\t')
        tem_name = tem[1]
        tem = tem[len(tem)-1]
        tem = int(tem.split('.')[0])
        if(tem == 1):
            tem = 2
        if(tem == 0):
            num_label0 += 1
            name = np.append(name, tem_name)
        elif(tem == 2 and old != 2):
            num_label2 +=1
            name = np.append(name, tem_name)
        old = tem

    # 删除第一行0
    name = np.delete(name, 0)

    return name, num_label0,num_label2

def diff_label_17545():
    # path = 'F:\\004Vaa3d\\Code_python\\machine_label\\method2_1to0_classification_label_1115_clf176.txt'
    path = 'F:\\004Vaa3d\\004feature\\17545\\tif_sample_name_17545.txt'

    # 读文本数据
    f = open(path)
    ftextlist = f.readlines()
    # name = np.array(0)

    # #删除注释行
    # ftextlist.pop(0)
    num_label0 = 0
    num_label2 = 0

    for text in ftextlist:
        tem = text.split('\t')

        tem = int(tem[len(tem)-1])

        if(tem == 0):
            num_label0 += 1
            # name = np.append(name, tem_name)
        else:
            num_label2 +=1
            # name = np.append(name, tem_name)

    print('label_0:',num_label0, 'label_2:',num_label2)
    # return num_label0,num_label2

def diff_seq_label(num):
    path = 'F:\\004Vaa3d\\004feature\\all_samples_name\\all_samples_name.txt'

    # 读文本数据
    f = open(path)
    ftextlist = f.readlines()
    # name = np.array(0)
    tar_name = np.empty(0)

    name_list = np.empty(0)
    # #删除注释行
    # ftextlist.pop(0)
    num_label0 = 0
    num_label2 = 0

    pre = 0
    back = 0

    old = -1
    count = 0
    for text in ftextlist:
        tem = text.split('\t')
        tem_name = tem[1]
        tem = tem[len(tem)-1]
        tem = int(tem.split('.')[0])
        name_list = np.append(name_list, tem_name)

        if(tem == 1):
            tem = 2

        if(tem == 0):
            num_label0 += 1
            tar_name = np.append(tar_name, tem_name)

        if(tem == 2 and old != 2):
            pre = count
        if((tem !=2 and old == 2) or count==len(ftextlist)-1 and tem == 2):

            if(tem !=2 and old == 2):
                back = count-1
            else:
                back = count
            if((back-pre) <= num):
                tar_name = np.append(tar_name, name_list[pre:(back+1)])
                num_label2 += back - pre + 1
            else:
                tar_name = np.append(tar_name, name_list[pre:(pre+1)])
                num_label2 += 1

        old = tem
        count += 1

    # # 删除第一行0
    # tar_name = np.delete(tar_name, 0)

    return tar_name, num_label0,num_label2


def save_spe_label2_name(data):
    save_path = 'F:\\004Vaa3d\\004feature\\Spe_incontinuity_samples\\incontinuity_seq_5_label2.txt'
    f = open(save_path, mode='w')
    count = 0
    for name in data:
        count += 1
        f.writelines([str(count),'\t',name,'\t',name.split('.')[0].split('l')[1],'\n'])

def main():
    # print('start...')
    # name,d1,d2 = diff_seq_label(5)
    # print(name[:10],d1,d2)
    # save_spe_label2_name(name)
    # print('end...')
    diff_label_17545()

if __name__=='__main__':
    main()



