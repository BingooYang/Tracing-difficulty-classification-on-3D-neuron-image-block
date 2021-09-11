import os
import tifffile
import numpy as np
# import torch
import torch.utils.data as da
from sklearn.preprocessing import normalize
# from torchvision import datasets, transforms
# import random

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

def read_rnn_seq_name_pair(path, seq):
    # 读文本数据
    f = open(path,encoding='gbk')
    ftextlist = f.readlines()

    #删除注释行
    ftextlist.pop(0)
    name_list = []

    label_list = []
    for text in ftextlist:
        tem_name = []
        label = []
        for i in range(seq):
            name = text.split('\t')[i+1]
            tem_name.append(name)

            tem_label = int(name.split('.')[0].split('l')[1])
            if(tem_label==0):
                tem_label = 0
            else:
                tem_label = 1
            label.append(tem_label)

        name_list.append(tuple(tem_name))
        label_list.append(label)
    # print(len(name_list),len(label_list))
    return  name_list,label_list

def get_lm_seq_data(lm_dic, name_tuple, lm_feature_num, tar_name):

    seq = len(name_tuple[0])
    tem_name_tuple = np.array(name_tuple)
    tem = tem_name_tuple[:,seq-1]

    pos = np.where(tem == tar_name)

    re = np.zeros((seq,lm_feature_num))

    for i in range(seq):
        name = name_tuple[pos[0][0]][i]
        re[i] = lm_dic[name.split('r')[0]]

    return re,pos


def app2_omit_background(data_ori):
    data_tem = np.copy(data_ori)
    # num = data_tem.shape[0]
    data_tem = data_tem.reshape(-1,1)
    ave = np.mean(data_tem)
    std = np.std(data_tem)
    if(std<10):
        std = 10

    threshold = ave + 0.7*std

    data_ori = np.where(data_ori >= threshold,data_ori,0)

    return data_ori

class GetSample(da.Dataset):
    def __init__(self, root = '',
                 lm_path='',
                 lm_feature_num=None,
                 mode = 'train',
                 background=True,
                 name_pair='',
                 seq=2):

        self.background = background
        self.root = root
        self.mode = mode

        self.seq = seq
        self.lm_feature_num = lm_feature_num
        self.name_tuple,self.label_list = read_rnn_seq_name_pair(name_pair,self.seq)
        self.lm_dic = read_nd_lm_normalization(lm_path, lm_feature_num)

        self.raw_samples = os.listdir(root)
        # 排序
        self.raw_samples.sort(key=lambda x: int(x[0:8]))
        self.samples = []
        for i in range(len(self.label_list)):
            if self.name_tuple[i][self.seq-1] in self.raw_samples:
                self.samples.append((self.name_tuple[i], self.label_list[i]))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):

        image_name, label = self.samples[index]
        image_dir = self.root + image_name[self.seq-1]

        lm_data,pos = get_lm_seq_data(self.lm_dic, self.name_tuple, self.lm_feature_num, image_name[self.seq-1])

        image = tifffile.imread(image_dir)
        # if(self.background):
        #     ##app2去背景
        #     image = app2_omit_background(image)

        label = np.array(label)
        return (image,lm_data[self.seq-1], label,image_name)

# ###test
# data_path_train = "/disk1/001yangbin/002vaa3d_img_samples/data agumentation/outimg_good_block_agumentation_0/"
# data_path_test = "/disk1/001yangbin/002vaa3d_img_samples/data agumentation/test_image/"
#
# # data = PatchMethod(root = data_path_train)
# val_data =PatchMethod(root = data_path_test, mode = 'test')
# # train_loader = torch.utils.data.DataLoader(data, shuffle = True, num_workers = 6, batch_size = 1)
# test_loader = torch.utils.data.DataLoader(val_data, shuffle = False, num_workers = 6, batch_size = 1)
#
# import time
# start_time = time.time()
# count = 0
# for batch_idx, (train_data, train_label) in enumerate(test_loader):
#
#     bag_label = train_label[0]
#     count +=1
# print("spend time:",time.time()-start_time, count)