import os
import tifffile
import numpy as np
import torch
# from torchvision import datasets, transforms
import torch.utils.data as da


def dynamic_binary_value_cuda_2_one(data, zoom1, zoom2):
    data_tem = np.copy(data)
    data_tem = torch.from_numpy(data_tem)
    num = torch.tensor(64 * 64 * 32)
    # ave = torch.tensor(data.shape[0])
    # 获得每一个样本灰度值的平均值×zoom
    tem = torch.flatten(data_tem)
    tem = torch.sum(tem)

    # print(tem.shape)
    zoom1 = torch.tensor(zoom1, dtype=torch.float)
    ave = tem / num * zoom1
    ave = np.array(ave, dtype='uint8')
    ave = torch.from_numpy(ave)
    kk = torch.zeros_like(data_tem)

    data_tem = torch.where(data_tem < ave, kk, data_tem)

    sum_num = torch.sum(data_tem > ave)

    # 获得每一个样本灰度值的平均值×zoom
    tem = torch.flatten(data_tem)
    tem = torch.sum(tem)

    zoom2 = torch.tensor(zoom2, dtype=torch.float)
    ave2 = tem / sum_num * zoom2
    ave2 = np.array(ave2, dtype='uint8')
    ave2 = torch.from_numpy(ave2)

    # data_tem = torch.where(data_tem < ave, kk, data_tem)

    # data_tem = data_tem.numpy()

    return ave2


def omit_background(data_ori, threshold):
    for i in range(data_ori.shape[0]):
        data_ori[i] = np.where(data_ori[i] < threshold[i], 0, data_ori[i])

    return data_ori


def app2_omit_background(data_ori):
    data_tem = np.copy(data_ori)
    # num = data_tem.shape[0]
    data_tem = data_tem.reshape(-1, 1)
    ave = np.mean(data_tem)
    std = np.std(data_tem)
    if (std < 10):
        std = 10

    threshold = ave + 0.7 * std

    data_ori = np.where(data_ori >= threshold, data_ori, 0)

    return data_ori

def read_rnn_name_pair(path):
    # 读文本数据
    f = open(path,encoding='gbk')
    ftextlist = f.readlines()

    #删除注释行
    ftextlist.pop(0)
    name_list = []
    label_list = []
    for text in ftextlist:
        name1 = text.split('\t')[1]
        tem   = name1.split('l')
        name1 = tem[0] + 'r_0_l' + tem[1].split('s')[0] + 'tif'

        name2 = text.split('\t')[2]
        tem   = name2.split('l')
        name2 = tem[0] + 'r_0_l' + tem[1].split('s')[0] + 'tif'

        label = text.split('\t')[3]
        label = int(label[0])
        if(label<=1):
            label = 0
        else:
            label = 1
        name_list.append((name1,name2))
        label_list.append(label)

    return  name_list,label_list

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

class GetSample(da.Dataset):
    def __init__(self,
                 root='',
                 name_pair='',
                 mode='train',
                 seq = 3,
                 background=True):
        self.seq = seq

        self.background = background
        self.root = root
        self.mode = mode
        self.name_tuple,self.label_list = read_rnn_seq_name_pair(name_pair,self.seq)

        self.raw_samples = os.listdir(root)
        # 排序
        self.raw_samples.sort(key=lambda x: int(x[0:8]))
        self.raw_samples = set(self.raw_samples)

        self.samples = []
        for i in range(len(self.label_list)):
            if self.name_tuple[i][self.seq-1] in self.raw_samples:
                self.samples.append((self.name_tuple[i], self.label_list[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # if self.mode == 'train':
        #     random.shuffle(self.samples)
        # print(index)
        name, label = self.samples[index]
        image_name = name[self.seq-1]

        image_dir = self.root + image_name

        image = tifffile.imread(image_dir)

        # for i in range(self.seq):
        #     image_dir = self.root + name[i]
        #     if(i==0):
        #         image = tifffile.imread(image_dir)
        #         # image = app2_omit_background(image)
        #         image = np.expand_dims(image,0)
        #     else:
        #         # image =
        #         tem = tifffile.imread(image_dir)
        #         # tem = app2_omit_background(tem)
        #         tem = np.expand_dims(tem,0)
        #         image = np.concatenate((image, tem), axis=0)
                
        label = np.array(label)
        return (image, label,name)

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