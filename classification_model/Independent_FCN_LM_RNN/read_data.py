import numpy as np
import torch.utils.data
import os

def get_label(na,label):
    # label = np.zeros((len(na)))
    for i in range(len(na)):
        # print(na[i],i,na[i].split('.')[0].split('l')[1])
        label[i] = int(na[i].split('.')[0].split('l')[1])
        if(label[i]==0):
            label[i] = 0
        else:
            label[i] = 1
    return label

# def read_rnn_seq_name_pair(path, seq):
#     # 读文本数据
#     f = open(path,encoding='gbk')
#     ftextlist = f.readlines()
#
#     #删除注释行
#     ftextlist.pop(0)
#     name_list = []
#
#     label_list = []
#     for text in ftextlist:
#         tem_name = []
#         for i in range(seq):
#             name = text.split('\t')[i+1]
#             tem_name.append(name)
#
#         label = int(tem_name[seq-1].split('.')[0].split('l')[1])
#
#         if(label==0):
#             label = 0
#         else:
#             label = 1
#         name_list.append(tuple(tem_name))
#         label_list.append(label)
#     # print(len(name_list),len(label_list))
#     return  name_list,label_list

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

def get_data(lm_dic, name_tuple_list, seq):
    feature_len = 0
    for i,j in lm_dic.items():
        feature_len = len(j)
        break
    data = np.zeros((len(name_tuple_list),seq,feature_len))
    for i in range(len(name_tuple_list)):
        tem_tuple = name_tuple_list[i]
        for j in range(seq):
            tem = lm_dic[tem_tuple[j].split('r')[0]]
            data[i,j,:] = tem
        # if(i == 0):
        #     tem_tuple = name_tuple_list[i]
        #     for j in range(seq):
        #         if(j == 0):
        #             tem = lm_dic[tem_tuple[j]]
        #             data = np.expand_dims(tem,axis=0)
        #         else:
        #             tem = lm_dic[tem_tuple[j]]
        #             tem = np.expand_dims(tem,axis=0)
        #             data = np.concatenate((data,tem),axis=0)
        #     re = data
        # else:
        #     tem_tuple = name_tuple_list[i]
        #     for j in range(seq):
        #         if(j == 0):
        #             tem = lm_dic[tem_tuple[j]]
        #             data = np.expand_dims(tem,axis=0)
        #         else:
        #             tem = lm_dic[tem_tuple[j]]
        #             tem = np.expand_dims(tem,axis=0)
        #             data = np.concatenate((data,tem),axis=0)
    return data

class GetSample(torch.utils.data.Dataset):
    def __init__(self,
                 name_pair='',
                 lm_dic =None,
                 mode = 'train',
                 seq = 2):
        self.seq = seq
        self.name_tuple,self.label_list = read_rnn_seq_name_pair(name_pair,self.seq)
        if(mode == 'train'):
            self.name_tuple = self.name_tuple[5342:]
            self.label_list = self.label_list[5342:]
        else:
            self.name_tuple = self.name_tuple[:5342]
            self.label_list = self.label_list[:5342]
        self.data = get_data(lm_dic,self.name_tuple,seq)

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):

        label = self.label_list[index]
        data = self.data[index]
        name = self.name_tuple[index]

        label = np.array(label)
        return (name, data, label)


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