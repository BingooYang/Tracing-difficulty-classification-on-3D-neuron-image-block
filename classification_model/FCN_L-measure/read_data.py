import os
import tifffile
import numpy as np
import torch
import torch.utils.data

from take_Lmeasure_feature import read_nd_lm_test,read_nd_lm_train

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

# dir structure would be: data/class_name(0 and 1)/dir_containing_img/img
class GetSample(torch.utils.data.Dataset):
    def __init__(self, root = '/home/zhang/disk2/001yangbin/001vaa3d/003_label_result/method2_auto_norag_nofew_block_Lmeasure_label_0104.txt', mode = 'train', transform = None):

        if(mode=='train'):
            self.name, self.data = read_nd_lm_train(root, 7)
        else:
            self.name, self.data = read_nd_lm_test(root, 7)

        self.label = get_label(self.name)

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):

        return (self.data[index], self.label[index])


class GetSample_data(torch.utils.data.Dataset):
    def __init__(self,
                 dataset,
                 name,
                 mode='train', transform=None):

        self.name = name
        self.data = dataset
        self.label = np.zeros((len(self.name)))

        self.label = get_label(self.name,self.label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):

        return (self.name[index],self.data[index], self.label[index])


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