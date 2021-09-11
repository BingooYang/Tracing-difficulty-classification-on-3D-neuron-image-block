import os
import tifffile
import numpy as np
import torch
import torch.utils.data as da

# from torchvision import datasets, transforms
# import random


def dynamic_binary_value_cuda_2_one(data,zoom1,zoom2):
    data_tem = np.copy(data)
    data_tem = torch.from_numpy(data_tem)
    num = torch.tensor(64*64*32)
    # ave = torch.tensor(data.shape[0])
    #获得每一个样本灰度值的平均值×zoom
    tem = torch.flatten(data_tem)
    tem = torch.sum(tem)

    # print(tem.shape)
    zoom1 = torch.tensor(zoom1, dtype=torch.float)
    ave = tem/num*zoom1
    ave = np.array(ave,dtype='uint8')
    ave = torch.from_numpy(ave)
    kk = torch.zeros_like(data_tem)

    data_tem = torch.where(data_tem < ave, kk, data_tem)

    sum_num = torch.sum(data_tem>ave)

    #获得每一个样本灰度值的平均值×zoom
    tem = torch.flatten(data_tem)
    tem = torch.sum(tem)

    zoom2 = torch.tensor(zoom2,dtype=torch.float)
    ave2 = tem/sum_num *zoom2
    ave2 = np.array(ave2,dtype='uint8')
    ave2 = torch.from_numpy(ave2)

    # data_tem = torch.where(data_tem < ave, kk, data_tem)

    # data_tem = data_tem.numpy()

    return ave2

def omit_background(data_ori, threshold):

    for i in range(data_ori.shape[0]):
        data_ori[i] = np.where(data_ori[i] < threshold[i], 0, data_ori[i])

    return data_ori


# dir structure would be: data/class_name(0 and 1)/dir_containing_img/img
class GetSample(da.Dataset):
    def __init__(self, root = '/disk1/001yangbin/002vaa3d_img_samples/data agumentation/outimg_good_block_agumentation_0/', mode = 'train', transform = None):
        self.root = root
        self.mode = mode
        # self.raw_samples = glob.glob(root+ '*')
        self.raw_samples = os.listdir(root)
        # 排序
        self.raw_samples.sort(key=lambda x: int(x[0:8]))
        # print(self.raw_samples)
        self.samples = []
        for raw_sample in self.raw_samples:
            self.samples.append((raw_sample, int(raw_sample.split('l')[1].split('.')[0])))
            # print(raw_sample,int(raw_sample.split('/')[-2]))
        # print(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        # if self.mode == 'train':
        #     random.shuffle(self.samples)
        # print(index)
        image_name, label = self.samples[index]

        image_dir = self.root + image_name

        image = tifffile.imread(image_dir)
        # threshold_ave = dynamic_binary_value_cuda_2_one(image, 1.0, 0.7)
        # image = omit_background(image, threshold_ave)
        # top_k = 30
        # image = segmentation_block_order(image, threshold_ave, top_k)

        if(label != 0):
            label = 1

        return (image, label,image_name)

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