# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 16:23:23 2020

@author: Administrator
"""
import  os
import  numpy as np
import time

import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.optim as optim

from tensorboardX import SummaryWriter

from mobilenetv2 import mobilenetv2
from read_data import GetSample
from save_para import SavePara

# 定义是否使用GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def comculate_f1(label, predict):
    num = len(label)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    # label=0 easy blocks---> positive sample
    for i in range(num):
        if(label[i] == 1 and predict[i] == 1):
            TP += 1
        elif(label[i] == 1 and predict[i] == 0):
            FP += 1
        elif(label[i] == 0 and predict[i] == 1):
            FN += 1
        else:
            TN += 1
    return TP,FP,FN,TN

def mobilenet():

    starttime = time.time()
    model_lr = 0.0001
    model_betas = (0.9, 0.999)
    model_weight_decay = 1e-2
    model_batch_size = 30
    model_num_workers = 5
    epoch = 30
    ave_num = 5
    # dropout = 0.2

    debug = False
    print('start mobilev2')

    data_path_train = "/home/zhang/disk2/001yangbin/002vaa3d_img_samples/data_agmentation/original_image_train_23/"
    data_path_test = "/home/zhang/disk2/001yangbin/002vaa3d_img_samples/data_agmentation/original_image_test_23/"

    data = GetSample(root=data_path_train)
    val_data = GetSample(root=data_path_test, mode='test')
    print('train shape:',data.__len__(),'test shape:', val_data.__len__())
    train_loader = torch.utils.data.DataLoader(data, shuffle=True, num_workers=model_num_workers, batch_size=model_batch_size)
    test_loader = torch.utils.data.DataLoader(val_data, shuffle=False, num_workers=model_num_workers, batch_size=model_batch_size)

    if(not debug):
        ###保存参数
        save_para_path = os.getcwd() + '/para/'
        save_para = SavePara(save_para_path)
        save_para.model_scalar('model_lr',model_lr)
        save_para.model_scalar('model_betas', model_betas)
        save_para.model_scalar('model_weight_decay', model_weight_decay)
        save_para.model_scalar('model_batch_size', model_batch_size)
        save_para.model_scalar('model_num_workers', model_num_workers)
        save_para.model_scalar('epoch', epoch)
        save_para.model_scalar('train path', data_path_train)
        save_para.model_scalar('test path', data_path_test)

    print('start train...')

    train_acc = np.zeros(ave_num)
    test_acc = np.zeros(ave_num)
    train_f1 = np.zeros(ave_num)
    test_f1 = np.zeros(ave_num)

    for n in range(ave_num):

        model = mobilenetv2().to(device)
        optimizer = optim.Adam(model.parameters(), lr=model_lr, betas=model_betas, weight_decay=model_weight_decay)
        criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上

        correct_train = 0
        total_train = 0
        correct_test = 0
        total_test = 0

        train_TP = 0
        train_FP = 0
        train_FN = 0
        train_TN = 0
        test_TP = 0
        test_FP = 0
        test_FN = 0
        test_TN = 0

        for e in range(epoch):
            model.train()
            sum_loss = 0.0

            if (not debug):
                save_para.writelines_para(['epoch:',str(e)])
                save_para.writelines_para(['n', 'neuron_name', 'true label', 'predict label'])
                save_para.clear_count_num()

            for batch_idx, (train_data, bag_label, image_name) in enumerate(train_loader):
                train_data, bag_label = train_data.to(device), bag_label.to(device)
                train_data, bag_label = Variable(train_data), Variable(bag_label)

                train_data = torch.squeeze(train_data)
                train_data = torch.unsqueeze(train_data, 1)
                train_data = train_data.float()
                optimizer.zero_grad()
                outputs = model(train_data)

                _, predicted = torch.max(outputs.data, 1)

                # if (not debug):
                #     ###　保存错误样本标签
                #     save_para.predict_para(image_name,bag_label.cpu().numpy(),predicted.cpu().numpy())

                total_train += bag_label.size(0)
                correct_train += (predicted == bag_label).sum()

                TP, FP, FN, TN = comculate_f1(label=bag_label, predict=predicted)
                train_TP += TP
                train_FP += FP
                train_FN += FN
                train_TN += TN

                loss = criterion(outputs, bag_label)
                loss.backward()
                optimizer.step()

                sum_loss += loss.item()

            model.eval()
            test_sum_loss = 0
            for batch_idx, (test_data, bag_label,image_name) in enumerate(test_loader):
                test_data, bag_label = test_data.to(device), bag_label.to(device)
                test_data, bag_label = Variable(test_data), Variable(bag_label)
                test_data = torch.squeeze(test_data)
                test_data = torch.unsqueeze(test_data, 1)
                test_data = test_data.float()

                outputs = model(test_data)
                # loss = criterion(outputs, bag_label)
                test_sum_loss += loss
                # 取得分最高的那个类
                _, predicted = torch.max(outputs.data, 1)

                if (not debug):
                    ###　保存错误样本标签
                    save_para.predict_para(image_name,bag_label.cpu().numpy(),predicted.cpu().numpy())

                total_test += bag_label.size(0)
                correct_test += (predicted == bag_label).sum()

                TP, FP, FN, TN = comculate_f1(label=bag_label, predict=predicted)
                test_TP += TP
                test_FP += FP
                test_FN += FN
                test_TN += TN

        train_acc[n] = (100 * float(correct_train) / float(total_train))
        test_acc[n] = (100 * float(correct_test) / float(total_test))
        # train_loss = sum_loss / (len(train_loader))
        print('num:%d train acc:%.03f %% ' % (n + 1, train_acc[n]))
        print('test acc：%.03f %% ' % (test_acc[n]))

        # f1
        train_f1[n] = 100 * (2*train_TP)/(total_train+train_TP-train_TN)
        test_f1[n] = 100 * (2 * test_TP) / (total_test + test_TP - test_TN)
        print('train---TP:%d FP:%d  FN:%d TN:%d' % (train_TP, train_FP, train_FN, train_TN))
        print('test---TP:%d FP:%d  FN:%d TN:%d' % (test_TP, test_FP, test_FN, test_TN))
        print('train f1:%.03f %%' % (train_f1[n]))
        print('test f1:%.03f %%' % (test_f1[n]))

    if(not debug):
        save_para.model_scalar('############Experiment num:', n+1)
        save_para.model_scalar('tarin acc', train_acc)
        save_para.model_scalar('train f1', train_f1)
        save_para.model_scalar('TP, FP, FN, TN=', [train_TP, train_FP, train_FN, train_TN])
        save_para.model_scalar('test acc', test_acc)
        save_para.model_scalar('test f1', test_f1)
        save_para.model_scalar('TP, FP, FN, TN=', [test_TP, test_FP, test_FN, test_TN])

    train_acc_mean = np.mean(train_acc)
    train_acc_std = np.std(train_acc)
    test_acc_mean = np.mean(test_acc)
    test_acc_std = np.std(test_acc)

    train_f1_mean = np.mean(train_f1)
    train_f1_std = np.std(train_f1)
    test_f1_mean = np.mean(test_f1)
    test_f1_std = np.std(test_f1)

    endtime = time.time()
    print("总共费时：", (endtime - starttime), "secs")

    if (not debug):
        save_para.model_scalar('#########################', 'end####################')
        save_para.model_scalar('train_acc_mean', train_acc_mean)
        save_para.model_scalar('train_acc_std', train_acc_std)
        save_para.model_scalar('test_acc_mean', test_acc_mean)
        save_para.model_scalar('test_acc_std', test_acc_std)
        save_para.model_scalar('train_f1_mean', train_f1_mean)
        save_para.model_scalar('train_f1_std', train_f1_std)
        save_para.model_scalar('test_f1_mean', test_f1_mean)
        save_para.model_scalar('test_f1_std', test_f1_std)
        save_para.model_scalar('总共费时：', (endtime - starttime))

    # model_save_path = os.getcwd() + '/model/' + 'Resnet_layers_num_' + str(model_layers_num) + '_' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '.pth'
    # torch.save(model,model_save_path)

    model_save_path = os.getcwd() + '/model/' + 'MobileNetV2_' + time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + '.pth'
    #保存模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'criterion_state_dict': criterion.state_dict(),
    }, model_save_path)

def main():
    starttime = time.time()

    #随机种子
    torch.manual_seed(1)  # 为CPU设置随机种子
    torch.cuda.manual_seed(1)  # 为当前GPU设置随机种子
    np.random.seed(1)

    mobilenet()
    endtime = time.time()
    print("总共费时：",(endtime-starttime),"secs")


if __name__ == '__main__':
    main()

