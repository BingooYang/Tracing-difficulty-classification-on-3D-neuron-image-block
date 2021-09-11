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
import torch.nn.functional as F
import torch.optim as optim


from resnet import generate_model
from read_data import GetSample
from save_para import SavePara

# 定义是否使用GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class RNN_net(nn.Module):
    def __init__(self,seq):
        super(RNN_net, self).__init__()
        self.seq = seq

        # self.linear1 = nn.Linear(feature_len, 50)
        # self.linear2 = nn.Linear(50, 20)
        # # self.linear3 = nn.Linear(100, 30)
        # self.relu = nn.ReLU()

        self.rnn_hiden = 10
        self.lstm = nn.LSTM(2, self.rnn_hiden, 2, batch_first=False)
        self.fc = nn.Linear(self.rnn_hiden, 2)

    def forward(self, x):

        # x = torch.unsqueeze(x, dim=2)
        # x = torch.transpose(x, dim0=1, dim1=0)
        # x = torch.repeat_interleave(x, repeats=2, dim=2)
        x, (h_n, c_n) = self.lstm(x)
        x = x[-1,:,:]

        x = self.fc(x)

        return x

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

def resnet_1():

    starttime = time.time()

    model_lr = 0.0001
    model_batch_size = 30
    model_num_workers = 5
    epoch = 30
    dropout = 0.2

    debug = False
    print('start Independent_resnet_rnn_0903')

    path_resnet = '/home/zhang/disk2/001yangbin/001vaa3d/001_img_classification_model/Code_python/Code_pytorch/Resnet_1115/model/Resnet_10_2021-09-05 03:26:56.pth'
    resnet_checkpoint = torch.load(path_resnet)
    Resnet_model = generate_model(10,dropout_radio=dropout).to(device)
    Resnet_model.load_state_dict(resnet_checkpoint['model_state_dict'])
    Resnet_model.eval()

    data_path_train = "/home/zhang/disk2/001yangbin/002vaa3d_img_samples/data_agmentation/original_image_train_23/"
    data_path_test = "/home/zhang/disk2/001yangbin/002vaa3d_img_samples/data_agmentation/original_image_test_23/"
    file_pair = '/home/zhang/disk2/001yangbin/001vaa3d/007_RNN_sample/'

    print('start train...')

    if(not debug):
        ###保存参数
        save_para_path = os.getcwd() + '/para/'
        save_para = SavePara(save_para_path)
        save_para.model_scalar('model_lr',model_lr)
        save_para.model_scalar('model_num_workers', model_num_workers)
        save_para.model_scalar('epoch', epoch)
        save_para.model_scalar('dropout', dropout)
        save_para.model_scalar('train path', data_path_train)
        save_para.model_scalar('test path', data_path_test)
        save_para.model_scalar('name pair:', file_pair)
        save_para.model_scalar('path resnet:', path_resnet)

    seq_num = [2,3,4,5]
    ave_num = 4
    train_acc = np.zeros(ave_num)
    test_acc = np.zeros(ave_num)

    train_f1 = np.zeros(ave_num)
    test_f1 = np.zeros(ave_num)

    for n in range(ave_num):
        rnn_seq = seq_num[n]
        name_tem = 'seq_' + str(rnn_seq) + '_rnn_augment_data_sample_name_3.30.txt'
        name_pair = file_pair + name_tem

        data = GetSample(root=data_path_train, name_pair=name_pair, seq=rnn_seq)
        val_data = GetSample(root=data_path_test, name_pair=name_pair, seq=rnn_seq)
        print('train shape:', data.__len__(), 'test shape:', val_data.__len__())
        train_loader = torch.utils.data.DataLoader(data, shuffle=True, num_workers=model_num_workers, batch_size=model_batch_size)
        test_loader = torch.utils.data.DataLoader(val_data, shuffle=False, num_workers=model_num_workers, batch_size=1)

        model = RNN_net(seq=rnn_seq).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=model_lr)

        for e in range(epoch):
            model.train()
            sum_loss = 0.0

            correct1,correct2 = 0, 0

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

            for batch_idx, (train_data, label_seq, image_name) in enumerate(train_loader):

                train_data, label_seq = train_data.to(device), label_seq.to(device)
                train_data, label_seq = Variable(train_data), Variable(label_seq)

                train_data = torch.squeeze(train_data)
                train_data = torch.unsqueeze(train_data, 1)
                train_data = train_data.float()

                optimizer.zero_grad()

                outputs = Resnet_model(train_data)

                output1 = torch.unsqueeze(outputs, dim=0)

                tem_label = label_seq[:,:(rnn_seq-1)]
                tem_label = torch.nn.functional.one_hot(tem_label, 2).float()
                tem_label = tem_label.transpose(0, 1).contiguous()
                # tem_label = torch.unsqueeze(tem_label,dim=0)
                output1 = torch.cat([tem_label,output1],dim=0)

                rnn_out = model(output1)
                _, predicted1 = torch.max(rnn_out, 1)
                _, predicted2 = torch.max(outputs, 1)

                total_train += label_seq.size(0)
                correct_train += (predicted1 == label_seq[:,(rnn_seq-1)]).sum()
                correct2 += (predicted2 == label_seq[:, (rnn_seq - 1)]).sum()

                TP, FP, FN, TN = comculate_f1(label=label_seq[:, (rnn_seq - 1)], predict=predicted1)
                train_TP += TP
                train_FP += FP
                train_FN += FN
                train_TN += TN

                loss = criterion(rnn_out, label_seq[:,(rnn_seq-1)])

                loss.backward()
                optimizer.step()

                sum_loss += loss.item()

            model.eval()
            correct1,correct2 = 0, 0
            dic_pre_label = {}
            for batch_idx, (test_data, label_seq,image_name) in enumerate(test_loader):

                test_pre_label = []
                for seq in range(rnn_seq-1):
                    #需要人工打标签的情况
                    if((image_name[seq] in dic_pre_label)==False):
                        test_pre_label.append(label_seq[0][seq])
                        dic_pre_label[image_name[seq]] = label_seq[0][seq]
                    else:
                        test_pre_label.append(dic_pre_label[image_name[seq]])

                test_pre_label = torch.Tensor(test_pre_label).long()
                test_pre_label = torch.unsqueeze(test_pre_label,dim=0)
                test_data, test_pre_label, label_seq = test_data.to(device), test_pre_label.to(device), label_seq.to(device)
                test_data, test_pre_label = Variable(test_data), Variable(test_pre_label)

                # test_data = torch.squeeze(test_data)
                test_data = torch.unsqueeze(test_data, 0)
                # test_data = torch.unsqueeze(test_data, 1)
                test_data = test_data.float()

                outputs = Resnet_model(test_data)
                # outputs = Resnet_model(test_data[:, :, rnn_seq-1, :, :, :])
                output1 = torch.unsqueeze(outputs, dim=0)

                # tem_label = label_seq[:,:(rnn_seq-1)]
                tem_label = test_pre_label
                tem_label = torch.nn.functional.one_hot(tem_label, 2).float()
                tem_label = tem_label.transpose(0, 1).contiguous()
                # tem_label = torch.unsqueeze(tem_label,dim=0)
                output1 = torch.cat([tem_label,output1],dim=0)

                rnn_out = model(output1)

                # 取得分最高的那个类
                _, predicted1 = torch.max(rnn_out, 1)
                _, predicted2 = torch.max(outputs, 1)
                #将预测的结果添加到字典中
                if ((image_name[rnn_seq - 1] in dic_pre_label) == False):
                    dic_pre_label[image_name[rnn_seq-1]] = predicted1
                # if (not debug):
                #     ###　保存错误样本标签
                #     save_para.predict_para(image_name,[label_seq[0][rnn_seq-1]].cpu().numpy(),predicted.cpu().numpy())

                total_test += label_seq.size(0)
                correct_test += (predicted1 == label_seq[:,(rnn_seq-1)]).sum()
                correct2 += (predicted2 == label_seq[:, (rnn_seq - 1)]).sum()

                TP, FP, FN, TN = comculate_f1(label=label_seq[:, (rnn_seq - 1)], predict=predicted1)
                test_TP += TP
                test_FP += FP
                test_FN += FN
                test_TN += TN

        train_acc[n] = (100 * float(correct_train) / float(total_train))
        test_acc[n] = (100 * float(correct_test) / float(total_test))
        print('num:%d train acc:%.03f %% ' % (n + 1, train_acc[n]))
        print('test acc：%.03f %% ' % (test_acc[n]))

        # f1
        train_f1[n] = 100 * (2*train_TP)/(total_train+train_TP-train_TN)
        test_f1[n] = 100 * (2 * test_TP) / (total_test + test_TP - test_TN)
        print('train---TP:%d FP:%d  FN:%d TN:%d' % (train_TP, train_FP, train_FN, train_TN))
        print('test---TP:%d FP:%d  FN:%d TN:%d' % (test_TP, test_FP, test_FN, test_TN))
        print('train f1:%.03f %%' % (train_f1[n]))
        print('test f1:%.03f %%' % (test_f1[n]))


        model_save_path = os.getcwd() + '/model/' + 'only_weight_resnet_rnn_seq_' + str(rnn_seq) + '_' + time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + '.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'criterion_state_dict': criterion.state_dict(),
            'rnn_seq':rnn_seq
        }, model_save_path)

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

def main():
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    starttime = time.time()

    resnet_1()
    endtime = time.time()
    print("总共费时：",(endtime-starttime),"secs")


if __name__ == '__main__':
    main()

