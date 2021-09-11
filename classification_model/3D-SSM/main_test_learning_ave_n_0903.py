
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import os
from read_seq_data_0421 import GetSample
from torch.autograd import Variable

from resnet import generate_model
from save_para import SavePara

# 定义是否使用GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class RNN_net(nn.Module):
    def __init__(self,seq):
        super(RNN_net, self).__init__()
        self.seq = seq
        self.rnn_hiden = 10
        self.lstm = nn.LSTM(2, self.rnn_hiden, 2, batch_first=False)
        self.fc = nn.Linear(self.rnn_hiden, 2)

    def forward(self, x):

        x, (h_n, c_n) = self.lstm(x)
        x = x[-1,:,:]
        x = self.fc(x)

        return x

class FCN_RNN_net(nn.Module):
    def __init__(self,seq):
        super(FCN_RNN_net, self).__init__()
        self.seq = seq

        self.relu = nn.ReLU()
        self.fc = nn.Linear(10,2)

        self.rnn_hiden = 10
        self.lstm = nn.LSTM(2, self.rnn_hiden, 2, batch_first=False)

    def forward(self, x, label):

        label_onehot = torch.nn.functional.one_hot(label, 2).float()
        label_onehot = label_onehot.transpose(0, 1).contiguous()

        x = torch.unsqueeze(x,dim=0)

        x1 = torch.cat([label_onehot,x],dim=0)

        xx, (h_n, c_n) = self.lstm(x1)
        xx = xx[-1, :, :]
        xx = self.fc(xx)

        return xx

class FCN_net(nn.Module):
    def __init__(self,feature=32):
        super(FCN_net, self).__init__()
        self.linear1 = nn.Linear(feature, 100)
        self.linear2 = nn.Linear(100, 50)
        self.linear3 = nn.Linear(50, 2)
        # self.linear4 = nn.Linear(20, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        # x = self.relu(x)
        # x = self.linear4(x)
        return x
    
class emsembel_FCN_net(nn.Module):
    def __init__(self,input_dim):
        super(emsembel_FCN_net, self).__init__()
        self.linear1 = nn.Linear(input_dim, 30)
        self.linear2 = nn.Linear(30, 2)
        # self.linear3 = nn.Linear(100, 20)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
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

def ensemble():
    path_fcn = '/home/zhang/disk2/001yangbin/001vaa3d/001_img_classification_model/Code_python/Code_pytorch/FCN_L-measure/model/FCN_LM32_2021-04-22 16:42:39.pth'
    path_fcn_rnn = '/home/zhang/disk2/001yangbin/001vaa3d/001_img_classification_model/Code_python/Code_pytorch/Independent_FCN_LM_RNN/model/only_weight_FCN_RNN_LPR_test_predict_label_seq_5_2021-04-22 21:28:18.pth'
    path_resnet = '/home/zhang/disk2/001yangbin/001vaa3d/001_img_classification_model/Code_python/Code_pytorch/Resnet_1115/model/Resnet_10_2021-09-05 03:26:56.pth'

    path_resnet_rnn = '/home/zhang/disk2/001yangbin/001vaa3d/001_img_classification_model/Code_python/Code_pytorch/Independent_Resnet_label_RNN_0314/model/only_weight_resnet_rnn_seq_5_2021-09-06 23:32:54.pth'

    file_name_pair = '/home/zhang/disk2/001yangbin/001vaa3d/007_RNN_sample/seq_5_rnn_all_sample_name_1.23.txt'
    file_lm = '/home/zhang/disk2/001yangbin/001vaa3d/003_label_result/method2_auto_norag_nofew_block_nosoma_Lmeasure_label_0109.txt'

    data_path_train = "/home/zhang/disk2/001yangbin/002vaa3d_img_samples/data_agmentation/original_image_train_23/"
    data_path_test = "/home/zhang/disk2/001yangbin/002vaa3d_img_samples/data_agmentation/original_image_test_23/"

    time1 = time.time()

    model_num_workers = 3
    model_batch_size = 30
    rnn_seq = 5
    lm_feature_num = 32
    debug = False

    resnet_checkpoint = torch.load(path_resnet)
    Resnet_model = generate_model(resnet_checkpoint['model_layers_num']).to(device)
    Resnet_model.load_state_dict(resnet_checkpoint['model_state_dict'])

    resnet_rnn_checkpoint = torch.load(path_resnet_rnn)
    Resnet_RNN_model = RNN_net(seq=rnn_seq).to(device)
    Resnet_RNN_model.load_state_dict(resnet_rnn_checkpoint['model_state_dict'])
    # Resnet_RNN_model = torch.load(path_resnet_rnn)
    
    #导入预训练模型
    fcn_checkpoint = torch.load(path_fcn)
    FCN_model = FCN_net(feature=lm_feature_num).to(device)
    FCN_model.load_state_dict(fcn_checkpoint['model_state_dict'])
    # FCN_model.eval()

    fcn_rnn_checkpoint = torch.load(path_fcn_rnn)
    FCN_RNN_model = FCN_RNN_net(seq=rnn_seq).to(device)
    FCN_RNN_model.load_state_dict(fcn_rnn_checkpoint['model_state_dict'])
    # FCN_RNN_model.eval()
    
    model1 = emsembel_FCN_net(input_dim=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model1.parameters(), lr = 0.0003)

    print('start train...')
    data = GetSample(root=data_path_train,name_pair=file_name_pair, lm_path=file_lm,lm_feature_num=lm_feature_num,seq=rnn_seq)
    test_data = GetSample(root=data_path_test,name_pair=file_name_pair, lm_path=file_lm,lm_feature_num=lm_feature_num,seq=rnn_seq)

    train_loader = torch.utils.data.DataLoader(data, shuffle=True, num_workers=model_num_workers,
                                               batch_size=model_batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, num_workers=model_num_workers,
                                              batch_size=1)

    if(not debug):
        ###保存参数
        save_para_path = os.getcwd() + '/para/'
        save_para = SavePara(save_para_path,s_name= 'seq_'+ str(rnn_seq)+'_')
        save_para.model_scalar('seq:',rnn_seq)
        save_para.model_scalar('path_fcn', path_fcn)
        save_para.model_scalar('path_fcn_rnn', path_fcn_rnn)
        save_para.model_scalar('path_resnet', path_resnet)
        save_para.model_scalar('path_resnet_rnn', path_resnet_rnn)
        save_para.model_scalar('file_name_pair', file_name_pair)
        save_para.model_scalar('file_lm', file_lm)
        save_para.model_scalar('data_path_train', data_path_train)
        save_para.model_scalar('data_path_test', data_path_test)

    # Resnet_model.eval()
    # Resnet_RNN_model.eval()
    # FCN_model.eval()
    # FCN_RNN_model.eval()

    epoch = 30
    ave_num = 5

    train_acc1,train_acc2,train_acc3,train_acc4,train_acc5 = np.zeros(ave_num),np.zeros(ave_num),np.zeros(ave_num),np.zeros(ave_num),np.zeros(ave_num)
    test_acc1, test_acc2, test_acc3, test_acc4, test_acc5 = np.zeros(ave_num), np.zeros(ave_num), np.zeros(ave_num), np.zeros(ave_num), np.zeros(ave_num)

    train_f1 = np.zeros(ave_num)
    test_f1 = np.zeros(ave_num)

    for num in range(ave_num):
        # save_para.writelines_para(['ave_num:', str(num)])
        test_acc_max = 0

        for e in range(epoch):

            # if (not debug):
            #     save_para.writelines_para(['epoch:', str(e)])
            #     save_para.writelines_para(['n', 'neuron_name', 'true label', 'predict label'])
            #     save_para.clear_count_num()

            Resnet_model.train()
            Resnet_RNN_model.train()
            FCN_model.train()
            FCN_RNN_model.train()
            model1.train()
            running_loss = 0

            train_correct1,train_correct2,train_correct3,train_correct4,train_correct5 = 0,0,0,0,0
            train_total = 0
            test_correct1,test_correct2,test_correct3,test_correct4,test_correct5 = 0,0,0,0,0
            test_total = 0

            train_TP = 0
            train_FP = 0
            train_FN = 0
            train_TN = 0
            test_TP = 0
            test_FP = 0
            test_FN = 0
            test_TN = 0

            for data in train_loader:
                image_data, lm_data, labels, name = data

                image_data, labels = image_data.to(device),labels.to(device)
                image_data = Variable(image_data)

                image_data = torch.squeeze(image_data)
                image_data = torch.unsqueeze(image_data, 1)
                # data = data.squeeze(0)
                image_data = image_data.float()

                lm_data = lm_data.float()
                lm_data = lm_data.to(device)
                lm_data = Variable(lm_data)

                resnet_label = Resnet_model(image_data)

                output1 = torch.unsqueeze(resnet_label, dim=0)
                tem_label = labels[:, :(rnn_seq - 1)]
                tem_label = torch.nn.functional.one_hot(tem_label, 2).float()
                tem_label = tem_label.transpose(0, 1).contiguous()
                # tem_label = torch.unsqueeze(tem_label,dim=0)
                output1 = torch.cat([tem_label, output1], dim=0)

                resnet_rnn_label = Resnet_RNN_model(output1)

                fcn_label = FCN_model(lm_data)

                fcn_rnn_label = FCN_RNN_model(fcn_label, labels[:, :(rnn_seq - 1)])

                # fcn_rnn_label = Variable(fcn_rnn_label)
                # resnet_rnn_label = Variable(resnet_rnn_label)
                input_x = torch.cat([fcn_rnn_label,resnet_rnn_label],dim=1)

                # 权重参数清零
                optimizer.zero_grad()
                out = model1(input_x)

                _, predicted1 = torch.max(out.data, 1)
                _, predicted2 = torch.max(resnet_label.data, 1)
                _, predicted3 = torch.max(resnet_rnn_label.data, 1)
                _, predicted4 = torch.max(fcn_label.data, 1)
                _, predicted5 = torch.max(fcn_rnn_label.data, 1)

                train_correct1 += (predicted1 == labels[:,rnn_seq-1]).sum()
                train_correct2 += (predicted2 == labels[:,rnn_seq-1]).sum()
                train_correct3 += (predicted3 == labels[:,rnn_seq-1]).sum()
                train_correct4 += (predicted4 == labels[:, rnn_seq-1]).sum()
                train_correct5 += (predicted5 == labels[:, rnn_seq - 1]).sum()
                train_total += labels.size(0)

                TP, FP, FN, TN = comculate_f1(label=labels[:, (rnn_seq - 1)], predict=predicted1)
                train_TP += TP
                train_FP += FP
                train_FN += FN
                train_TN += TN

                loss = criterion(out, labels[:,rnn_seq-1])
                loss.backward()  #  反向传播
                optimizer.step()
                running_loss += loss.item()

            Resnet_model.eval()
            Resnet_RNN_model.eval()
            FCN_model.eval()
            FCN_RNN_model.eval()
            model1.eval()
            dic_pre_label = {}

            for data in test_loader:
                image_data, lm_data, labels, name = data

                test_pre_label = []
                for seq in range(rnn_seq - 1):
                    # 需要人工打标签的情况
                    if ((name[seq] in dic_pre_label) == False):
                        test_pre_label.append(labels[0][seq])
                        dic_pre_label[name[seq]] = labels[0][seq]
                    else:
                        test_pre_label.append(dic_pre_label[name[seq]])

                image_data, labels = image_data.to(device),labels.to(device)
                image_data = Variable(image_data)

                image_data = torch.squeeze(image_data)
                image_data = torch.unsqueeze(image_data, 0)
                image_data = torch.unsqueeze(image_data, 1)
                # data = data.squeeze(0)
                image_data = image_data.float()

                lm_data = lm_data.float()
                lm_data = lm_data.to(device)

                resnet_label = Resnet_model(image_data)

                output1 = torch.unsqueeze(resnet_label, dim=0)
                test_pre_label = torch.Tensor(test_pre_label).long().to(device)
                test_pre_label = torch.unsqueeze(test_pre_label, dim=0)
                tem_label = test_pre_label
                tem_label = torch.nn.functional.one_hot(tem_label, 2).float()
                tem_label = tem_label.transpose(0, 1).contiguous()
                # tem_label = torch.unsqueeze(tem_label,dim=0)
                output1 = torch.cat([tem_label, output1], dim=0)

                resnet_rnn_label = Resnet_RNN_model(output1)

                fcn_label = FCN_model(lm_data)

                fcn_rnn_label = FCN_RNN_model(fcn_label, labels[:, :(rnn_seq - 1)])

                fcn_rnn_label = Variable(fcn_rnn_label)
                resnet_rnn_label = Variable(resnet_rnn_label)
                input_x = torch.cat([fcn_rnn_label, resnet_rnn_label], dim=1)

                out = model1(input_x)

                _, predicted1 = torch.max(out.data, 1)
                _, predicted2 = torch.max(resnet_label.data, 1)
                _, predicted3 = torch.max(resnet_rnn_label.data, 1)
                _, predicted4 = torch.max(fcn_label.data, 1)
                _, predicted5 = torch.max(fcn_rnn_label.data, 1)

                # 将预测的结果添加到字典中
                if ((name[rnn_seq - 1] in dic_pre_label) == False):
                    dic_pre_label[name[rnn_seq - 1]] = predicted5

                test_correct1 += (predicted1 == labels[:, rnn_seq - 1]).sum()
                test_correct2 += (predicted2 == labels[:, rnn_seq - 1]).sum()
                test_correct3 += (predicted3 == labels[:, rnn_seq - 1]).sum()
                test_correct4 += (predicted4 == labels[:, rnn_seq - 1]).sum()
                test_correct5 += (predicted5 == labels[:, rnn_seq - 1]).sum()

                test_total += labels.size(0)

                TP, FP, FN, TN = comculate_f1(label=labels[:, (rnn_seq - 1)], predict=predicted1)
                test_TP += TP
                test_FP += FP
                test_FN += FN
                test_TN += TN

                if (not debug):
                    ###　保存错误样本标签
                    save_para.predict_para(name[rnn_seq-1], labels[:, rnn_seq - 1].cpu().numpy(), predicted1.cpu().numpy())

            print('ave:%d epoch:%d'%(num+1, e+1))
            tem = (100 * float(test_correct1) / float(test_total))
            if(tem > test_acc_max):

                train_acc1[num] = (100 * float(train_correct1) / float(train_total))
                train_acc2[num] = (100 * float(train_correct2) / float(train_total))
                train_acc3[num] = (100 * float(train_correct3) / float(train_total))
                train_acc4[num] = (100 * float(train_correct4) / float(train_total))
                train_acc5[num] = (100 * float(train_correct5) / float(train_total))
                train_loss = running_loss / (len(train_loader))
                print('epoch:%d train acc1:%.03f%% resnet acc:%.03f%%  rnn acc:%.03f%% fcn acc:%.03f%% fcn rnn acc:%.03f%% loss:%f' % (e + 1, train_acc1[num],train_acc2[num], train_acc3[num],train_acc4[num],train_acc5[num],train_loss))

                test_acc1[num] = (100 * float(test_correct1) / float(test_total))
                test_acc2[num] = (100 * float(test_correct2) / float(test_total))
                test_acc3[num] = (100 * float(test_correct3) / float(test_total))
                test_acc4[num] = (100 * float(test_correct4) / float(test_total))
                test_acc5[num] = (100 * float(test_correct5) / float(test_total))
                print('test acc1:%.03f%% resnet acc:%.03f%% rnn acc:%.03f%% fcn acc:%.03f%% fcn rnn acc:%.03f%%' % (test_acc1[num], test_acc2[num], test_acc3[num],test_acc4[num],test_acc5[num]))

                # f1
                train_f1[num] = 100 * (2 * train_TP) / (train_total + train_TP - train_TN)
                test_f1[num] = 100 * (2 * test_TP) / (test_total + test_TP - test_TN)
                print('train---TP:%d FP:%d  FN:%d TN:%d' % (train_TP, train_FP, train_FN, train_TN))
                print('test---TP:%d FP:%d  FN:%d TN:%d' % (test_TP, test_FP, test_FN, test_TN))
                print('train f1:%.03f %%' % (train_f1[num]))
                print('test f1:%.03f %%' % (test_f1[num]))

                model_save_path = os.getcwd() + '/model/' + '2021_09_03_' + 'fcn_end_seq_' + str(rnn_seq) + '_' +'.pth'
                # 保存模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model1.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'criterion_state_dict': criterion.state_dict(),
                    'seq': seq,
                }, model_save_path)

    if(not debug):
        save_para.model_scalar('tarin acc', train_acc1)
        save_para.model_scalar('train f1', train_f1)
        save_para.model_scalar('TP, FP, FN, TN=', [train_TP, train_FP, train_FN, train_TN])
        save_para.model_scalar('test acc', test_acc1)
        save_para.model_scalar('test f1', test_f1)
        save_para.model_scalar('TP, FP, FN, TN=', [test_TP, test_FP, test_FN, test_TN])

    train_ave1, train_std1 = np.average(train_acc1), np.std(train_acc1)
    train_ave2, train_std2 = np.average(train_acc2), np.std(train_acc2)
    train_ave3, train_std3 = np.average(train_acc3), np.std(train_acc3)
    train_ave4, train_std4 = np.average(train_acc4), np.std(train_acc4)
    train_ave5, train_std5 = np.average(train_acc5), np.std(train_acc5)

    test_ave1, test_std1 = np.average(test_acc1), np.std(test_acc1)
    test_ave2, test_std2 = np.average(test_acc2), np.std(test_acc2)
    test_ave3, test_std3 = np.average(test_acc3), np.std(test_acc3)
    test_ave4, test_std4 = np.average(test_acc4), np.std(test_acc4)
    test_ave5, test_std5 = np.average(test_acc5), np.std(test_acc5)

    train_f1_mean = np.mean(train_f1)
    train_f1_std = np.std(train_f1)
    test_f1_mean = np.mean(test_f1)
    test_f1_std = np.std(test_f1)


    if(debug == False):
        save_para.model_scalar('train ', 'acc:')
        save_para.model_scalar('resnet acc:', str(train_acc2))
        save_para.model_scalar('resnet_rnn acc:', str(train_acc3))
        save_para.model_scalar('fcn acc:', str(train_acc4))
        save_para.model_scalar('fcn_rnn acc:', str(train_acc5))

        save_para.writelines_para(['resnet acc:', str(train_ave2), str(train_std2)])
        save_para.writelines_para(['resnet_rnn acc:', str(train_ave3), str(train_std3)])
        save_para.writelines_para(['fcn acc:', str(train_ave4), str(train_std4)])
        save_para.writelines_para(['fcn_rnn acc:', str(train_ave5), str(train_std5)])

        save_para.model_scalar('test ', 'acc:')
        save_para.model_scalar('resnet acc:', str(test_acc2))
        save_para.model_scalar('resnet_rnn acc:', str(test_acc3))
        save_para.model_scalar('fcn acc:', str(test_acc4))
        save_para.model_scalar('fcn_rnn acc:', str(test_acc5))
        save_para.writelines_para(['resnet acc:', str(test_ave2), str(test_std2)])
        save_para.writelines_para(['resnet_rnn acc:', str(test_ave3), str(test_std3)])
        save_para.writelines_para(['fcn acc:', str(test_ave4), str(test_std4)])
        save_para.writelines_para(['fcn_rnn acc:', str(test_ave5), str(test_std5)])

        save_para.writelines_para(['end acc:', str(train_ave1), str(train_std1)])
        save_para.writelines_para(['end acc:', str(test_ave1), str(test_std1)])
        save_para.model_scalar('train_f1_mean', train_f1_mean)
        save_para.model_scalar('train_f1_std', train_f1_std)
        save_para.model_scalar('test_f1_mean', test_f1_mean)
        save_para.model_scalar('test_f1_std', test_f1_std)

        time2 = time.time()
        save_para.model_scalar('spend time:', str(time2-time1))

def main():
    starttime = time.time()
    ensemble()
    endtime = time.time()
    print("总共费时：",(endtime-starttime),"secs")


if __name__ == '__main__':
    main()