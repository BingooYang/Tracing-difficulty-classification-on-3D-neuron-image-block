
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from read_seq_data_0421 import GetSample
from torch.autograd import Variable

from resnet import generate_model

# 定义是否使用GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class RNN_net(nn.Module):
    def __init__(self, seq):
        super(RNN_net, self).__init__()
        self.seq = seq
        self.rnn_hiden = 10
        self.lstm = nn.LSTM(2, self.rnn_hiden, 2, batch_first=False)
        self.fc = nn.Linear(self.rnn_hiden, 2)

    def forward(self, x):
        x, (h_n, c_n) = self.lstm(x)
        x = x[-1, :, :]
        x = self.fc(x)

        return x


class FCN_RNN_net(nn.Module):
    def __init__(self, seq):
        super(FCN_RNN_net, self).__init__()
        self.seq = seq

        self.relu = nn.ReLU()
        self.fc = nn.Linear(10, 2)

        self.rnn_hiden = 10
        self.lstm = nn.LSTM(2, self.rnn_hiden, 2, batch_first=False)

    def forward(self, x, label):
        label_onehot = torch.nn.functional.one_hot(label, 2).float()
        label_onehot = label_onehot.transpose(0, 1).contiguous()

        x = torch.unsqueeze(x, dim=0)

        x1 = torch.cat([label_onehot, x], dim=0)

        xx, (h_n, c_n) = self.lstm(x1)
        xx = xx[-1, :, :]
        xx = self.fc(xx)

        return xx


class FCN_net(nn.Module):
    def __init__(self, feature=32):
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
    def __init__(self, input_dim):
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
    path_fcn_rnn = '/home/zhang/disk2/001yangbin/001vaa3d/001_img_classification_model/Code_python/Code_pytorch/Independent_FCN_LM_RNN/model/only_weight_FCN_RNN_LPR_test_label_seq_5_2021-04-22 19:25:35.pth'
    path_resnet = '/home/zhang/disk2/001yangbin/001vaa3d/001_img_classification_model/Code_python/Code_pytorch/Resnet_1115/model/Resnet_10_2021-09-05 03:26:56.pth'

    path_resnet_rnn = '/home/zhang/disk2/001yangbin/001vaa3d/001_img_classification_model/Code_python/Code_pytorch/Independent_Resnet_label_RNN_0314/model/only_weight_resnet_rnn_seq_5_2021-09-06 23:32:54.pth'

    file_name_pair = '/home/zhang/disk2/001yangbin/001vaa3d/007_RNN_sample/17545_RNN_sample_name/seq_5_rnn_name_threshold_100_0420.txt'
    file_lm = '/home/zhang/disk2/001yangbin/001vaa3d/003_label_result/17545/LM_auto_nosoma_Lmeasure_label_32_0424.txt'

    path_fcn_end = '/home/zhang/disk2/001yangbin/001vaa3d/001_img_classification_model/Code_python/Code_pytorch/01_framwork/model/2021_09_03_fcn_end_seq_5_.pth'

    data_path_train = "/home/zhang/disk2/001yangbin/002vaa3d_img_samples/17545/17545_blocks_415_backname/outimg_good_block/"

    model_num_workers = 3
    model_batch_size = 1
    rnn_seq = 5
    lm_feature_num = 32

    resnet_checkpoint = torch.load(path_resnet)
    Resnet_model = generate_model(resnet_checkpoint['model_layers_num'],dropout_radio=0.2).to(device)
    Resnet_model.load_state_dict(resnet_checkpoint['model_state_dict'])

    resnet_rnn_checkpoint = torch.load(path_resnet_rnn)
    Resnet_RNN_model = RNN_net(seq=rnn_seq).to(device)
    Resnet_RNN_model.load_state_dict(resnet_rnn_checkpoint['model_state_dict'])
    # Resnet_RNN_model = torch.load(path_resnet_rnn)
    
    #导入预训练模型
    fcn_checkpoint = torch.load(path_fcn)
    FCN_model = FCN_net(feature=lm_feature_num).to(device)
    FCN_model.load_state_dict(fcn_checkpoint['model_state_dict'])

    fcn_rnn_checkpoint = torch.load(path_fcn_rnn)
    FCN_RNN_model = FCN_RNN_net(seq=rnn_seq).to(device)
    FCN_RNN_model.load_state_dict(fcn_rnn_checkpoint['model_state_dict'])

    fcn_end_checkpoint = torch.load(path_fcn_end)
    FCN_END_model = emsembel_FCN_net(input_dim=4).to(device)
    FCN_END_model.load_state_dict(fcn_end_checkpoint['model_state_dict'])

    Resnet_model.eval()
    Resnet_RNN_model.eval()
    FCN_model.eval()
    FCN_RNN_model.eval()
    FCN_END_model.eval()

    print('start train...')
    data = GetSample(root=data_path_train,name_pair=file_name_pair, lm_path=file_lm,lm_feature_num=lm_feature_num,seq=rnn_seq)

    train_loader = torch.utils.data.DataLoader(data, shuffle=True, num_workers=model_num_workers,
                                               batch_size=model_batch_size)

    print('train shape:', data.__len__())


    epoch = 1
    for e in range(epoch):

        running_loss = 0
        correct1,correct2,correct3,correct4,correct5 = 0,0,0,0,0
        total = 0

        test_TP = 0
        test_FP = 0
        test_FN = 0
        test_TN = 0

        fcn_lstm_TP = 0
        fcn_lstm_FP = 0
        fcn_lstm_FN = 0
        fcn_lstm_TN = 0

        resnet_lstm_TP = 0
        resnet_lstm_FP = 0
        resnet_lstm_FN = 0
        resnet_lstm_TN = 0

        fcn_TP = 0
        fcn_FP = 0
        fcn_FN = 0
        fcn_TN = 0

        resnet_TP = 0
        resnet_FP = 0
        resnet_FN = 0
        resnet_TN = 0

        dic_pre_label = {}
        for data in train_loader:
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
            # image_data = Variable(image_data)

            image_data = torch.squeeze(image_data)
            image_data = torch.unsqueeze(image_data, 0)
            image_data = torch.unsqueeze(image_data, 1)
            # data = data.squeeze(0)
            image_data = image_data.float()

            lm_data = lm_data.float()
            lm_data = lm_data.to(device)
            # lm_data = Variable(lm_data)

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
            input_x = torch.cat([fcn_rnn_label,resnet_rnn_label],dim=1)

            out = FCN_END_model(input_x)

            _, predicted1 = torch.max(out.data, 1)
            _, predicted2 = torch.max(resnet_label.data, 1)
            _, predicted3 = torch.max(resnet_rnn_label.data, 1)
            _, predicted4 = torch.max(fcn_label.data, 1)
            _, predicted5 = torch.max(fcn_rnn_label.data, 1)

            # 将预测的结果添加到字典中
            dic_pre_label[name[rnn_seq - 1]] = predicted1

            correct1 += (predicted1 == labels[:,rnn_seq-1]).sum()
            correct2 += (predicted2 == labels[:,rnn_seq-1]).sum()
            correct3 += (predicted3 == labels[:,rnn_seq-1]).sum()
            correct4 += (predicted4 == labels[:, rnn_seq-1]).sum()
            correct5 += (predicted5 == labels[:, rnn_seq - 1]).sum()
            total += labels.size(0)

            TP, FP, FN, TN = comculate_f1(label=labels[:, (rnn_seq - 1)], predict=predicted1)
            test_TP += TP
            test_FP += FP
            test_FN += FN
            test_TN += TN

            TP, FP, FN, TN = comculate_f1(label=labels[:, (rnn_seq - 1)], predict=predicted2)
            resnet_TP += TP
            resnet_FP += FP
            resnet_FN += FN
            resnet_TN += TN

            TP, FP, FN, TN = comculate_f1(label=labels[:, (rnn_seq - 1)], predict=predicted3)
            resnet_lstm_TP += TP
            resnet_lstm_FP += FP
            resnet_lstm_FN += FN
            resnet_lstm_TN += TN

            TP, FP, FN, TN = comculate_f1(label=labels[:, (rnn_seq - 1)], predict=predicted4)
            fcn_TP += TP
            fcn_FP += FP
            fcn_FN += FN
            fcn_TN += TN

            TP, FP, FN, TN = comculate_f1(label=labels[:, (rnn_seq - 1)], predict=predicted5)
            fcn_lstm_TP += TP
            fcn_lstm_FP += FP
            fcn_lstm_FN += FN
            fcn_lstm_TN += TN

        test_acc1 = (100 * float(correct1) / float(total))
        test_acc2 = (100 * float(correct2) / float(total))
        test_acc3 = (100 * float(correct3) / float(total))
        test_acc4 = (100 * float(correct4) / float(total))
        test_acc5 = (100 * float(correct5) / float(total))
        # test_loss = running_loss / (len(train_loader))

        test_f1 = 100 * (2 * test_TP) / (total + test_TP - test_TN)
        test_f2 = 100 * (2 * resnet_TP) / (total + resnet_TP - resnet_TN)
        test_f3 = 100 * (2 * resnet_lstm_TP) / (total + resnet_lstm_TP - resnet_lstm_TN)
        test_f4 = 100 * (2 * fcn_TP) / (total + fcn_TP - fcn_TN)
        test_f5 = 100 * (2 * fcn_lstm_TP) / (total + fcn_lstm_TP - fcn_lstm_TN)
        print('epoch:%d test acc1:%.03f%% resnet acc:%.03f%%  resnet_rnn acc:%.03f%% fcn acc:%.03f%% fcn_rnn acc:%.03f%%' % (e + 1, test_acc1,test_acc2, test_acc3,test_acc4,test_acc5))

        print('epoch:%d test f1:%.03f%% resnet f1:%.03f%%  rnn f1:%.03f%% resnet_fcn f1:%.03f%% fcn_rnn f1:%.03f%%' % (e + 1, test_f1,test_f2, test_f3,test_f4,test_f5,))


def main():
    starttime = time.time()
    ensemble()
    endtime = time.time()
    print("总共费时：",(endtime-starttime),"secs")


if __name__ == '__main__':
    main()