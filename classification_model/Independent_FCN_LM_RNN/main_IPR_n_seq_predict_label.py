import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import time
import os
from read_data import GetSample
from take_Lmeasure_feature import read_nd_lm_normalization,read_nd_lm
from save_para import SavePara

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def get_label(na,la):
    for i in range(len(na)):
        print(na[i],i,na[i].split('.')[0].split('l')[1])
        la[i] = int(na[i].split('.')[0].split('l')[1])
    return la


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
    

def fcn_lmeasure():
    file_path = '/home/zhang/disk2/001yangbin/001vaa3d/003_label_result/method2_auto_norag_nofew_block_Lmeasure_32_have_label_0424.txt'
    file_pair = '/home/zhang/disk2/001yangbin/001vaa3d/007_RNN_sample/'
    # file_pair = '/home/zhang/disk2/001yangbin/001vaa3d/007_RNN_sample/17302_RNN_sample_name/'
    path_fcn = '/home/zhang/disk2/001yangbin/001vaa3d/001_img_classification_model/Code_python/Code_pytorch/FCN_L-measure/model/FCN_LM32_2021-04-22 16:42:39.pth'
    model_batch_size = 10
    model_num_workers = 2
    model_lr = 0.001
    Epoch = 30
    feature_num = 32
    lm_dic = read_nd_lm_normalization(file_path, feature_num)

    debug = False
    if(debug == False):
        ###保存参数
        save_para_path = os.getcwd() + '/para/'
        tem_name = 'RNN_FCN_'
        save_para = SavePara(pre_name=tem_name,path=save_para_path)
        save_para.model_scalar('file:', file_path)
        save_para.model_scalar('model_lr', model_lr)
        save_para.model_scalar('batch_size', model_batch_size)
        save_para.model_scalar('epoch', Epoch)
        save_para.model_scalar('feature_num:', feature_num)

    print('start train...')

    #导入预训练模型
    fcn_checkpoint = torch.load(path_fcn)
    FCN_model = FCN_net(feature=feature_num).to(device)
    FCN_model.load_state_dict(fcn_checkpoint['model_state_dict'])
    FCN_model.eval()
    
    max_train=0
    max_test=0
    acc_train_tem = 0
    acc_test_tem = 0

    seq_num = [2, 3, 4, 5]
    train_acc = np.zeros(len(seq_num))
    test_acc = np.zeros(len(seq_num))

    for n in range(len(seq_num)):
        rnn_seq = seq_num[n]
        # pair_name = 'seq_'+ str(seq_num[n])+'_rnn_name_threshold_100_0420.txt'
        pair_name = 'seq_' + str(seq_num[n]) + '_rnn_all_sample_name_1.23.txt'
        file_name_pair = file_pair +pair_name

        data = GetSample(name_pair=file_name_pair, lm_dic=lm_dic, seq=rnn_seq, mode='train')
        val_data = GetSample(name_pair=file_name_pair, lm_dic=lm_dic, seq=rnn_seq, mode='test')

        train_loader = torch.utils.data.DataLoader(data, shuffle=True, num_workers=model_num_workers,
                                                   batch_size=2)
        test_loader = torch.utils.data.DataLoader(val_data, shuffle=False, num_workers=model_num_workers,
                                                  batch_size=1)

        model = FCN_RNN_net(seq=rnn_seq).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=model_lr)
        # model.train()

        for epoch in range(Epoch):
            if (debug == False):
                save_para.writelines_para(['epoch:', str(epoch)])
                save_para.writelines_para(['n', 'neuron_name', 'true label', 'predict label'])
                save_para.clear_count_num()
    
            model.train()
            # FCN_model.train()
            running_loss = 0.0
            correct1,correct2 = 0,0
            correct_train = 0
            total_train = 0
            correct_test = 0
            total_test = 0

            for i, data in enumerate(train_loader, 0):
                # 训练数据
                trian_images_name, inputs, labels = data
                labels = labels.long()
                inputs = inputs.float()
                # inputs = torch.tensor(inputs, dtype=torch.float32)
                inputs, labels = inputs.to(device), labels.to(device)

                # 权重参数清零
                optimizer.zero_grad()

                out = FCN_model(inputs[:, rnn_seq - 1, :])
                outputs = model(out,labels[:,:(rnn_seq-1)])  # 正向传播

                loss = criterion(outputs, labels[:,rnn_seq-1])
                # aux_loss = criterion(aux_out, labels[:,seq-1])
                # loss = loss*torch.Tensor([0.7]).to(device) + aux_loss*torch.Tensor([0.3]).to(device)

                loss.backward()  #  反向传播
                optimizer.step()

                _, predicted1 = torch.max(outputs.data, 1)
                _, predicted2 = torch.max(out.data, 1)

                total_train += labels.size(0)

                correct_train += (predicted1 == labels[:,rnn_seq-1]).sum()
                correct1 += (predicted2 == labels[:,rnn_seq-1]).sum()

                running_loss += loss.item()
            #
            # train_acc1 = (100 * float(correct_train) / float(total_train))
            # train_acc2 = (100 * float(correct2) / float(total))
            # train_loss = running_loss / (len(train_loader))
            # print('epoch:%d train rnn acc:%.03f%% train fcn acc:%.03f%%train loss:%f' % (epoch + 1, train_acc1, train_acc2, train_loss))
    
            #测试模型
            correct2 = 0
            model.eval()
            FCN_model.eval()
            dic_pre_label = {}
            for data in test_loader:
                test_images_name, inputs, labels = data
    
                test_pre_label = []
                for seq in range(rnn_seq - 1):
                    # 需要人工打标签的情况
                    if ((test_images_name[seq] in dic_pre_label) == False):
                        test_pre_label.append(labels[0][seq])
                        dic_pre_label[test_images_name[seq]] = labels[0][seq]
                    else:
                        test_pre_label.append(dic_pre_label[test_images_name[seq]])
    
                test_pre_label = torch.Tensor(test_pre_label).long()
                test_pre_label = torch.unsqueeze(test_pre_label,dim=0)
    
                labels = labels.long()
                inputs = inputs.float()
                # images = torch.tensor(images, dtype=torch.float32)
                inputs, labels, test_pre_label = inputs.to(device), labels.to(device), test_pre_label.to(device)
    
                out = FCN_model(inputs[:, rnn_seq - 1, :])
    
                tem_label = test_pre_label
                # tem_label = torch.nn.functional.one_hot(tem_label, 2).float()
                # tem_label = tem_label.transpose(0, 1).contiguous()
    
                # outputs = model(out,labels[:,:(rnn_seq-1)])
                outputs = model(out, tem_label)
                _,predicted = torch.max(outputs.data, 1)
    
                _, predicted1 = torch.max(outputs.data, 1)
                _, predicted2 = torch.max(out.data, 1)

                #将预测的结果添加到字典中
                if((test_images_name[rnn_seq - 1] in dic_pre_label) == False):
                    dic_pre_label[test_images_name[rnn_seq-1]] = predicted1

                total_test += labels.size(0)
                correct_test += (predicted1 == labels[:,rnn_seq-1]).sum()
                correct2 += (predicted2 == labels[:, rnn_seq - 1]).sum()
                    
            acc_test_tem = 100 * correct_test / total_test
            if(max_test<acc_test_tem):
                max_train = acc_train_tem
                max_test = acc_test_tem
    
            # test_acc1 = (100 * float(correct1) / float(total))
            # test_acc2 = (100 * float(correct2) / float(total))
            # print('test rnn acc：%.03f%% test fcn acc：%.03f%%' % (test_acc1,test_acc2))

    # if(debug == False):
    #     save_para.model_scalar('train max acc:', str(max_train))
    #     save_para.model_scalar('test max acc:', str(max_test))
    #     save_para.model_scalar('train end acc:', str(acc_train_tem))
    #     save_para.model_scalar('test end acc:', str(acc_test_tem))

        train_acc[n] = (100 * float(correct_train) / float(total_train))
        train_1 = (100 * float(correct1) / float(total_train))
        train_loss = running_loss / (len(train_loader))
        print('num:%d train rnn acc:%.03f %% train fcn:%.03f %% train loss:%f' % (n + 1, train_acc[n],train_1, train_loss))

        test_acc[n] = (100 * float(correct_test) / float(total_test))
        test_1 = (100 * float(correct2) / float(total_test))
        print('test rnn acc：%.03f %% test fcn：%.03f %%' % (test_acc[n],test_1))

        model_save_path = os.getcwd() + '/model/' + 'only_weight_FCN_RNN_LPR_test-predict-label_fcn-no-nomalization_seq_' + str(rnn_seq) + '_' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '.pth'
        # 保存模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'criterion_state_dict': criterion.state_dict(),
            'feature_num': feature_num,
            'seq': seq,
        }, model_save_path)

    if(not debug):
        save_para.model_scalar('############Experiment num:', n+1)
        save_para.model_scalar('tarin acc', train_acc)
        save_para.model_scalar('train loss', train_loss)
        save_para.model_scalar('test acc', test_acc)

    train_acc_mean = np.mean(train_acc)
    # train_acc_std = np.std(train_acc)
    test_acc_mean = np.mean(test_acc)
    # test_acc_std = np.std(test_acc)

    if (not debug):
        save_para.model_scalar('#########################', 'end####################')
        save_para.model_scalar('train_acc_mean', train_acc_mean)
        # save_para.model_scalar('train_acc_std', train_acc_std)
        save_para.model_scalar('test_acc_mean', test_acc_mean)
        # save_para.model_scalar('test_acc_std', test_acc_std)

    # model_save_path = os.getcwd() + '/model/' + 'fcn_rnn_seq_' + str(rnn_seq) + '_' + time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + '.pth'
    # # #保存模型
    # torch.save(model, model_save_path)

    # model_save_path = os.getcwd() + '/model/' + 'whole_FCN_RNN_LPR_seq_' + str(seq) + '_' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '.pth'
    # torch.save(model,model_save_path)
    # # # 保存模型
    # # torch.save({
    # #     'epoch': epoch,
    # #     'model_state_dict': model.state_dict(),
    # #     'optimizer_state_dict': optimizer.state_dict(),
    # #     'criterion_state_dict': criterion.state_dict(),
    # #     'feature_num': feature_num,
    # #     'seq': seq,
    # # }, model_save_path)

def main():
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    starttime = time.time()

    print('start...')
    fcn_lmeasure()

    endtime = time.time()
    print("总共费时：",(endtime-starttime),"secs")


if __name__ == '__main__':
    main()