import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import time
import os
from read_data import GetSample_data
from take_Lmeasure_feature import read_nd_lm_test,read_nd_lm_train
from save_para import SavePara

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_label(na,la):
    for i in range(len(na)):
        print(na[i],i,na[i].split('.')[0].split('l')[1])
        la[i] = int(na[i].split('.')[0].split('l')[1])
    return la


class FCN_net(nn.Module):
    def __init__(self,n_class):
        super(FCN_net, self).__init__()
        self.linear1 = nn.Linear(n_class, 100)
        self.linear2 = nn.Linear(100, 50)
        self.linear3 = nn.Linear(50, 2)
        # self.linear4 = nn.Linear(20, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.linear3(x)
        # x = self.relu(x)
        # x = self.linear4(x)
        return x


def fcn_lmeasure():
    file_path = '/home/zhang/disk2/001yangbin/001vaa3d/003_label_result/method2_auto_norag_nofew_block_Lmeasure_32_have_label_0424.txt'
    # file_path = '/home/zhang/disk2/001yangbin/001vaa3d/003_label_result/method2_auto_norag_nofew_block_Lmeasure_9_label_0130.txt'
    # file_path = '/home/zhang/disk2/001yangbin/001vaa3d/003_label_result/method2_auto_norag_nofew_block_Lmeasure_label_0104.txt'

    # name, data = read_nd_lm(file_path,7)
    #
    # label = np.zeros((len(name)))
    #
    # label = get_label(name,label)
    #
    # # print(name.shape,label.shape)
    model_batch_size = 50
    model_num_workers = 2
    model_lr = 0.0001
    Epoch = 30
    feasure_num = 32

    neu_name_train, data_train = read_nd_lm_train(file_path, feasure_num)
    neu_name_test, data_test = read_nd_lm_test(file_path, feasure_num)

    print(data_train.shape,data_test.shape)

    ###保存参数
    save_para_path = os.getcwd() + '/para/'
    tem_name = '001stem_'
    save_para = SavePara(pre_name=tem_name,path=save_para_path)
    save_para.model_scalar('file:', file_path)
    save_para.model_scalar('model_lr', model_lr)
    save_para.model_scalar('batch_size', model_batch_size)
    save_para.model_scalar('epoch', Epoch)


    print('start train...')
    data = GetSample_data(dataset=data_train,name=neu_name_train)
    val_data = GetSample_data(dataset=data_test, name=neu_name_test)

    # print('train shape:', data.__len__(), 'test shape:', val_data.__len__())
    train_loader = torch.utils.data.DataLoader(data, shuffle=True, num_workers=model_num_workers,
                                               batch_size=model_batch_size)
    test_loader = torch.utils.data.DataLoader(val_data, shuffle=False, num_workers=model_num_workers,
                                              batch_size=model_batch_size)

    model = FCN_net(n_class=feasure_num).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = model_lr)

    # #导入预训练模型
    # path_fcn = '/home/zhang/disk2/001yangbin/001vaa3d/001_img_classification_model/Code_python/Code_pytorch/FCN_L-measure/model/Resnet_10_2021-03-30 10:56:50.pth'
    # fcn_checkpoint = torch.load(path_fcn)
    # FCN_model = FCN_net(n_class=feasure_num).to(device)
    # FCN_model.load_state_dict(fcn_checkpoint['model_state_dict'])

    max_train=0
    max_test=0
    acc_train_tem = 0
    acc_test_tem = 0
    # 训练模型

    for epoch in range(Epoch):
        model.train()

        save_para.writelines_para(['epoch:', str(epoch)])
        save_para.writelines_para(['n', 'neuron_name', 'true label', 'predict label'])
        save_para.clear_count_num()

        running_loss = 0.0
        correct = 0
        correct1 = 0
        total = 0
        for data in train_loader:
            # 训练数据
            trian_images_name, inputs, labels = data
            labels = labels.long()
            inputs = inputs.float()
            # inputs = torch.tensor(inputs, dtype=torch.float32)
            inputs, labels = inputs.to(device), labels.to(device)

            # 权重参数清零
            optimizer.zero_grad()
            outputs = model(inputs)  # 正向传播

            _,predicted = torch.max(outputs.data, 1)

            # print(labels)

            loss = criterion(outputs, labels)
            loss.backward()  #  反向传播
            optimizer.step()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # _, predict1 = torch.max(out.data, 1)
            # correct1 += (predict1 == labels).sum().item()

            running_loss += loss.item()

        acc_train_tem = 100 * correct / total
        # acc_train_1 = 100 * correct1 / total
        print('epoch:%d  train acc:%.3f%% '%(epoch + 1, 100 * correct / total))

        #测试模型
        correct = 0
        # correct1 = 0
        total = 0
        model.eval()
        # with torch.no_grad():
        for data in test_loader:
            test_images_name, inputs, labels = data
            labels = labels.long()
            inputs = inputs.float()
            # inputs = torch.tensor(inputs, dtype=torch.float32)
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _,predicted = torch.max(outputs.data, 1)

            ###　保存错误样本标签
            save_para.predict_para(test_images_name, labels.cpu().numpy(), predicted.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()


        acc_test_tem = 100 * correct / total
        # acc_test_1 = 100 * correct1 / total
        if(max_test<acc_test_tem):
            max_train = acc_train_tem
            max_test = acc_test_tem
        print('test images：%.3f%%'%(100 * correct / total))

    save_para.model_scalar('train max acc:', str(max_train))
    save_para.model_scalar('test max acc:', str(max_test))
    save_para.model_scalar('train end acc:', str(acc_train_tem))
    save_para.model_scalar('test end acc:', str(acc_test_tem))


    model_save_path = os.getcwd() + '/model/' + 'FCN_LM32_nomalization_' + time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + '.pth'
    #保存模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'criterion_state_dict': criterion.state_dict()
    }, model_save_path)


def main():
    starttime = time.time()

    print('start...')
    fcn_lmeasure()

    endtime = time.time()
    print("总共费时：",(endtime-starttime),"secs")


if __name__ == '__main__':
    main()