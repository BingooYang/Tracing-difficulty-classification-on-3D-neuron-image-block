import numpy as np
import pandas as pd
import copy

def save_rnn_sample_txt(data):
    save_path = 'F:\\004Vaa3d\\004feature\\RNN_sample\\rnn_4_class_sample_name_all_samples.txt'

    f = open(save_path, mode='w')
    s = u'####考虑跳跃性，没有配对的样本自己跟自己配对'
    f.write(s)
    f.write('\n')
    # print(data.iat[0,0],data.iat[0,1],data.iat[0,2])
    print(data.shape,data.shape[0],data.shape[1])
    for i in range(data.shape[0]):
        f.writelines([str(i+1),'\t',data.iat[i,0],'\t',data.iat[i,1],'\t',str(data.iat[i,2]),'\n'])

def save_rnn_seq_sample_txt(data,num):
    name = 'seq_' + str(num) + '_rnn_name_threshold_100_0420.txt'
    save_path = 'F:\\004Vaa3d\\004feature\\17545\\RNN_sample_name\\' + name

    f = open(save_path, mode='w')
    s = u'####一行中最后一个是最新的样本'
    f.write(s)
    f.write('\n')

    for i in range(len(data)):
        f.write(str(i+1))
        f.write('\t')
        for j in range(num):
            f.write(data[i][j])
            f.write('\t')
        f.write('\n')

def rnn_4_class_sample_name():
    path = 'F:\\004Vaa3d\\004feature\\all_samples_name\\all_samples_name.txt'

    # 读文本数据
    f = open(path)
    ftextlist = f.readlines()
    re = pd.DataFrame(columns=['name1', 'name2', 'label'])
    class_num = np.zeros(4)
    pre_sample = ''
    pre_label = -1
    pre_x,pre_y,pre_z,x,y,z = -100,-100,-100,0,0,0
    neuron_num, pre_neuron_num = -10,-10
    count = 0
    for text in ftextlist:
        tem = text.split('\t')
        tem_name = tem[1]
        label = int(tem_name.split('l')[1].split('.')[0])
        x = int(tem_name.split('x')[1].split('_')[1])
        y = int(tem_name.split('y')[1].split('_')[1])
        z = int(tem_name.split('z')[1].split('_')[1])
        neuron_num = int(tem_name.split('_')[1])
        if(label!=0):
            label=1
        if(abs(x-pre_x)<=100 and abs(y-pre_y)<=100 and abs(z-pre_z) and abs(neuron_num-pre_neuron_num)==1):
            if(label==0 and pre_label == 0):
                re.loc[count] = [tem_name,pre_sample,0]
                class_num[0] += 1
            elif(label==0 and pre_label == 1):
                re.loc[count] = [tem_name,pre_sample,1]
                class_num[1] += 1
            elif(label==1 and pre_label == 0):
                re.loc[count] = [tem_name,pre_sample,2]
                class_num[2] += 1
            elif(label==1 and pre_label == 1):
                re.loc[count] = [tem_name,pre_sample,3]
                class_num[3] += 1
        else:
            if(label==0):
                re.loc[count] = [tem_name,tem_name,0]
                class_num[0] += 1
            else:
                re.loc[count] = [tem_name,tem_name,3]
                class_num[3] += 1

        pre_sample = tem_name
        pre_label = label
        pre_x = x
        pre_y = y
        pre_z = z
        pre_neuron_num = neuron_num
        count+=1

    print(class_num)
    return re

def rnn_seq_sample_name(seq):
    path = 'F:\\004Vaa3d\\004feature\\all_samples_name\\all_tif_samples_name.txt'

    # 读文本数据
    f = open(path)
    ftextlist = f.readlines()
    re = []
    re_seq = np.empty(0)
    # print(re_seq.shape)
    pre_sample = ''
    pre_x,pre_y,pre_z,x,y,z = -100,-100,-100,0,0,0
    neuron_num, pre_neuron_num = -10,-10

    count = 0
    for text in ftextlist:
        tem = text.split('\t')
        tem_name = tem[1]
        x = int(tem_name.split('x')[1].split('_')[1])
        y = int(tem_name.split('y')[1].split('_')[1])
        z = int(tem_name.split('z')[1].split('_')[1])
        neuron_num = int(tem_name.split('_')[1])

        if(abs(x-pre_x)<=100 and abs(y-pre_y)<=100 and abs(z-pre_z) and abs(neuron_num-pre_neuron_num)==1):
            if(re_seq.shape[0]==0):
                re_seq = np.append(re_seq, pre_sample)
            re_seq = np.append(re_seq,tem_name)
            if(re_seq.shape[0]<seq):
                tem_re_seq = np.copy(re_seq)
                for i in range(re_seq.shape[0],seq):
                    tem_re_seq = np.insert(tem_re_seq,0,re_seq[0])
            else:
                tem_re_seq = re_seq[re_seq.shape[0]-seq:re_seq.shape[0]]
            # print(tuple(tem_re_seq))
            re.append(tuple(tem_re_seq))
        else:
            re_seq = np.empty(0)
            for i in range(seq):
                re_seq = np.append(re_seq,tem_name)
            # print(tuple(re_seq))
            re.append(tuple(re_seq))

            re_seq = np.empty(0)

        pre_sample = tem_name
        pre_x = x
        pre_y = y
        pre_z = z
        pre_neuron_num = neuron_num
        count+=1

    return re

def find_neighbor(name_list,target_name):
    target_x = int(target_name.split('x')[1].split('_')[1])
    target_y = int(target_name.split('y')[1].split('_')[1])
    target_z = int(target_name.split('z')[1].split('_')[1].split('.')[0])
    length = len(name_list)
    for i in range(length):
        x = int(name_list[length-1-i].split('x')[1].split('_')[1])
        y = int(name_list[length-1-i].split('y')[1].split('_')[1])
        z = int(name_list[length-1-i].split('z')[1].split('_')[1].split('.')[0])
        if (abs(x - target_x) <= 100 and abs(y - target_y) <= 100 and abs(z - target_z)<=100):
            if(x!=target_x and y!=target_y and z!=target_z):
                return name_list[length-1-i]

    return -1

def find_neighbor_seq(nei_name, seq_list,seq_num):
    for seq_name in seq_list:
        if(seq_name[seq_num-1] == nei_name):
            return seq_name
    return -1


def rnn_seq_sample_name_new(seq):
    path = 'F:\\004Vaa3d\\004feature\\17545\\tif_sample_name_17545.txt'

    # 读文本数据
    f = open(path)

    ftextlist = f.readlines()
    re = []
    re_seq = np.empty(0)

    memory_list_name = []
    neuron_num, pre_neuron_num = 1,1
    count = 0
    for text in ftextlist:
        print('complete:',str(count+1),'/',str(len(ftextlist)))
        tem = text.split('\t')
        tem_name = tem[1]
        neuron_num = int(tem_name.split('_')[0])

        neighbor_name = find_neighbor(memory_list_name,tem_name)
        if(neighbor_name == -1):
            for i in range(seq):
                re_seq = np.append(re_seq,tem_name)
            re.append(re_seq)
        else:
            neighbor_seq = find_neighbor_seq(neighbor_name,re,seq)
            for i in range(seq-1):
                re_seq = np.append(re_seq, neighbor_seq[i+1])
            re_seq = np.append(re_seq, tem_name)
            re.append(re_seq)

        re_seq = np.empty(0)

        if(neuron_num == pre_neuron_num):
            memory_list_name.append(tem_name)
        else:
            memory_list_name = []

        pre_neuron_num = neuron_num
        count += 1

    return re

def rnn_seq_sample_name_325(seq):
    path = 'F:\\004Vaa3d\\004feature\\17545\\tif_sample_name_17545.txt'

    # 读文本数据
    f = open(path)

    ftextlist = f.readlines()
    re = []
    re_seq = np.empty(0)

    neuron_num, pre_neuron_num = 1,1
    count = 0
    dict = {}
    list_name = []
    length = len(ftextlist)
    for text in ftextlist:
        count += 1
        tem = text.split('\t')
        tem_name = tem[1]
        neuron_num = int(tem_name.split('_')[0])

        if(neuron_num == pre_neuron_num and count != length):
            list_name.append(tem_name)
        else:
            if(count == length):
                list_name.append(tem_name)
            dict[str(pre_neuron_num)] = list_name
            list_name = []
            list_name.append(tem_name)
            pre_neuron_num = neuron_num

    count = 0
    for text in ftextlist:
        print('complete:',str(count+1),'/',str(len(ftextlist)))

        tem = text.split('\t')
        tem_name = tem[1]
        neuron_num = int(tem_name.split('_')[0])

        #序列中插入当前神经元名字
        re_seq = np.append(re_seq,tem_name)

        if(count == 18072):
            kk = 1
            pass
        list_name = copy.deepcopy(dict[str(neuron_num)])
        # list_name = dict[str(neuron_num)].copy()
        list_name.remove(tem_name)

        for i in range(seq-1):

            neighbor_name = find_neighbor(list_name,tem_name)
            if(neighbor_name == -1):
                re_seq = np.insert(re_seq, 0, tem_name)
            else:
                re_seq = np.insert(re_seq, 0, neighbor_name)
                list_name.remove(neighbor_name)

        re.append(re_seq)
        re_seq = np.empty(0)

        count += 1

    return re

def main():
    print('start...')

    # data = rnn_4_class_sample_name()
    # save_rnn_sample_txt(data)
    # print(data.shape,data.head(5))
    seq = 9
    data = rnn_seq_sample_name_new(seq)
    print(data[:3],len(data))
    save_rnn_seq_sample_txt(data,seq)
    print('end...')

if __name__=='__main__':
    main()

