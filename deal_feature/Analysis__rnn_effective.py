import numpy as np

def read_rnn_sample():
    path = 'F:\\004Vaa3d\\004feature\\RNN_sample\\seq_2_rnn_good_sample_name_1.23.txt'
    # 读文本数据
    f = open(path)
    ftextlist = f.readlines()
    ftextlist = ftextlist[:5324]
    ftextlist.pop(0)
    total = len(ftextlist)
    correct1,correct2 = 0,0
    for text in ftextlist:
        s = text.split('\t')
        name1 = s[1]
        name2 = s[2]
        if(name1 == name2):
            continue
        label1 = int(s[1].split('l')[1].split('.')[0])
        label2 = int(s[2].split('l')[1].split('.')[0])
        if(label1 != 0):
            label1 = 1
        if(label2 != 0):
            label2 = 1
        if(label2 == label1 and label1 == 0) :
            correct1 += 1
        elif(label2 == label1 and label1 == 1):
            correct2 += 1
    print('acc:',float((correct1+correct2)/total), correct1, correct2, total)


def main():
    print('start...')
    read_rnn_sample()
    print('end...')

if __name__=='__main__':
    main()