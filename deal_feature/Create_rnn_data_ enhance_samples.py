import os

def func():
    path = 'F:\\004Vaa3d\\004feature\\RNN_sample\\seq_10_rnn_good_sample_name_1.23.txt'
    save_path = 'F:\\004Vaa3d\\004feature\\RNN_sample\\seq_10_rnn_augment_data_sample_name_3.30.txt'
    # 读文本数据
    f = open(path)
    ftextlist = f.readlines()
    ftextlist.pop(0)
    seq = 10
    agu_data = ['90','180','270']

    save_f = open(save_path, mode='w')
    s = u'####一行中最后一个是当前样本'
    save_f.write(s)
    save_f.write('\n')

    for text in ftextlist:
        save_f.writelines(text)

        s_list = text.split('\t')
        for agu in agu_data:
            new_list = []
            new_list.append(s_list[0])
            for m in range(1,seq+1):
                tem = s_list[m]
                new_name = tem.split('r')[0] + 'r_'+ agu + '_l' + tem.split('l')[1]
                new_list.append(new_name)

            for name in new_list:
                save_f.write(name)
                save_f.write('\t')
            save_f.write('\n')


def main():
    print('start...')
    func()
    print('end...')

if __name__=='__main__':
    main()