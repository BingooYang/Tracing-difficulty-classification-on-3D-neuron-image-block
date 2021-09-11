import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter

def compute_save_tree_num():
    path = 'F:/004Vaa3d/002Data/001block_10/001no_rag_few_seg/method2_auto_norag_nofew_block_swc_104/'
    save_path = 'F:/004Vaa3d/004feature/tree_num/tree_num_1_14.txt'
    file_list = os.listdir(path)
    count = 0
    f2 = open(save_path, mode='w')
    f2.write('#### n    name    tree_num \n')
    for name in file_list:
        count+=1
        file_path = path + name
        tree_num = 0

        # 读文本数据
        f = open(file_path)
        ftextlist = f.readlines()
        # 删除注释行
        del ftextlist[0:3]
        # ftextlist.pop(0)
        for text in ftextlist:
            tem = text.split(' ')
            tem = tem[len(tem)-1]
            if(tem=='-1\n'):
                tree_num+=1

        f2.writelines([str(count),'\t',name,'\t',str(tree_num),'\n'])

def read_tree_num(path):
    # 读文本数据
    f = open(path)
    ftextlist = f.readlines()
    ftextlist.pop(0)
    data = np.zeros(len(ftextlist))

    count = 0
    for text in ftextlist:
        tem = text.split('\t')
        tem = tem[len(tem)-1]
        data[count] = int(tem)
        count += 1

    return data

def analysis():
    file = 'F:/004Vaa3d/004feature/tree_num/tree_num_1_14.txt'
    data = read_tree_num(file)
    print(data.shape)
    countDict = Counter(data)
    # print(countDict[:5])
    countDict = sorted(countDict.items(), key=lambda a: a[0],reverse=False)
    print(countDict[:5])
    # print(countDict)
    countDict = np.array(countDict)

    #画图
    x = countDict[:,0]
    y = countDict[:,1]
    plt.xticks([])
    plt.bar(x, y,tick_label=x)
    plt.title("Tree num analysis")  # 图形标题
    plt.xlabel("tree num")  # x轴名称
    plt.ylabel("frequency")  # y 轴名称
    for m, n in zip(x, y):
        plt.text(m + 0.05, n + 0.05, '%.0f' % n, ha='center', va='bottom')
    plt.show()

def main():
    print('start...')
    # compute_save_tree_num()
    analysis()
    print('end...')

if __name__=='__main__':
    main()