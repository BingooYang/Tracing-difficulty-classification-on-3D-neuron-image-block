import numpy as np
# from sklearn.preprocessing import normalize
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt

def read_feasure(path, f_num, pos):
    # 读文本数据
    f = open(path)
    ftextlist = f.readlines()

    length = int(len(ftextlist) / f_num)
    data = np.zeros((length))
    #    print(data.shape)
    name = np.array(0)

    count = 0
    for i in range(length):
        tem = i*f_num + pos
        s = ''.join(ftextlist[tem])
        # print(ftextlist[tem])
        name = np.append(name, (s.split('\t')[0].split('\\')[6]))
        data[i] = float(s.split('\t')[2])


    # 删除第一行0
    name = np.delete(name, 0)
    return name, data

def analysis():
    file = 'F:\\004Vaa3d\\004feature\\LM\\method2_auto_norag_nofew_block_Lmeasure_10_label_0105.txt'
    name,data = read_feasure(file,10,1)
    print(name.shape,data.shape)
    countDict = Counter(data)
    # print(countDict[:5])
    countDict = sorted(countDict.items(), key=lambda a: a[0],reverse=False)
    print(countDict[:5])
    print(countDict)
    countDict = np.array(countDict)

    #画图
    x = countDict[:,0]
    y = countDict[:,1]
    plt.xticks([])
    plt.bar(x, y,tick_label=x)
    plt.title("L-measure N_branch")  # 图形标题
    plt.xlabel("branch num")  # x轴名称
    plt.ylabel("frequency")  # y 轴名称
    for m, n in zip(x, y):
        plt.text(m + 0.05, n + 0.05, '%.0f' % n, ha='center', va='bottom')
    plt.show()

def save_spe_branch_name(path,name,data,num):
    f = open(path, mode='w')

    lenth = len(name)

    count = 0
    for i in range(lenth):
        if(data[i]<=num):
            count+=1
            f.writelines([str(count),'\t',str(name[i]),'\t',str(data[i]),'\n'])


def get_spe_branch():
    file = 'F:\\004Vaa3d\\004feature\\LM\\method2_auto_norag_nofew_block_nosoma_Lmeasure_label_0109.txt'
    name,data = read_feasure(file,32,2)
    print(name.shape,data.shape)
    save_path = 'F:\\004Vaa3d\\001note\\错误样本分析\\spe_branch\\spe_branch_4.txt'
    save_spe_branch_name(save_path,name,data,4)


if __name__=="__main__":
    # analysis()
    get_spe_branch()
