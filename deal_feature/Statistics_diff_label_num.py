import numpy as np
import os

def diff_label():
    path = 'F:\\004Vaa3d\\004feature\\all_samples_name\\all_good_samples_name.txt'

    # 读文本数据
    f = open(path)
    ftextlist = f.readlines()

    ftextlist = ftextlist[2954:]
    num_label0 = 0
    num_label2 = 0
    for text in ftextlist:
        tem = text.split('\t')
        tem = tem[len(tem)-1]
        tem = int(tem)
        if(tem == 0):
            num_label0 += 1
        else:
            num_label2 += 1

    return num_label0,num_label2


num1, num2 = diff_label()
print(num1,num2)