import os

path = 'F:\\004Vaa3d\\004feature\\17545\\machine_lable\\method2_machine_label_clf176_17545.txt'
f = open(path)
filelist = f.readlines()
del filelist[0]
num1,num2 = 0,0
for a in filelist:
    lable = a.split('\t')[2]
    lable = lable[0]
    if(int(lable)==0):
        num1+=1
    else:
        num2+=1

print('lable=0,num:',num1,'label=2,num:',num2)
