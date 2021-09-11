import numpy as np

def compute_swc_distance_iterate_all():
    path = 'F:\\004Vaa3d\\002Data\\17302_neurons_swc\\17302_00087.swc_resampled.swc'
    # 读文本数据
    f = open(path)
    ftextlist = f.readlines()

    pre_x,pre_y,pre_z = 0,0,0
    count = 0
    for text in ftextlist:
        tem = text.split(' ')[0]
        count += 1
        if(tem != '#'):
            # pre_x = float(text.split(' ')[2])
            # pre_y = float(text.split(' ')[3])
            # pre_z = float(text.split(' ')[4])
            break
    del ftextlist[0:count]

    flag = 1
    for text1 in ftextlist:

        x = float(text1.split(' ')[2])
        y = float(text1.split(' ')[3])
        z = float(text1.split(' ')[4])
        for text2 in ftextlist:
            x2 = float(text2.split(' ')[2])
            y2 = float(text2.split(' ')[3])
            z2 = float(text2.split(' ')[4])
            if(abs(x-x2)>0 and abs(y-y2)>0 and abs(z-z2)>0 ):
                if(abs(x-x2)<=100 or abs(y-y2)<=100 or abs(z-z2)<=100 ):
                    flag = 0
        if(flag == 1):
            print(abs(x-x2),abs(y-y2),abs(z-z2))
        else:
            flag = 1

def compute_swc_sort_distance():
    path = 'F:\\004Vaa3d\\002Data\\17302_neurons_swc\\17302_00087.swc_resampled.swc_sorted_0.swc'
    # 读文本数据
    f = open(path)
    ftextlist = f.readlines()

    #删除注释行
    count = 0
    for text in ftextlist:
        tem = text.split(' ')[0]
        if(tem != '#'):
            break
        count += 1

    del ftextlist[0:count]

    print(ftextlist[0])
    for text in ftextlist:
        x = float(text.split(' ')[2])
        y = float(text.split(' ')[3])
        z = float(text.split(' ')[4])
        par = int(text.split(' ')[6])
        if(par!=-1):
            par_x = float(ftextlist[par-1].split(' ')[2])
            par_y = float(ftextlist[par-1].split(' ')[3])
            par_z = float(ftextlist[par-1].split(' ')[4])
            if (abs(x - par_x) > 1 or abs(y - par_y) > 1 or abs(z - par_z) > 1):
                print(par,abs(x - par_x),abs(y - par_y),abs(z - par_z))



def main():
    print('start...')
    compute_swc_sort_distance()
    print('end...')

if __name__=='__main__':
    main()

