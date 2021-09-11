import os


def func():
    swc_path = 'F:\\004Vaa3d\\002Data\\17302_delete_rag_few_seg\\method2_auto_norag_nofew_block\\'
    label_path = 'F:\\004Vaa3d\\004feature\\machine_label\\method2_1to0_classification_label_1115_clf176.txt'

    file_list = os.listdir(swc_path)
    #读文本数据
    f = open(label_path)
    ftextlist = f.readlines()
    #删除注释行
    del ftextlist[0]
    dic = {}
    for text in ftextlist:
        name = text.split('\t')[1]
        label = text.split('\t')[2]
        label = label[:3]
        dic[name] = label

    for name in file_list:
        sr_path = swc_path + name
        tem = name.split('.')
        new_name = tem[0] + '_l' + dic[name] + '.' + tem[1]
        tar_path = swc_path +new_name
        os.rename(sr_path,tar_path)
    pass

def main():
    print('start...')
    func()
    print('end...')

if __name__=='__main__':
    main()
