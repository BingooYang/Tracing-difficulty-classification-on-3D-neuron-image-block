import  os
import  numpy as np
import sys
import  xlrd

def read_test_label(path):
    data = xlrd.open_workbook(path)
    # 打开sheet1
    table = data.sheet_by_index(0)
    method2_lable = np.zeros((0))
    name = np.array((0))

    for i in range(len(table.col_values(9))):
        # 前两行是注释，从第三行开始写入
        if (i > 1):
            if (table.col_values(9)[i] != 0 and table.col_values(9)[i] != 1 and table.col_values(9)[i] != 2 and
                    table.col_values(9)[i] != -1):
                break
            if (table.col_values(9)[i] >= 0):
                method2_lable = np.append(method2_lable, table.col_values(9)[i])
                name = np.append(name, table.col_values(1)[i])

    name = np.delete(name, 0)
    return name.reshape((len(name), 1)), method2_lable.reshape(len(method2_lable), 1)


def read_train_label(path):
    # 读文本数据
    f = open(path, encoding='GB2312')
    ftextlist = f.readlines()
    # 删除注释行
    ftextlist.pop(0)
    label_name = np.array(0)
    label_data = np.zeros(0)
    for s in ftextlist:
        str = ''.join(s)
        label_name = np.append(label_name, (str.split('\t')[1]))
    label_name = np.delete(label_name, 0)
    for s in ftextlist:
        str = ''.join(s)
        label_data = np.append(label_data, (int(str.split('\t')[2][0])))
    #        data_x = data_x.reshape(len(data_x),1)

    return label_name.reshape((len(label_name), 1)), label_data.reshape((len(label_data), 1))


def image_label_backname():
    path_img_agumentation = "/home/zhang/disk2/001yangbin/002vaa3d_img_samples/test_image"
    # path_img = "/home/zhang/disk2/001yangbin/002vaa3d_img_samples/sample_6.27/outimg_good_block"
    path_train_label = "F:/004Vaa3d/Code_python/machine_label/method2_1to0_classification_label_1115_clf176.txt"
    path_test_label = "F:/004Vaa3d/003label/001_latest_label_1113.xlsx"

    label_train_name,label_train_data = read_train_label(path_train_label)
    label_test_name,label_test_data = read_test_label(path_test_label)

    print(label_train_data.shape,label_test_data.shape)
    #除去人工标签部分
    label_train_name = label_train_name[label_test_name.shape[0]:]
    label_all_name = np.concatenate((label_test_name,label_train_name),axis=0)
    label_train_data = label_train_data[label_test_data.shape[0]:]
    label_all_data = np.concatenate((label_test_data,label_train_data),axis=0)

    images_name_list = os.listdir(path_img_agumentation)
    #排序
    images_name_list.sort(key=lambda x: int(x[0:8]))

    count = 0
    for j in range(label_all_name.shape[0]):
        for i in range(len(images_name_list)):
            tem = images_name_list[i].split('.')[0].split('r')[0]
            tem = tem[:(len(tem) - 1)]
            if(label_all_name[j][0].split('.')[0] == tem):
                # print(i)
                images_name_new = images_name_list[i].split('l')[0] + 'l' + str(label_all_data[j][0]) +'.tif'
                #重新命名
                os.rename(path_img_agumentation+'/'+images_name_list[i], path_img_agumentation+'/'+images_name_new)
                # print(images_name_list[i])
                break
            elif (i == (len(images_name_list) - 1)):
                print("error,can not find march file or end...")
                sys.exit(0)



def main():
    # path_img = "/disk1/001yangbin/002vaa3d_img_samples/data agumentation/outimg_good_block_agumentation_90"

    # data,label = tifffile_to_read_2(path_img)
    # print(label[:20])
    # test_save_tiff()
    image_label_backname()

if __name__ == '__main__':
    main()



