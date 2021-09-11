import numpy as np
import os
import time

class SavePara():
    def __init__(self, pre_name, path = (os.getcwd() + 'para')):
        self.name = pre_name + 'para_' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '.txt'
        self.dir = path + self.name
        self.f = open(self.dir, mode='w')

        self.f.writelines(['model para','\n'])

        self.total_num = 0

    def writelines_para(self,data):
        self.f.writelines(['##############################', '\n'])
        for i in range(len(data)):
            self.f.write(str(data[i]))
            self.f.write('\t')
        self.f.write('\n')

    def model_scalar(self,name, data):
        self.f.writelines([name, '\t', str(data), '\t', '\n'])


    def predict_para(self, image_name, bag_label, predicted):

        for i in range(len(image_name)):
            # print(bag_label[i] ,predicted[i])
            if(bag_label[i] != predicted[i]):
                self.total_num += 1
                self.f.writelines([str(self.total_num), '\t', image_name[i], '\t', str(bag_label[i]), '\t', str(predicted[i]), '\n'])

    def clear_count_num(self):
        self.total_num = 0

# test_save = SavePara('/home/zhang/disk2/001yangbin/001vaa3d/001_img_classification_model/Code_python/Code_pytorch/Resnet_1113/para/')
#
# test_save.model_scalar('mdk',23.1)
#
# k =9
# test_save.writelines_para(['model',str(k)])


