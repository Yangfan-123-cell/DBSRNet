# ref_files_data, dis_files_data, score_data = [], [], []
# with open("F:\\yanjiusheng_changyong\\shujuji\\gai-tu\\gaitu.txt", 'r') as listFile:
#         for line in listFile:
#                 # dis1, score = line[:-1].split(',')#[:-1]：就是去除了这行文本的最后一个字符（换行符）后剩下的部分。
#                 # #dis = dis[:-1]
#                 # ref, dis = dis1.split(' ') #切片符号，b = a[i:j] 表示复制a[i]到a[j-1]，以生成新的list对象
#             score, dis1 = line[:-1].split(' ')
#             print(score)
#             print(dis1)
#             ref = dis1[:5]
#             dis = dis1[:-4]
#
#             print(ref)
#             print(dis)



                #当i缺省时，默认为0，当j缺省时，默认为len(alist),当i,j都缺省时，a[:]就相当于完整复制一份a了
                # score = float(score)
                # ref_files_data.append(ref)
                # dis_files_data.append(dis)
                # score_data.append(score)
import cv2
import os
from torchvision import transforms
# r_lbp_img = cv2.imread(os.path.join("F:\\yanjiusheng_changyong\\shujuji\\gai-tu\\source-structure-images\\",'img01.bmp'))
# print(r_lbp_img.shape)
# r_lbp_img = cv2.cvtColor(r_lbp_img, cv2.COLOR_BGR2RGB)
# print(r_lbp_img.shape)
# import cv2
# import os
# d_img_name = 'img01'
# d_structure_img = cv2.imread(os.path.join('F:\\yanjiusheng_changyong\\shujuji\\gai-tu\\source-structure-images\\', d_img_name+".bmp"), cv2.IMREAD_COLOR)
# d_lbp_img = cv2.imread(
#     os.path.join('F:\\yanjiusheng_changyong\\shujuji\\gai-tu\\shizhen-lbp-images\\', "img01_2_01" + ".jpg"))
# #d_lbp_img = cv2.cvtColor(d_lbp_img, cv2.COLOR_BGR2RGB)
# print('d_lbp_img:',d_lbp_img.shape)
# d_lbp_img = transforms.ToTensor()(d_lbp_img)
# print('d_lbp_img:',d_lbp_img.shape)
# r_lbp_img = cv2.imread(os.path.join("F:\\yanjiusheng_changyong\\shujuji\\gai-tu\\source-lbp-images\\", 'img01'+".jpg"))
# #r_lbp_img = cv2.cvtColor(r_lbp_img, cv2.COLOR_BGR2RGB)
# print('r_lbp_img:',r_lbp_img.shape)
# r_lbp_img = transforms.ToTensor()(r_lbp_img)
# print('r_lbp_img:',r_lbp_img.shape)
# d_structure_img = cv2.imread(r'F:\yanjiusheng_changyong\shujuji\gai-tu\super-resolved-images\img01_2_01.bmp', cv2.IMREAD_COLOR)
# d_structure_img = cv2.cvtColor(d_structure_img, cv2.COLOR_BGR2RGB)
# print('d_structure_img:',d_structure_img.shape)
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.ops.deform_conv import DeformConv2d

conv1 = nn.Sequential(
            nn.Conv2d(in_channels=768,out_channels=512, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=768, kernel_size=3, padding=1, stride=2)
        )


cnn_ref = np.zeros((768, 56, 56))


b=conv1(cnn_ref)
print("b:",b.shape)


