import torch
from torch import nn
from keras.layers import (Activation, BatchNormalization, Conv2D, Lambda,
                          MaxPooling2D, UpSampling2D, concatenate)


def get_resnet_feature(save_output):
    feat = torch.cat(
        (
            save_output.outputs[0],
            save_output.outputs[1],
            save_output.outputs[2]
            #save_output.outputs[7],
            # save_output.outputs[8],
            # save_output.outputs[9],
            # save_output.outputs[10],
            # save_output.outputs[11]
        ),
        dim=1
    )
    return feat
def get_resnet_feature_4(save_output):
    feat = save_output.outputs[7]   #返回1024的特征张量
    return feat

def get_resnet_feature_5(save_output):
    # 直接提取最后一层的特征张量
    feat = save_output.outputs[-1]  # 假设最后一层输出在outputs的最后一个位置
    return feat  # 返回2048的特征张量

def get_vit_feature(save_output):
    feat = torch.cat(
        (
            save_output.outputs[0][:,1:,:],
            save_output.outputs[1][:,1:,:],
            save_output.outputs[2][:,1:,:],
            save_output.outputs[3][:,1:,:],
            save_output.outputs[4][:,1:,:],
        ),
        dim=2
    )
    return feat


