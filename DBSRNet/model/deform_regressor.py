import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.deform_conv import DeformConv2d


class deform_fusion(nn.Module):
    def __init__(self, opt, in_channels=256 * 3, out_channels=256 * 3):
        super().__init__()
        # 定义隐藏层通道数
        self.d_hidn = 512

        # 定义一个卷积序列，用于特征提取和下采样
        self.conv1 = nn.Sequential(
            # 第一个卷积层：输入通道数为out_channels，输出通道数为d_hidn
            nn.Conv2d(in_channels=out_channels, out_channels=self.d_hidn, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),  # 激活函数
            # 第二个卷积层：输入通道数为d_hidn，输出通道数为out_channels
            nn.Conv2d(in_channels=self.d_hidn, out_channels=out_channels, kernel_size=3, padding=1, stride=2)
        )

    def forward(self, cnn_feat):
        # 输入特征cnn_feat通过卷积序列处理
        cnn_feat = self.conv1(cnn_feat)
        # 返回处理后的特征
        return cnn_feat

class CustomFusion(nn.Module):
    def __init__(self, opt,in_channels=2048, out_channels=2304):
        super().__init__()
        # 定义隐藏层通道数
        self.d_hidn = 512

        # 定义一个卷积序列，用于特征提取和上采样
        self.conv1 = nn.Sequential(
            # 第一个卷积层：输入通道数为in_channels，输出通道数为d_hidn
            nn.Conv2d(in_channels=in_channels, out_channels=self.d_hidn, kernel_size=3, padding=1),
            nn.ReLU(),  # 激活函数
            # 转置卷积：将特征图上采样到14x14
            nn.ConvTranspose2d(in_channels=self.d_hidn, out_channels=out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        # 输入特征x通过卷积序列处理
        x = self.conv1(x)
        # 返回处理后的特征
        return x



class Pixel_Prediction(nn.Module):  # 双分支是3840+768,resnet50阶段4的六个cnn是5120
    def __init__(self, inchannels=3840+3840, outchannels=256, d_hidn=1024):
        super().__init__()
        self.down_channel = nn.Conv2d(inchannels, outchannels, kernel_size=1)
        self.conv_attent = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
        )
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.linear = nn.Sequential(
            nn.Linear(3840, 2048), nn.ReLU(), nn.Dropout(),
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(1024, 1)
        )


    def forward(self, f_dis, cnn_dis):
        f_dis = torch.cat((f_dis, cnn_dis), 1)
        f_dis = self.down_channel(f_dis)
        f = self.conv(f_dis)
        w = self.conv_attent(f_dis)
        pred = (f * w).sum(dim=2).sum(dim=2) / w.sum(dim=2).sum(dim=2)
        return pred