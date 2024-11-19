import os
import torch
import numpy as np
import cv2


class PIPAL(torch.utils.data.Dataset):
    def __init__(self, dis_structure_path, dis_lbp_path, txt_file_name, transform, resize=False, size=None, flip=False):
        # 初始化 PIPAL 数据集类
        super(PIPAL, self).__init__()

        # 设置各个参数，包括路径、转换、是否翻转、调整大小等
        self.dis_structure_path = dis_structure_path  # 结构图像文件路径
        self.dis_lbp_path = dis_lbp_path  # LBP 图像文件路径
        self.txt_file_name = txt_file_name  # 包含图像文件名和分数的文本文件
        self.transform = transform  # 可能的图像转换
        self.flip = flip  # 是否进行图像翻转
        self.resize = resize  # 是否调整图像大小
        self.size = size  # 调整后的图像大小

        dis_files_data, score_data = [], []  # 初始化存储图像文件名和分数的列表

        # 从文本文件读取图像名称和对应的分数
        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                score, dis1 = line[:-1].split('  ')  # 按行读取分数和文件名
                dis = dis1[:-4]  # 去掉文件扩展名
                score = float(score)  # 转换分数为浮点数
                dis_files_data.append(dis)  # 添加文件名到列表
                score_data.append(score)  # 添加分数到列表

        score_data = np.array(score_data)  # 转换分数列表为 NumPy 数组
        score_data = score_data.astype('float').reshape(-1, 1)  # 将分数数据形状调整为列向量

        # 创建数据字典，包含图像文件名和分数
        self.data_dict = {'score_list': score_data, 'd_img_list': dis_files_data}

    def normalization(self, data):
        # 数据归一化函数
        range = np.max(data) - np.min(data)  # 计算数据范围
        return (data - np.min(data)) / range  # 归一化到 [0, 1]

    def __len__(self):
        # 返回数据集中样本的数量
        return len(self.data_dict['d_img_list'])

    def __getitem__(self, idx):
        # 从数据字典中获取当前索引的图像名称
        d_img_name = self.data_dict['d_img_list'][idx]
        # 加载对应的结构图像，格式为 BMP
        d_structure_img = cv2.imread(os.path.join(self.dis_structure_path, d_img_name+'.bmp'), cv2.IMREAD_COLOR)

        # 将加载的图像从 BGR 格式转换为 RGB 格式
        d_structure_img = cv2.cvtColor(d_structure_img, cv2.COLOR_BGR2RGB)

        # 如果设置了翻转，则对图像进行左右翻转
        if self.flip:
            d_structure_img = np.fliplr(d_structure_img).copy()

        # 如果设置了调整大小，则根据指定的大小调整图像
        if self.resize:
            d_structure_img = cv2.resize(d_structure_img, self.size)

        # 转换图像数据为浮点数并归一化到 [0, 1]
        d_structure_img = np.array(d_structure_img).astype('float32') / 255

        # 进行标准化处理，使数据中心化到 [-1, 1]
        d_structure_img = (d_structure_img - 0.5) / 0.5

        # 将图像的维度从 (H, W, C) 转换为 (C, H, W)
        d_structure_img = np.transpose(d_structure_img, (2, 0, 1))

        # 加载对应的 LBP 图像
        d_lbp_img = cv2.imread(os.path.join(self.dis_lbp_path, d_img_name + ".bmp"))

        # 对 LBP 图像进行处理（省略了颜色转换等）
        if self.flip:
            d_lbp_img = np.fliplr(d_lbp_img).copy()
        if self.resize:
            d_lbp_img = cv2.resize(d_lbp_img, self.size)

        # 转换 LBP 图像数据为浮点数并归一化到 [0, 1]
        d_lbp_img = np.array(d_lbp_img).astype('float32') / 255

        # 进行标准化处理
        d_lbp_img = (d_lbp_img - 0.5) / 0.5
        # 将 LBP 图像的维度从 (H, W, C) 转换为 (C, H, W)
        d_lbp_img = np.transpose(d_lbp_img, (2, 0, 1))

        # 获取当前样本的评分标签
        score = self.data_dict['score_list'][idx]

        # 构建样本字典，包含图像和标签
        sample = {
            'd_structure_img_org': d_structure_img,  # 结构图像
            'd_lbp_img_org': d_lbp_img,  # LBP 图像
            'score': score,  # 评分标签
            'd_img_name': d_img_name  # 图像名称
        }

        # 如果存在任何转换，则应用于样本
        if self.transform:
            sample = self.transform(sample)

        # 返回处理后的样本字典
        return sample

