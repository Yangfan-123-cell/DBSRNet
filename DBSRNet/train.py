import logging
from scipy import stats
from torch.utils.data import DataLoader
from utils.util import setup_seed, set_logging, SaveOutput
from script.extract_feature import get_resnet_feature_4,get_resnet_feature_5,get_vit_feature,get_resnet_feature
from options.train_options import TrainOptions
from model.deform_regressor import deform_fusion, Pixel_Prediction
from data.pipal import PIPAL
from utils.process_image import ToTensor, RandHorizontalFlip, RandCrop, five_point_crop
from torchvision import transforms
import os
import timm
from timm.models.vision_transformer import Block
from timm.models.resnet import BasicBlock,Bottleneck

from torch import nn
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr, pearsonr, stats
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr, stats
from fvcore.nn import FlopCountAnalysis
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 定义卷积层 -> 批标准化 -> ReLU 的结构
class Conv2dBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, relu=True):
        super(Conv2dBnRelu, self).__init__()
        # 定义一个二维卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        # 定义批标准化层
        self.bn = nn.BatchNorm2d(out_channels)
        # 如果需要ReLU激活函数，则使用ReLU，否则使用恒等映射
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        # 前向传播过程，按顺序通过卷积层、批标准化层和激活函数
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class GEMPooling(nn.Module):
    def __init__(self, p=3):
        super(GEMPooling, self).__init__()
        self.p = p

    def forward(self, x):
        # x的形状是 (B, C, H, W)
        gem_out = torch.mean(x.view(x.size(0), x.size(1), -1) ** self.p, dim=2) ** (1. / self.p)
        return gem_out.unsqueeze(-1).unsqueeze(-1)  # 变为 (B, C, 1, 1)


# 定义BasicRFB模块（受启发自Receptive Field Block）
class BasicRFB(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(BasicRFB, self).__init__()
        # 缩放系数，用于控制残差项的比例
        self.scale = scale
        # 输出通道数
        self.out_channels = out_planes
        # 中间层的通道数设为输入通道的1/8
        inter_planes = in_planes // 8

        # branch0：第一分支，卷积 -> 膨胀卷积
        self.branch0 = nn.Sequential(
            Conv2dBnRelu(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),  # 1x1卷积降低维度
            Conv2dBnRelu(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)  # 3x3膨胀卷积
        )

        # branch1：第二分支，1x1卷积 -> 3x3卷积 -> 膨胀卷积
        self.branch1 = nn.Sequential(
            Conv2dBnRelu(in_planes, inter_planes, kernel_size=1, stride=1),  # 1x1卷积降低维度
            Conv2dBnRelu(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),  # 3x3卷积
            Conv2dBnRelu(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual + 1, dilation=visual + 1, relu=False)  # 膨胀卷积
        )

        # branch2：第三分支，1x1卷积 -> 3x3卷积 -> 3x3卷积 -> 膨胀卷积
        self.branch2 = nn.Sequential(
            Conv2dBnRelu(in_planes, inter_planes, kernel_size=1, stride=1),  # 1x1卷积降低维度
            Conv2dBnRelu(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),  # 3x3卷积
            Conv2dBnRelu((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),  # 3x3卷积
            Conv2dBnRelu(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2 * visual + 1, dilation=2 * visual + 1, relu=False)  # 膨胀卷积
        )

        # 对三个分支的输出进行通道整合
        self.ConvLinear = Conv2dBnRelu(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        # 使用1x1卷积调整维度以匹配残差
        self.shortcut = Conv2dBnRelu(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        # 最后的ReLU激活函数
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        # 获取各个分支的输出
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        # 将三个分支的输出在通道维度上进行拼接
        out = torch.cat((x0, x1, x2), 1)
        # 对拼接后的结果进行线性卷积
        out = self.ConvLinear(out)
        # 计算shortcut（残差）
        short = self.shortcut(x)
        # 残差相加，并进行缩放
        out = out * self.scale + short
        # 通过ReLU激活函数
        out = self.relu(out)

        return out




class Train:
    def __init__(self, config):
        self.opt = config
        self.create_model()
        self.init_saveoutput()
        self.init_data()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam([
            {'params': self.regressor.parameters(), 'lr': self.opt.learning_rate,
             'weight_decay': self.opt.weight_decay},
            {'params': self.deform_net.parameters(), 'lr': self.opt.learning_rate,
             'weight_decay': self.opt.weight_decay}
        ])
        # 示例调用
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.opt.T_max,
                                                                    eta_min=self.opt.eta_min)
        self.conv1x1_4 = nn.Conv2d(3840, 3840, kernel_size=1).to(device)
        self.conv1x1_5 = nn.Conv2d(1536, 1536, kernel_size=1).to(device)
        self.conv1x1_x = nn.Conv2d(768, 768, kernel_size=1).to(device)

        # 在构造函数中实例化 BasicRFB 模块
        self.rfb = BasicRFB(768, 3840,1,0.11,1).to(device)
        # 在Train类中使用GEMPooling
        self.gem_pool = GEMPooling(p=3)  # p可以根据需求调整
        self.train()

    def create_model(self):
        self.resnet50 = timm.create_model('resnet50', pretrained=True).cuda()
        if self.opt.patch_size == 8:
            self.vit = timm.create_model('vit_base_patch8_224',pretrained=True).cuda()
        else:
            self.vit = timm.create_model('vit_base_patch16_224',pretrained=True).cuda()
        self.deform_net = deform_fusion(self.opt).cuda()
        self.regressor = Pixel_Prediction().cuda()

    def init_saveoutput(self):
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.resnet50.modules():
            if isinstance(layer, Bottleneck):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                    handle = layer.register_forward_hook(self.save_output)
                    hook_handles.append(handle)
        print('开始训练！')

    def init_data(self):
        train_dataset = PIPAL(
            dis_structure_path=self.opt.train_dis_structure_path,
            dis_lbp_path=self.opt.train_dis_lbp_path,
            txt_file_name=self.opt.train_list,
            transform=transforms.Compose(
                [
                    RandCrop(self.opt.crop_size, self.opt.num_crop),
                    RandHorizontalFlip(),
                    ToTensor(),
                ]
            ),
        )
        val_dataset = PIPAL(
            dis_structure_path=self.opt.val_dis_structure_path,
            dis_lbp_path=self.opt.val_dis_lbp_path,
            txt_file_name=self.opt.val_list,
            transform=ToTensor(),
        )
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.opt.batch_size,
            num_workers=self.opt.num_workers,
            drop_last=True,
            shuffle=True
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.opt.batch_size,
            num_workers=self.opt.num_workers,
            drop_last=True,
            shuffle=False
        )

    def load_model(self):
        models_dir = self.opt.checkpoints_dir
        if os.path.exists(models_dir):
            if self.opt.load_epoch == -1:
                load_epoch = 0
                for file in os.listdir(models_dir):
                    if file.startswith("epoch_"):
                        load_epoch = max(load_epoch, int(file.split('.')[0].split('_')[1]))
                self.opt.load_epoch = load_epoch
                checkpoint = torch.load(os.path.join(models_dir, "epoch_" + str(self.opt.load_epoch) + ".pth"))
                self.regressor.load_state_dict(checkpoint['regressor_model_state_dict'])
                self.deform_net.load_state_dict(checkpoint['deform_net_model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
                loss = checkpoint['loss']
            else:
                found = False
                for file in os.listdir(models_dir):
                    if file.startswith("epoch_"):
                        found = int(file.split('.')[0].split('_')[1]) == self.opt.load_epoch
                        if found: break
                assert found, 'Model for epoch %i not found' % self.opt.load_epoch
        else:
            assert self.opt.load_epoch < 1, 'Model for epoch %i not found' % self.opt.load_epoch
            self.opt.load_epoch = 0

    def train_epoch(self, epoch):
        losses = []  # 用于存储每个batch的损失
        self.regressor.train()  # 设置回归器为训练模式
        self.deform_net.train()  # 设置变形网络为训练模式
        self.resnet50.eval()  # 设置ResNet50为评估模式
        # 获取并打印每个模型的参数数量
        total_params_regressor = sum(p.numel() for p in self.regressor.parameters() if p.requires_grad)
        total_params_deform_net = sum(p.numel() for p in self.deform_net.parameters() if p.requires_grad)
        total_params_resnet50 = sum(p.numel() for p in self.resnet50.parameters() if p.requires_grad)
        total_params = total_params_regressor + total_params_deform_net + total_params_resnet50
        print(f"Total parameters for this epoch: {total_params}")
        # 用于保存每个epoch的预测结果和标签
        pred_epoch = []
        labels_epoch = []

        # 遍历训练集
        for data in tqdm(self.train_loader):
            # 获取输入数据并转移到GPU
            d_structure_img_org = data['d_structure_img_org'].cuda()
            labels = data['score']
            labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()

            self.vit(d_structure_img_org)
            vit_dis = get_vit_feature(self.save_output)
            self.save_output.outputs.clear()

            B, N, C = vit_dis.shape
            if self.opt.patch_size == 8:
                H, W = 28, 28
            else:
                H, W = 14, 14
            assert H * W == N
            vit_dis = vit_dis.transpose(1, 2).view(B, C, H, W)

            self.resnet50(d_structure_img_org)
            cnn_dis = get_resnet_feature(self.save_output)  # [B, 256, 56, 56]
            self.save_output.outputs.clear()
            cnn_dis = self.deform_net(cnn_dis)
            cnn_dis = self.rfb(cnn_dis)
            cnn_dis = self.conv1x1_4(cnn_dis)

            # 关联性处理
            K = self.conv1x1_4(cnn_dis)  # [8, 768, 14, 14]
            V = self.conv1x1_4(cnn_dis)  # [8, 768, 14, 14]
            Q = vit_dis  # [8, 768, 14, 14]
            V_reshaped = V.view(8, 3840, 14 * 14)
            K_reshaped = K.view(8, 3840, 14 * 14).permute(0, 2, 1)  # [batch_size, 14*14, 1024]
            Q_reshaped = Q.view(8, 3840, 196).permute(0, 2, 1)  # [batch_size, 1, 1024]
            A = torch.bmm(Q_reshaped, K_reshaped.transpose(1, 2))  # [batch_size, 1, 14*14]
            A = A / torch.sqrt(torch.tensor(1024.0))
            A = F.softmax(A, dim=-1)  # [8, 1, 196]
            V1 = torch.bmm(A, V_reshaped.permute(0, 2, 1))  # [batch_size, 1, 1024]
            V1 = V1.view(8, 3840, 14, 14)
            V1 = self.conv1x1_4(V1)

            # 预测
            pred = self.regressor(V1, vit_dis)
            flops = FlopCountAnalysis(self.regressor, (V1, vit_dis))  # 输入 V1 和 vit_dis
            print(f"FLOPs for this batch: {flops.total()}")
            self.optimizer.zero_grad()  # 清空梯度
            loss = self.criterion(torch.squeeze(pred), labels)  # 计算损失
            losses.append(loss.item())

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # 保存结果
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)

        # 计算相关系数
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rmse = np.sqrt(((np.squeeze(pred_epoch) - np.squeeze(labels_epoch)) ** 2).mean())
        rho_k, _ = stats.kendalltau(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        ret_loss = np.mean(losses)
        print("第%s个epoch得到的训练结果：" % epoch)
        print('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4} / RMSE:{:.4} / KROCC:{:.4}'.format(
            epoch, ret_loss, rho_s, rho_p, rmse, rho_k))
        logging.info(
            'train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4} / RMSE:{:.4} / KROCC:{:.4}'.format(
                epoch, ret_loss, rho_s, rho_p, rmse, rho_k))

        # 只在第200轮保存结果
        if epoch == 200:
            df = pd.DataFrame({
                'Predicted Scores': pred_epoch,
                'Actual Scores': labels_epoch
            })
            df.to_excel("C:\\Users\\HP\\Desktop\\1.xlsx", index=False)

        return ret_loss, rho_s, rho_p, rmse, rho_k

    def train(self):
        # 遍历所有epoch进行训练
        for epoch in range(self.opt.load_epoch + 2, self.opt.n_epoch):
            ret_loss, rho_s, rho_p, rmse, rho_k = self.train_epoch(epoch)  # 训练当前epoch
            # 在第200个epoch时进行评估
            if epoch == 100:
                losses, rho_s, rho_p = self.eval_epoch(epoch)  # 评估当前模型性能
                self.save_model(epoch, 'qads.pth', losses, rho_s, rho_p)  # 保存模型


    def eval_epoch(self, epoch):
        print('开始调用eval函数')
        with torch.no_grad():
            losses = []
            self.regressor.train()
            self.deform_net.train()
            self.resnet50.eval()
            # save data for one epoch
            pred_epoch = []
            labels_epoch = []
            for data in tqdm(self.val_loader):
                pred = 0
                for i in range(self.opt.n_ensemble):
                    d_structure_img_org = data['d_structure_img_org'].cuda()
                    d_lbp_img_org = data['d_lbp_img_org'].cuda()
                    d_img_name = data['d_img_name']
                    labels = data['score']
                    labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
                    b, c, h, w = d_structure_img_org.size()
                    if self.opt.n_ensemble > 9:
                        new_h = config.crop_size
                        new_w = config.crop_size
                        top = np.random.randint(0, h - new_h)
                        left = np.random.randint(0, w - new_w)
                        d_structure_img = d_structure_img_org[:, :, top: top + new_h, left: left + new_w]
                        d_lbp_img = d_lbp_img_org[:, :, top: top + new_h, left: left + new_w]
                    elif self.opt.n_ensemble == 1:
                        d_structure_img = d_structure_img_org
                        d_lbp_img = d_lbp_img_org
                    else:
                        d_structure_img, d_lbp_img = five_point_crop(i, d_structure_img=d_structure_img_org,
                                                                     d_lbp_img=d_lbp_img_org, config=self.opt)

                    self.vit(d_structure_img)
                    vit_dis = get_vit_feature(self.save_output)
                    self.save_output.outputs.clear()

                    B, N, C = vit_dis.shape
                    if self.opt.patch_size == 8:
                        H, W = 28, 28
                    else:
                        H, W = 14, 14
                    assert H * W == N
                    vit_dis = vit_dis.transpose(1, 2).view(B, C, H, W)  # [8,768,14,14]

                    self.resnet50(d_structure_img)
                    cnn_dis = get_resnet_feature(self.save_output)  # [B, 256, 56, 56]
                    self.save_output.outputs.clear()
                    cnn_dis = self.deform_net(cnn_dis)  # [8,768,14,14]
                    cnn_dis = self.rfb(cnn_dis)  # [8,1536,14,14]
                    cnn_dis = self.conv1x1_4(cnn_dis)

                    # 关联性处理
                    K = self.conv1x1_4(cnn_dis)  # [8, 768, 14, 14]
                    V = self.conv1x1_4(cnn_dis)  # [8, 768, 14, 14]
                    Q = vit_dis  # [8, 768, 14, 14]
                    V_reshaped = V.view(8, 3840, 14 * 14)
                    K_reshaped = K.view(8, 3840, 14 * 14).permute(0, 2, 1)  # [batch_size, 14*14, 1024]
                    Q_reshaped = Q.view(8, 3840, 196).permute(0, 2, 1)  # [batch_size, 1, 1024]
                    A = torch.bmm(Q_reshaped, K_reshaped.transpose(1, 2))  # [batch_size, 1, 14*14]
                    A = A / torch.sqrt(torch.tensor(1024.0))
                    A = F.softmax(A, dim=-1)  # [8, 1, 196]
                    V1 = torch.bmm(A, V_reshaped.permute(0, 2, 1))  # [batch_size, 1, 1024]
                    V1 = V1.view(8, 3840, 14, 14)
                    V1 = self.conv1x1_4(V1)
                    # 预测
                    pred += self.regressor(V1, vit_dis)

                pred /= self.opt.n_ensemble
                loss = self.criterion(torch.squeeze(pred), labels)
                loss_val = loss.item()
                losses.append(loss_val)
                # save results in one epoch
                pred_batch_numpy = pred.data.cpu().numpy()
                labels_batch_numpy = labels.data.cpu().numpy()
                pred_epoch = np.append(pred_epoch, pred_batch_numpy)
                labels_epoch = np.append(labels_epoch, labels_batch_numpy)
            rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
            rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
            rmse = np.sqrt(((np.squeeze(pred_epoch) - np.squeeze(labels_epoch)) ** 2).mean())
            rho_k, _ = stats.kendalltau(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

            print("第%s个epoch后得到的验证结果：" % epoch)
            print(' test:{} / SROCC:{:.4} / PLCC:{:.4} / RMSE:{:.4} / KROCC:{:.4}'.format(epoch, rho_s, rho_p, rmse,
                                                                                         rho_k))
            return np.mean(losses), rho_s, rho_p

    def save_model(self, epoch, weights_file_name, loss, rho_s, rho_p):
        print('-------------saving weights---------')
        weights_file = os.path.join(self.opt.checkpoints_dir, weights_file_name)
        torch.save({
            'epoch': epoch,
            'regressor_model_state_dict': self.regressor.state_dict(),
            'deform_net_model_state_dict': self.deform_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss
        }, weights_file)
        print("Model will be saved to:", weights_file)
        logging.info('Saving weights and model of epoch{}, SRCC:{}, PLCC:{}'.format(epoch, rho_s, rho_p))


if __name__ == '__main__':
    config = TrainOptions().parse()
    print('config:', config)
    config.checkpoints_dir = os.path.join(config.checkpoints_dir, config.name)
    print('config.checkpoints_dir:', config.checkpoints_dir)
    setup_seed(config.seed)
    set_logging(config)
    Train(config)

