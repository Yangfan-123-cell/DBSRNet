from tqdm import tqdm 
import os
import torch 
import numpy as np
import logging
from scipy.stats import spearmanr, pearsonr
import timm
from timm.models.vision_transformer import Block
from timm.models.resnet import Bottleneck

from scipy import stats
#这个能跑出一样的结果

from torch.utils.data import DataLoader

from utils.util import setup_seed,set_logging,SaveOutput
from script.extract_feature import get_resnet_feature, get_vit_feature
from options.train_options import TrainOptions
from model.deform_regressor import deform_fusion, Pixel_Prediction

from data.pipal import PIPAL
from utils.process_image import ToTensor, RandHorizontalFlip, RandCrop, crop_image, Normalize, five_point_crop
from torchvision import transforms

class Train:
    def __init__(self, config):
        self.opt = config
        self.create_model()
        self.init_saveoutput()
        self.init_data()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam([
        {'params': self.regressor.parameters(), 'lr': self.opt.learning_rate,'weight_decay':self.opt.weight_decay}, 
        {'params': self.deform_net.parameters(),'lr': self.opt.learning_rate,'weight_decay':self.opt.weight_decay}
        ])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.opt.T_max, eta_min=self.opt.eta_min)
        self.train()

    def create_model(self):
        self.resnet50 =  timm.create_model('resnet50',pretrained=True).cuda()
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
        print('init_saveoutput调用完毕！')
    
    def init_data(self):
        train_dataset = PIPAL(
            dis_structure_path=self.opt.train_dis_structure_path,
            dis_lbp_path=self.opt.train_dis_lbp_path,
            txt_file_name=self.opt.train_list,
            transform=transforms.Compose(
                [
                    RandCrop(self.opt.crop_size, self.opt.num_crop),
                    #Normalize(0.5, 0.5),
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
        checkpoint = torch.load(os.path.join(models_dir,"qads3", "best_cnn" + ".pth"))
        self.regressor.load_state_dict(checkpoint['regressor_model_state_dict'])

    def train_epoch(self, epoch):
        losses = []
        self.regressor.train()
        self.deform_net.train()
        self.vit.eval()
        self.resnet50.eval()

        pred_epoch = []
        labels_epoch = []
        
        for data in tqdm(self.train_loader):
            d_structure_img_org = data['d_structure_img_org'].cuda()
            d_lbp_img_org = data['d_lbp_img_org'].cuda()

            labels = data['score']
            labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()

            _x = self.vit(d_structure_img_org)
            vit_dis = get_vit_feature(self.save_output)
            self.save_output.outputs.clear()


            B, N, C = vit_dis.shape
            if self.opt.patch_size == 8:
                H,W = 28,28
            else:
                H,W = 14,14
            assert H*W==N 

            vit_dis = vit_dis.transpose(1, 2).view(B, C, H, W)

            _ = self.resnet50(d_structure_img_org)
            cnn_dis = get_resnet_feature(self.save_output)   #0,1,2都是[B,256,56,56]
            self.save_output.outputs.clear()
            print(cnn_dis.shape)
            cnn_dis = self.deform_net(cnn_dis)

            pred =self.regressor(vit_dis)


            self.optimizer.zero_grad()
            loss = self.criterion(torch.squeeze(pred), labels)
            losses.append(loss.item())

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)

        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rmse = np.sqrt(((np.squeeze(pred_epoch)-np.squeeze(labels_epoch))**2).mean())
        rho_k, _ = stats.kendalltau(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        ret_loss = np.mean(losses)
        print("第%s 个epoch得到的训练结果："%epoch)

        print('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4} / RMSE:{:.4} / KROCC:{:.4}'.format(epoch , ret_loss, rho_s, rho_p,rmse,rho_k))
        logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4} / RMSE:{:.4} /KROCC:{:.4}'.format(epoch , ret_loss, rho_s, rho_p,rmse,rho_k))

        return ret_loss, rho_s, rho_p, rmse, rho_k
    
    def train(self):
        best_srocc = 0
        best_plcc = 0


        a =[]
        b =[]
        for epoch in range(self.opt.load_epoch+2, self.opt.n_epoch):

            ret_loss, rho_s, rho_p, rmse, rho_k = self.train_epoch(epoch)
            a.append(ret_loss)
        
            if epoch == 200 :
                losses, rho_s, rho_p =self.eval_epoch(epoch)

                self.save_model( epoch, "cviu_200_vit.pth", losses, rho_s, rho_p)

            
    
    def eval_epoch(self, epoch):
        with torch.no_grad():
            losses = []
            self.regressor.train()
            self.vit.eval()

            pred_epoch = []
            labels_epoch = []

            for data in tqdm(self.val_loader):
                d_structure_img_org = data['d_structure_img_org'].cuda()
                d_lbp_img_org = data['d_lbp_img_org'].cuda()

                labels = data['score']
                labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
                pred = 0
                
                for i in range(self.opt.n_ensemble):
                    
                    b, c, h, w = d_structure_img_org.size()

                    if self.opt.n_ensemble > 9:
                        new_h = config.crop_size
                        new_w = config.crop_size
                        top = np.random.randint(0, h - new_h)
                        left = np.random.randint(0, w - new_w)

                        d_structure_img = d_structure_img_org[:,:, top: top+new_h, left: left+new_w]
                        d_lbp_img = d_lbp_img_org[:,:, top: top+new_h, left: left+new_w]
                    elif self.opt.n_ensemble ==1:
                        d_structure_img = d_structure_img_org
                        d_lbp_img = d_lbp_img_org
                    else :
                        d_structure_img, d_lbp_img = five_point_crop(i, d_structure_img=d_structure_img_org, d_lbp_img=d_lbp_img_org, config=self.opt)

                    _x = self.vit(d_structure_img)
                    vit_dis = get_vit_feature(self.save_output)
                    self.save_output.outputs.clear()


                    B, N, C = vit_dis.shape
                    if self.opt.patch_size == 8:
                        H,W = 28,28
                    else:
                        H,W = 14,14
                    assert H*W==N

                    vit_dis = vit_dis.transpose(1, 2).view(B, C, H, W)


                    pred += self.regressor(vit_dis)


                pred /= self.opt.n_ensemble
                pred_batch_numpy = pred.data.cpu().numpy()
                labels_batch_numpy = labels.data.cpu().numpy()
                pred_epoch = np.append(pred_epoch, pred_batch_numpy)
                labels_epoch = np.append(labels_epoch, labels_batch_numpy)
            
            rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
            rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
            rmse = np.sqrt(((np.squeeze(pred_epoch)-np.squeeze(labels_epoch))**2).mean())
            rho_k, _ = stats.kendalltau(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
            print("第%s个epoch后得到的验证结果："%epoch)

            print(' test:{} / SRCC:{:.4} / PLCC:{:.4} / RMSE:{:.4} / KROCC:{:.4}'.format(epoch ,  rho_s, rho_p,rmse,rho_k))
            return np.mean(losses), rho_s, rho_p

                

    
    

    def save_model(self, epoch, weights_file_name, loss, rho_s, rho_p):
        print('-------------saving weights---------')
        weights_file = os.path.join(self.opt.checkpoints_dir, weights_file_name)
        torch.save({
            'epoch': epoch,
            'regressor_model_state_dict': self.regressor.state_dict(),
            #'deform_net_model_state_dict': self.deform_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss
        }, weights_file)
        logging.info('Saving weights and model of epoch{}, SRCC:{}, PLCC:{}'.format(epoch, rho_s, rho_p))

if __name__ == '__main__':
    config = TrainOptions().parse()
    print('config:',config)
    config.checkpoints_dir = os.path.join(config.checkpoints_dir, config.name)
    print('config.checkpoints_dir:',config.checkpoints_dir)
    setup_seed(config.seed)
    set_logging(config)
    Train(config)
    