import torch
import numpy as np


def crop_image(top, left, new_h, new_w, img=None):
    b, c, h, w = img.shape
    tmp_img = img[ : , : , top: top + new_h, left: left + new_w]
    return tmp_img

class RandCrop(object):
    def __init__(self, patch_size, num_crop):
        self.patch_size = patch_size
        self.num_crop = num_crop
        
    def __call__(self, sample):
        # r_img : C x H x W (numpy)
        d_structure_img = sample['d_structure_img_org']
        d_lbp_img = sample['d_lbp_img_org']
        score = sample['score']
        d_img_name = sample['d_img_name']

        c, h, w = d_structure_img.shape
        new_h = self.patch_size
        new_w = self.patch_size
        #ret_r_structure_img = np.zeros((c, self.patch_size, self.patch_size))
        #print("ret_r_structure_img:",ret_r_structure_img.shape)
        ret_d_structure_img = np.zeros((c, self.patch_size, self.patch_size))
        #ret_r_lbp_img = np.zeros((c, self.patch_size, self.patch_size))
        ret_d_lbp_img = np.zeros((c, self.patch_size, self.patch_size))
        #print("ret_d_lbp_img:",ret_d_lbp_img.size)
        for _ in range(self.num_crop):
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            #tmp_r_structure_img = r_structure_img[:, top: top + new_h, left: left + new_w]
            #print("tmp_r_structure_img:",tmp_r_structure_img.shape)
            tmp_d_structure_img = d_structure_img[:, top: top + new_h, left: left + new_w]
            #tmp_r_lbp_img = r_lbp_img[:, top: top + new_h, left: left + new_w]
            tmp_d_lbp_img = d_lbp_img[:, top: top + new_h, left: left + new_w]
            #print("tmp_d_lbp_img:",tmp_d_lbp_img.size)

            #ret_r_structure_img += tmp_r_structure_img
            #print("1.加成功了")
            ret_d_structure_img += tmp_d_structure_img
            #print("2.加成功了")
            #ret_r_lbp_img += tmp_r_lbp_img
            #print("3.加成功了")
            ret_d_lbp_img += tmp_d_lbp_img
            #print("4.加成功了")
            #ret_d_img += tmp_d_img
        #ret_r_structure_img /= self.num_crop
        ret_d_structure_img /= self.num_crop
        #ret_r_lbp_img /= self.num_crop
        ret_d_lbp_img /= self.num_crop

        sample = {

            'd_structure_img_org': ret_d_structure_img,

            'd_lbp_img_org': ret_d_lbp_img,
            'score': score, 'd_img_name':d_img_name
        }

        return sample

def five_point_crop(idx, d_structure_img, d_lbp_img, config):
    new_h = 224
    new_w = 224
    if len(d_structure_img.shape) == 3:   
        c, h, w = d_structure_img.shape
    else:
        b, c, h, w = d_structure_img.shape
    center_h = h // 2
    center_w = w // 2
    if idx == 0:
        top = 0
        left = 0
    elif idx == 1:
        top = 0
        left = w - new_w
    elif idx == 2:
        top = h - new_h
        left = 0
    elif idx == 3:
        top = h - new_h
        left = w - new_w
    elif idx == 4:
        top = center_h - new_h // 2
        left = center_w - new_w // 2
    elif idx == 5:
        left = 0
        top = center_h - new_h // 2
    elif idx == 6:
        left = w - new_w
        top = center_h - new_h // 2
    elif idx == 7:
        top = 0
        left = center_w - new_w // 2
    elif idx == 8:
        top = h - new_h
        left = center_w - new_w // 2
    if len(d_structure_img.shape) == 3:   
        d_structure_img_org = d_structure_img[: , top: top + new_h, left: left + new_w]
        d_lbp_img_org = d_lbp_img[: , top: top + new_h, left: left + new_w]
    else:
        d_structure_img_org = d_structure_img[ :,: , top: top + new_h, left: left + new_w]
        d_lbp_img_org = d_lbp_img[ :,: , top: top + new_h, left: left + new_w]
    return d_structure_img_org, d_lbp_img_org

class RandCrop_fivepoints(object):
    def __init__(self, patch_size, num_crop, config):
        self.patch_size = patch_size
        self.num_crop = num_crop
        self.config = config
    def __call__(self, sample):
        # r_img : C x H x W (numpy)
        r_img, d_img = sample['r_img_org'], sample['d_img_org']
        score = sample['score']

        for idx in range(5):
            r_img, d_img = five_point_crop(idx, d_img, r_img, self.config)
            if idx == 0:
                ret_r_img = r_img.unsqueeze(0)
                ret_d_img = d_img.unsqueeze(0)
            else:
                ret_r_img = torch.cat((ret_r_img, r_img.unsqueeze(0)),dim=0)
                ret_d_img = torch.cat((ret_d_img, d_img.unsqueeze(0)),dim=0)

        sample = {
            'r_img_org': ret_r_img,
            'd_img_org': ret_d_img,
            'score': score
        }

        return sample

class RandCrop_points(object):
    def __init__(self, patch_size, num_crop, config):
        self.patch_size = patch_size
        self.num_crop = num_crop
        self.config = config
    def __call__(self, sample):
        # r_img : C x H x W (numpy)
        r_img, d_img = sample['r_img_org'], sample['d_img_org']
        score = sample['score']
        d_img_name = sample['d_img_name']

        c, h, w = d_img.shape
        new_h = self.patch_size
        new_w = self.patch_size

        for idx in range(self.num_crop):
            if self.num_crop == 5 or self.num_crop == 9:
                r_img_org, d_img_org = five_point_crop(idx, d_img, r_img, self.config)
            else:
                top = np.random.randint(0, h - new_h)
                left = np.random.randint(0, w - new_w)
                r_img_org = r_img[:, top: top + new_h, left: left + new_w]
                d_img_org = d_img[:, top: top + new_h, left: left + new_w]
            if idx == 0:
                ret_r_img = r_img_org.unsqueeze(0)
                ret_d_img = d_img_org.unsqueeze(0)
            else:
                ret_r_img = torch.cat((ret_r_img, r_img_org.unsqueeze(0)),dim=0)
                ret_d_img = torch.cat((ret_d_img, d_img_org.unsqueeze(0)),dim=0)

        sample = {
            'r_img_org': ret_r_img,
            'd_img_org': ret_d_img,
            'score': score, 'd_img_name':d_img_name
        }

        return sample

class Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        # r_img: C x H x W (numpy)
        r_img, d_img = sample['r_img_org'], sample['d_img_org']
        score = sample['score']
        d_img_name = sample['d_img_name']

        r_img = (r_img - self.mean) / self.var
        d_img = (d_img - self.mean) / self.var

        sample = {'r_img_org': r_img, 'd_img_org': d_img, 'score': score, 'd_img_name':d_img_name}
        return sample


class RandHorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        d_structure_img = sample['d_structure_img_org']
        d_lbp_img = sample['d_lbp_img_org']
        score = sample['score']
        d_img_name = sample['d_img_name']
        prob_lr = np.random.random()
        # np.fliplr needs HxWxC
        if prob_lr > 0.5:
            d_structure_img = np.fliplr(d_structure_img).copy()
            #r_structure_img = np.fliplr(r_structure_img).copy()
            d_lbp_img = np.fliplr(d_lbp_img).copy()
            #r_lbp_img = np.fliplr(r_lbp_img).copy()
        
        sample = {
            #'r_structure_img_org': r_structure_img,
            'd_structure_img_org': d_structure_img,
            #'r_lbp_img_org': r_lbp_img,
            'd_lbp_img_org': d_lbp_img,
            'score': score, 'd_img_name':d_img_name
        }
        return sample


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        d_structure_img = sample['d_structure_img_org']
        d_lbp_img = sample['d_lbp_img_org']
        score = sample['score']
        d_img_name = sample['d_img_name']
        d_structure_img = torch.from_numpy(d_structure_img).type(torch.FloatTensor)
        #r_structure_img = torch.from_numpy(r_structure_img).type(torch.FloatTensor)
        d_lbp_img = torch.from_numpy(d_lbp_img).type(torch.FloatTensor)
        #r_lbp_img = torch.from_numpy(r_lbp_img).type(torch.FloatTensor)
        score = torch.from_numpy(score).type(torch.FloatTensor)
        sample = {
            #'r_structure_img_org': r_structure_img,
            'd_structure_img_org': d_structure_img,
            #'r_lbp_img_org': r_lbp_img,
            'd_lbp_img_org': d_lbp_img,
            'score': score, 'd_img_name':d_img_name
        }
        return sample