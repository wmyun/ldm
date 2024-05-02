import os, yaml, pickle, shutil, tarfile, glob
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import PIL
import torch
from torch import nn
import numpy as np
import tifffile as tiff
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
from omegaconf import OmegaConf
import random

def tif_loader(path, trd):
    img = tiff.imread(path).astype(np.float32)
    if trd is None:
        #img = (img-img.min())/(img.max()-img.min())
        img = img/img.max()
    else:
        img = rescale(img, trd)
    return img

def rescale(img, trd):
    if trd is not None:
        img[img <= trd[0]] = trd[0]
        img[img >= trd[1]] = trd[1]
        img = (img - trd[0]) / (trd[1] - trd[0])
    return img

class ZEnhanceDataset(Dataset):
    '''
    img = original image
    cond_image = (fake yz) interpolated image
    mask = mask
    '''
    def __init__(self, data_root, data_len=None, mask_config={}, image_size=256, loader=tif_loader, mode='train', trd=None):
        self.path = data_root

        self.flist = os.listdir(self.path[0])
        if mode == 'train':
            random.shuffle(self.flist)
            split_ratio = 0.2
            num_elements_second_list = int(len(self.flist) * split_ratio)
            self.flist_set = self.set_base(num_elements_second_list)
        else:
            self.flist = sorted(self.flist)
            self.flist_set = self.flist[data_len[0]:data_len[1]]
            # self.p_flist = sorted(os.listdir(self.path[1]))
            # self.p_flist_set = self.p_flist[data_len[0]:data_len[1]]       

        self.tfs = A.Compose([
                A.RandomCrop(height=image_size, width=image_size, p=1.),
                ToTensorV2(p=1.0),
            ], additional_targets={'cond_image':'image'})#, 'pseudo':'image'
        self.loader = loader
        self.down_size = mask_config['down_size']
        self.direction = mask_config['direction']
        self.image_size = image_size
        self.trd = trd
        self.mode = mode

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist_set[index])

        img = self.loader('{}/{}'.format(self.path[0], file_name), self.trd)
        # if self.mode == 'train':
        #     pseudo = self.loader('{}/{}'.format(self.path[1], file_name), self.trd)
        # else:
        #     pseudo = self.loader('{}/{}'.format(self.path[1], str(self.p_flist_set[index])), self.trd)
        #     pseudo = pseudo[:self.image_size,:self.image_size]
        
        if self.mode == 'train':
            #cond_image = self.generate_yz_like(img)
            cond_image = self.loader('{}/{}'.format(self.path[1], file_name), self.trd)
        else:
            img = img[:self.image_size,:self.image_size]
            cond_image = img
        transformed = self.tfs(image=img,cond_image=cond_image)#, pseudo=pseudo)
        t_img = transformed['image']
        t_cond_img = transformed['cond_image']
        # t_pseudo = transformed['pseudo']
        
        #mask = self.get_mask()
        #mask_img = t_img*(1. - mask)# + mask*torch.randn_like(t_img)
        
        t_img = (t_img - 0.5) / 0.5
        t_cond_img = (t_cond_img - 0.5) / 0.5
        # t_pseudo = (t_pseudo - 0.5) / 0.5
        #mask_img = (mask_img - 0.5) / 0.5
        
        ret["image"] = t_img
        ret["cond_image"] = t_cond_img
        # ret["pseudo"] = t_pseudo
        #ret['mask_image'] = t_cond_img
        #ret["mask"] = mask
        ret["file_path_"] = file_name
        return ret

    def __len__(self):
        return len(self.flist_set)

    def get_mask(self):
        h = w = self.image_size
        block_num = int(h/self.down_size)
        block_size = self.down_size-1

        mask = np.zeros((h, w, 1), dtype='uint8')
        h_s = 1
        w_s = 1
        for i in range(block_num):
            if h_s > h or w_s > w:
                raise ValueError('mask shape is incorrect!!!')
            if self.direction == 'vertical':
                mask[:, w_s:w_s+block_size] = 1
                w_s += self.down_size
            else:
                mask[h_s:h_s+block_size,:] = 1
                h_s += self.down_size

        return torch.from_numpy(mask).permute(2,0,1)
    
    def generate_yz_like(self, img):
        h, w = img.shape
        block_num = int(h/self.down_size)

        yz_like_img = []
        h_s = 0
        w_s = 0
        for i in range(block_num):
            if h_s > h or w_s > w:
                raise ValueError('mask shape is incorrect!!!')
            if self.direction == 'vertical':
                yz_like_img.append(img[:, w_s:w_s+1])
                w_s += self.down_size
            else:
                yz_like_img.append(img[h_s:h_s+1,:])
                h_s += self.down_size
        yz_like_img = np.squeeze(yz_like_img,axis=1)
        yz_like_img = np.array(yz_like_img)

        # upsample
        up_img = torch.tensor(yz_like_img[np.newaxis, np.newaxis, :])
        up_img = torch.nn.Upsample(size=(img.shape[0],img.shape[1]),mode='bilinear')(up_img)
        up_img = torch.squeeze(up_img, dim=(0, 1)).numpy()
        
        return up_img

class ZEnhanceTrain(ZEnhanceDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_base(self, num_elements_second_list):
        return self.flist[:-num_elements_second_list]
    

class ZEnhanceValidation(ZEnhanceDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_base(self, num_elements_second_list):
        return self.flist[-num_elements_second_list:]


class AEDataset(Dataset):
    '''
    img = original image
    '''
    def __init__(self, data_root, image_size=256, loader=tif_loader, trd=None):
        self.path = data_root

        self.flist = os.listdir(self.path)
        random.shuffle(self.flist)
        split_ratio = 0.2
        num_elements_second_list = int(len(self.flist) * split_ratio)
        self.flist_set = self.set_base(num_elements_second_list)        

        self.tfs = A.Compose([
                A.RandomCrop(height=image_size, width=image_size, p=1.),
                ToTensorV2(p=1.0),
            ])#, additional_targets={'cond_image':'image'})
        self.loader = loader
        self.image_size = image_size
        self.trd = trd

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist_set[index])

        img = self.loader('{}/{}'.format(self.path, file_name), self.trd)
        
        img = (img - 0.5) / 0.5
        transformed = self.tfs(image=img)
        t_img = transformed['image']

        ret["image"] = t_img
        ret["file_path_"] = os.path.join(self.path, file_name)
        return ret

    def __len__(self):
        return len(self.flist_set)


class AETrain(AEDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_base(self, num_elements_second_list):
        return self.flist[:-num_elements_second_list]
    

class AEValidation(AEDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_base(self, num_elements_second_list):
        return self.flist[-num_elements_second_list:]