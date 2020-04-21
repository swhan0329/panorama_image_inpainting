from __future__ import print_function, division

import os
import glob
import json
import io
import base64

from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt


def b64utf82ndarr(b_string):
    b64_barr = b_string.encode('utf-8')
    content = base64.b64decode(b64_barr)
    img = Image.open(io.BytesIO(content))
    inp_np = np.asarray(img)
    return inp_np


class Normalize(object):
    def __inint__(self, mean=0.5,std=0.5):
        self.mean=mean
        self.std=std

    def __call__(self, sample):
        sample_img, mask_img = sample['image'], sample['mask']
        sample_img = sample_img / 255.
        #sample_img = (sample_img-self.mean)/self.std

        # mask already has 0, 1 value
        #if mask_img is not None:
        #    mask_img = mask_img/255.

        return {'image': sample_img, 'mask': mask_img}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sample_img, mask_img = sample['image'], sample['mask']

        # (F, H, W, C) -> (F, C, H, W)
        sample_img = np.transpose(sample_img, (0, 3, 1, 2))
        mask_img = np.transpose(mask_img, (0, 3, 1, 2))

        return {'image': torch.from_numpy(sample_img), 'mask': torch.from_numpy(mask_img)}

class PanoramaDataset(data.Dataset):
    def __init__(self, in_dir, transform=None):
        self.inp_paths = glob.glob(os.path.join(in_dir, "*.json"))
        self.face_order = ['f', 'r', 'b', 'l', 't', 'd']
        self.transform = transform

    def __len__(self):
        return len(self.inp_paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item_path = self.inp_paths[idx]
        with open(item_path, 'r') as f:
            in_json = json.load(f)
            
            sample_img = list()
            mask_img = list()
            if in_json['type'] == 'erp':
                if in_json['mask_flag']:
                    # (1, H, W, C)
                    sample_img.append(b64utf82ndarr(in_json['imgs']['erp'][0]))
                    mask_img.append(b64utf82ndarr(in_json['imgs']['erp'][1]))
                else:
                    # (1, H, W, C)
                    sample_img.append(b64utf82ndarr(in_json['imgs']['erp'][0]))
            else:
                for k in self.face_order:
                    if in_json['mask_flag']:
                        # (6, H, W, C)
                        sample_img.append(b64utf82ndarr(in_json['imgs'][k][0]))
                        mask_img.append(b64utf82ndarr(in_json['imgs'][k][1]))
                    else:
                        # (6, H, W, C)
                        sample_img.append(b64utf82ndarr(in_json['imgs'][k][0]))
                        
            sample_img = np.asarray(sample_img)
            if in_json['mask_flag']:
                mask_img = np.asarray(mask_img)
        
        if in_json['mask_flag']:
            sample = {'image': sample_img, 'mask': mask_img}
        else:
            mask_img = np.zeros(sample_img.shape) # make mask's value all zero
            sample = {'image': sample_img, 'mask': mask_img}

        if self.transform:
            sample = self.transform(sample)

        return sample

def show_imgs(image, fig):
    for idx in range(image.shape[0]):
        ax = fig.add_subplot(2, 3, idx + 1)
        ax.axis('off')
        ax.imshow(image[idx])

    

if __name__ == "__main__":
    transformed_dataset = PanoramaDataset(in_dir='/home/sw/360VR+Inpainting/code/erp_inpainting/data/out_temp', transform=transforms.Compose([Normalize(), ToTensor()]))
    #transformed_dataset = PanoramaDataset(in_dir='/home/sw/360VR+Inpainting/code/erp_inpainting/data/out_temp')
    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
        
        print(i, sample['image'].shape)

        #print(sample['image'].shape)
        
        # if i == len(transformed_dataset) - 1:
        #     fig = plt.figure()
        #     show_imgs(sample['image'], fig)
        #     plt.show()