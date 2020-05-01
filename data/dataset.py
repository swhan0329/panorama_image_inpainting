from __future__ import print_function, division

import os
import glob
import json
import io
import base64

from PIL import Image
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


def b64utf82ndarr(b_string):
    b64_barr = b_string.encode('utf-8')
    content = base64.b64decode(b64_barr)
    img = Image.open(io.BytesIO(content))
    inp_np = np.asarray(img)
    return inp_np


class PanoramaDataset(data.Dataset):
    def __init__(self, in_dir, transform=None):
        self.inp_paths = glob.glob(os.path.join(in_dir, "*.json"))
        self.face_order = ['f', 'r', 'b', 'l', 't', 'd']
        self.face_mask_order = ['f_mask', 'r_mask', 'b_mask', 'l_mask', 't_mask', 'd_mask']
        self.transform = transform

    def __len__(self):
        return len(self.inp_paths)
    
    def __getitem__(self, idx, color_format='RGB'):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item_path = self.inp_paths[idx]
        with open(item_path, 'r') as f:
            in_json = json.load(f)

            sample_cube_img = list()
            cube_mask_img = list()
            sample_pano_img = list()
            pano_mask_img = list()
            if in_json['mask_flag']:
                # (1, H, W, C)
                sample_pano_img.append(b64utf82ndarr(in_json['pano']['pano'][0]))
                pano_mask_img.append(b64utf82ndarr(in_json['pano']['pano_mask'][0]))
            else:
                # (1, H, W, C)
                sample_pano_img.append(b64utf82ndarr(in_json['pano']['pano'][0]))
            
            for k in self.face_order:
                # (6, H, W, C)
                sample_cube_img.append(b64utf82ndarr(in_json['cube'][k][0]))
            
            for l in self.face_mask_order:
                if in_json['mask_flag']:
                    # (6, H, W, C)
                    cube_mask_img.append(b64utf82ndarr(in_json['cube'][l][0]))
            
            sample_cube_img = np.asarray(sample_cube_img)
            sample_pano_img = np.asarray(sample_pano_img)
            
            if in_json['mask_flag']:
                np_cube_mask_img = np.asarray(cube_mask_img)
                pano_mask_img = np.asarray(pano_mask_img)
                for ff in range(6):
                    cube_mask_img_temp = (np_cube_mask_img[ff,:,:,0]+np_cube_mask_img[ff,:,:,1]+np_cube_mask_img[ff,:,:,2])
                    cube_mask_img_temp = cube_mask_img_temp[...,np.newaxis]
                    cube_mask_img_temp = cube_mask_img_temp[np.newaxis,...]
                    if ff == 0:
                        cube_mask_img = cube_mask_img_temp
                    else:
                        cube_mask_img = np.concatenate((cube_mask_img,cube_mask_img_temp),axis=0)

                pano_mask_img = (pano_mask_img[:,:,:,0]+pano_mask_img[:,:,:,1]+pano_mask_img[:,:,:,2])
                pano_mask_img = pano_mask_img[...,np.newaxis]

        if in_json['mask_flag']:
            sample = {'cube': sample_cube_img, 'cube_mask': cube_mask_img, 'pano': sample_pano_img, 'pano_mask': pano_mask_img}
        else:
            cube_mask_img = np.zeros(sample_cube_img.shape) # make mask's value all zero
            pano_mask_img = np.zeros(sample_pano_img.shape) # make mask's value all zero
            sample = {'cube': sample_cube_img, 'cube_mask': cube_mask_img, 'pano': sample_pano_img, 'pano_mask': pano_mask_img}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

class Normalize(object):
    def __init__(self, mean=0.5,std=0.5):
        self.mean=mean
        self.std=std

    def __call__(self, sample):
        sample_cube_img, cube_mask_img, sample_pano_img, pano_mask_img = sample['cube'], sample['cube_mask'], sample['pano'], sample['pano_mask']

        sample_cube_img = sample_cube_img / 255.
        sample_pano_img = sample_pano_img / 255.
        cube_mask_img = cube_mask_img / 255.
        pano_mask_img = pano_mask_img / 255.

        cube_mask_img[cube_mask_img<0.5]=0.0
        cube_mask_img[cube_mask_img>=0.5]=1.0
        pano_mask_img[pano_mask_img<0.5]=0.0
        pano_mask_img[pano_mask_img>=0.5]=1.0

        sample_cube_img = (sample_cube_img - self.mean) / self.std
        sample_pano_img = (sample_pano_img - self.mean) / self.std

        return {'cube': sample_cube_img, 'cube_mask': cube_mask_img, 'pano': sample_pano_img, 'pano_mask': pano_mask_img}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sample_cube_img, cube_mask_img, sample_pano_img, pano_mask_img = sample['cube'], sample['cube_mask'], sample['pano'], sample['pano_mask']

        # (F, H, W, C) -> (F, C, H, W)
        sample_cube_img = np.transpose(sample_cube_img, (0, 3, 1, 2))
        cube_mask_img = np.transpose(cube_mask_img, (0, 3, 1, 2))
        sample_pano_img = np.transpose(sample_pano_img, (0, 3, 1, 2))
        pano_mask_img = np.transpose(pano_mask_img, (0, 3, 1, 2))

        return {'cube': torch.from_numpy(sample_cube_img), 'cube_mask': torch.from_numpy(cube_mask_img),\
             'pano': torch.from_numpy(sample_pano_img), 'pano_mask': torch.from_numpy(pano_mask_img)}


class Resize(object):
    def __init__(self, shape):
        self.shape =shape

    def __call__(self, sample):
        sample_cube_img, cube_mask_img, sample_pano_img, pano_mask_img = sample['cube'], sample['cube_mask'], sample['pano'], sample['pano_mask']

        sample_pano_img = resize(sample_pano_img, output_shape = (sample_pano_img.shape[0],self.shape[0],self.shape[1],3),preserve_range=True)
        pano_mask_img = resize(pano_mask_img, output_shape = (pano_mask_img.shape[0],self.shape[0],self.shape[1],1),preserve_range=True)

        return {'cube': sample_cube_img, 'cube_mask': cube_mask_img, 'pano': sample_pano_img, 'pano_mask': pano_mask_img}
        

def show_imgs(image, fig):
    for idx in range(image.shape[0]):
        ax = fig.add_subplot(2, 3, idx + 1)
        ax.axis('off')
        ax.imshow(image[idx])
   

if __name__ == "__main__":
    #transformed_dataset = PanoramaDataset(in_dir='/home/sw/360VR+Inpainting/code/erp_inpainting/images/out_temp_image', transform=transforms.Compose([Normalize(), Resize((1080,720,3)),ToTensor()]))
    transformed_dataset = PanoramaDataset(in_dir='/home/sw/360VR+Inpainting/code/erp_inpainting/images/out_temp_image')
    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]

        print(i, sample['cube'].shape)

        #print(sample['image'].shape)
        
        if i == len(transformed_dataset) - 1:
            fig = plt.figure()
            show_imgs(sample['cube'], fig)
            plt.show()

        if i == len(transformed_dataset) - 1:
            plt.imshow(sample['pano'][0])
            plt.axis('off')
            plt.show()

        sample['cube_mask'] = np.concatenate((sample['cube_mask'],sample['cube_mask'],sample['cube_mask']),axis=3)
        if i == len(transformed_dataset) - 1:
            fig = plt.figure()
            show_imgs(sample['cube_mask'], fig)
            plt.show()