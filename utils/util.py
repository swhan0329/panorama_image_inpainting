import os
import math
import random
import numpy as np

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms


def set_random_seed(seed: int = 0):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def face_save(ckpt_dir, netFaceG, netFaceD, optimFG, optimFD, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'netFaceG': netFaceG.state_dict(), 'netFaceD': netFaceD.state_dict(),
                'optimFG': optimFG.state_dict(), 'optimFD': optimFD.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

def cube_save(ckpt_dir,netCubeG, netWholeD,
                    netSliceD,optimCG, optimCD,epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'netCubeG': netCubeG.state_dict(), 'netWholeD': netWholeD.state_dict(),
                'netSliceD': netSliceD.state_dict(), 'optimCG': optimCG.state_dict(), 'optimCD': optimCD.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

def save(ckpt_dir, netG, netD, optimG, optimD, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'netG': netG.state_dict(), 'netD': netD.state_dict(),
                'optimG': optimG.state_dict(), 'optimD': optimD.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

def parallel_save(ckpt_dir, netG, netD, optimG, optimD, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'netG': netG.module.state_dict(), 'netD': netD.module.state_dict(),
                'optimG': optimG.module.state_dict(), 'optimD': optimD.module.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

## Load network
def load(ckpt_dir, netG, netD, optimG, optimD):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return netG, netD, optimG, optimD, epoch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]), map_location='cpu')

    netG.load_state_dict(dict_model['netG'])
    netD.load_state_dict(dict_model['netD'])
    optimG.load_state_dict(dict_model['optimG'])
    optimD.load_state_dict(dict_model['optimD'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return netG, netD, optimG, optimD, epoch

def face_load(ckpt_dir, netFaceG, netFaceD, optimFG, optimFD):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return netFaceG, netFaceD, optimFG, optimFD, epoch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]), map_location='cpu')

    netFaceG.load_state_dict(dict_model['netFaceG'])
    netFaceD.load_state_dict(dict_model['netFaceD'])
    optimFG.load_state_dict(dict_model['optimFG'])
    optimFD.load_state_dict(dict_model['optimFD'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return netFaceG, netFaceD, optimFG, optimFD, epoch

def cube_load(ckpt_dir,netCubeG, netWholeD,netSliceD, optimCG, optimCD):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return netCubeG, netWholeD,netSliceD, optimCG, optimCD, epoch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]), map_location='cpu')

    netCubeG.load_state_dict(dict_model['netCubeG'])
    netWholeD.load_state_dict(dict_model['netWholeD'])
    netSliceD.load_state_dict(dict_model['netSliceD'])
    optimCG.load_state_dict(dict_model['optimCG'])
    optimCD.load_state_dict(dict_model['optimCD'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return netCubeG, netWholeD,netSliceD, optimCG, optimCD, epoch

def poisson_blend(x, output, mask):
    """
    * inputs:
        - x (torch.Tensor, required)
                Input image tensor of shape (N, 3, H, W).
        - output (torch.Tensor, required)
                Output tensor from Completion Network of shape (N, 3, H, W).
        - mask (torch.Tensor, required)
                Input mask tensor of shape (N, 1, H, W).
    * returns:
                An image tensor of shape (N, 3, H, W) inpainted
                using poisson image editing method.
    """
    x = x.clone().cpu()
    output = output.clone().cpu()
    mask = mask.clone().cpu()
    mask = torch.cat((mask,mask,mask), dim=1) # convert to 3-channel format
    num_samples = x.shape[0]
    ret = []
    for i in range(num_samples):
        dstimg = transforms.functional.to_pil_image(x[i])
        dstimg = np.array(dstimg)[:, :, [2, 1, 0]]
        srcimg = transforms.functional.to_pil_image(output[i])
        srcimg = np.array(srcimg)[:, :, [2, 1, 0]]
        msk = transforms.functional.to_pil_image(mask[i])
        msk = np.array(msk)[:, :, [2, 1, 0]]

        if msk.sum() != 0:
            # compute mask's center
            xs, ys = [], []
            for i in range(msk.shape[0]):
                for j in range(msk.shape[1]):
                    if msk[i,j,0] == 255:
                        ys.append(i)
                        xs.append(j)
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            center = ((xmax + xmin) // 2, (ymax + ymin) // 2)
            out = cv2.seamlessClone(srcimg, dstimg, msk, center, cv2.NORMAL_CLONE)
            out = out[:, :, [2, 1, 0]]
            out = transforms.functional.to_tensor(out)
            out = torch.unsqueeze(out, dim=0)
            ret.append(out)
        else:
            out = transforms.functional.to_tensor(dstimg)
            out = torch.unsqueeze(out, dim=0)
            ret.append(out)
    ret = torch.cat(ret, dim=0)
    return ret


def save_pretrained_weights(ckpt_dir: str, netFaceG, netCubeG, filename: str = 'pretrained.pth'):
    """Save only generator weights for inference."""
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    torch.save({'netFaceG': netFaceG.state_dict(),
                'netCubeG': netCubeG.state_dict()},
               os.path.join(ckpt_dir, filename))
