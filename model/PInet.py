import os
import numpy as np

import torch
import torch.nn as nn

from . import layers

class Generator(nn.Module):
    def __init__(self,pano_in_channels, cube_in_channels,pano_out_channels,cube_out_channels,decoder_in_channels, decoder_out_channels, learning_type="plain", nker=64, norm = "bonrm"):
        super(Generator,self).__init__()

        self.pano_in_channels = pano_in_channels
        self.pano_out_channels = pano_out_channels
        self.cube_in_channels = cube_in_channels
        self.cube_out_channels = cube_out_channels
        self.decoder_in_channels=decoder_in_channels
        self.decoder_out_channels=decoder_out_channels
        self.learning_type=learning_type
        self.nker=nker
        self.norm = norm

        self.pano_encoder=PanoNet(self.pano_in_channels,self.pano_out_channels,self.nker,self.norm)
        self.cube_encoder=CubeNet(self.cube_in_channels,self.cube_out_channels,self.nker,self.norm)
        self.decoder = Decoder(self.decoder_in_channels,self.decoder_out_channels,self.nker,self.norm)

    def forward(self,pano_x,cube_x):
        pano_out = self.pano_encoder(pano_x)
        cube_out = self.cube_encoder(cube_x)

        pano_cube = torch.cat((pano_out, cube_out), dim=1)

        pano_cube = pano_cube.unsqueeze(2)
        pano_cube = pano_cube.unsqueeze(3)

        out = self.decoder(pano_cube)

        return pano_out, cube_out, pano_cube, out
    

class PanoNet(nn.Module):
    def __init__(self, pano_in_channels, pano_out_channels, nker=64, norm="bnorm"):
        super(PanoNet,self).__init__()
        
        # panorama encoder
        self.enc1_1 = layers.CBR2d(in_channels=pano_in_channels, out_channels=1 * nker, norm=norm)
        self.enc1_2 = layers.CBR2d(in_channels=1 * nker, out_channels=1 * nker, norm=norm)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = layers.CBR2d(in_channels=1 * nker, out_channels=2 * nker, norm=norm)
        self.enc2_2 = layers.CBR2d(in_channels=2 * nker, out_channels=2 * nker, norm=norm)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = layers.CBR2d(in_channels=2 * nker, out_channels=4 * nker, norm=norm)
        self.enc3_2 = layers.CBR2d(in_channels=4 * nker, out_channels=4 * nker, norm=norm)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = layers.CBR2d(in_channels=4 * nker, out_channels=8 * nker, norm=norm)
        self.enc4_2 = layers.CBR2d(in_channels=8 * nker, out_channels=8 * nker, norm=norm)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = layers.CBR2d(in_channels=8 * nker, out_channels=pano_out_channels, norm=norm)

        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6',nn.Linear(pano_out_channels*20*11, 4096))
        self.fc6.add_module('relu6',nn.ReLU(inplace=True))
        self.fc6.add_module('drop6',nn.Dropout(p=0.5))

        self.fc7 = nn.Linear(4096,pano_out_channels)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.enc1_1(x)
        x = self.enc1_2(x)
        x = self.pool1(x)

        x = self.enc2_1(x)
        x = self.enc2_2(x)
        x = self.pool2(x)

        x = self.enc3_1(x)
        x = self.enc3_2(x)
        x = self.pool3(x)

        x = self.enc4_1(x)
        x = self.enc4_2(x)
        x = self.pool4(x)

        x = self.enc5_1(x)

        x = self.fc6(x.view(B,-1))

        x = self.fc7(x)

        return x


class CubeNet(nn.Module):
    def __init__(self, cube_in_channels, cube_out_channels, nker=64, norm="bnorm"):
        super(CubeNet,self).__init__()

        self.cube_encoder = nn.Sequential()
        self.cube_encoder.add_module('enc1_1',layers.CBR2d(in_channels=cube_in_channels, out_channels=1 * nker, norm=norm))
        self.cube_encoder.add_module('enc1_2',layers.CBR2d(in_channels=1 * nker, out_channels=1 * nker, norm=norm))

        self.cube_encoder.add_module('pool1',nn.MaxPool2d(kernel_size=2))

        self.cube_encoder.add_module('enc2_1',layers.CBR2d(in_channels=1 * nker, out_channels=2 * nker, norm=norm))
        self.cube_encoder.add_module('enc2_2',layers.CBR2d(in_channels=2 * nker, out_channels=2 * nker, norm=norm))

        self.cube_encoder.add_module('pool2',nn.MaxPool2d(kernel_size=2))

        self.cube_encoder.add_module('enc3_1',layers.CBR2d(in_channels=2 * nker, out_channels=4 * nker, norm=norm))
        self.cube_encoder.add_module('enc3_2',layers.CBR2d(in_channels=4 * nker, out_channels=4 * nker, norm=norm))

        self.cube_encoder.add_module('pool3',nn.MaxPool2d(kernel_size=2))

        self.cube_encoder.add_module('enc4_1',layers.CBR2d(in_channels=4 * nker, out_channels=8 * nker, norm=norm))
        self.cube_encoder.add_module('enc4_2',layers.CBR2d(in_channels=8 * nker, out_channels=8 * nker, norm=norm))

        self.cube_encoder.add_module('pool4',nn.MaxPool2d(kernel_size=2))

        self.cube_encoder.add_module('enc5_1',layers.CBR2d(in_channels=8 * nker, out_channels=cube_out_channels, norm=norm))

        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6',nn.Linear(cube_out_channels*8*8, 1024))
        self.fc6.add_module('relu6',nn.ReLU(inplace=True))
        self.fc6.add_module('drop6',nn.Dropout(p=0.5))

        self.fc7 = nn.Sequential()
        self.fc7.add_module('fc7',nn.Linear(6*1024,4096))
        self.fc7.add_module('relu7',nn.ReLU(inplace=True))
        self.fc7.add_module('drop7',nn.Dropout(p=0.5))

        self.fc8 = nn.Sequential()
        self.fc8.add_module('fc8',nn.Linear(4096, cube_out_channels))

    def forward(self, x):
        B,FC,H,W = x.size() # batch, face, channel, height, width
        
        x_list = []
        for i in range(6):
            z = x[:,4*i:4*(i+1),:,:]
            z = self.cube_encoder(z)
            z = self.fc6(z.view(B,-1))
            z = z.view([B,1,-1])
            x_list.append(z)

        x = torch.cat(x_list,1)
        x = self.fc7(x.view(B,-1))
        x = self.fc8(x)

        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm="bnorm"):
        super(Decoder, self).__init__()

        self.dec1 = layers.DECBR2d(1 * in_channels, 16 * nker, kernel_size=4, stride=1,
                            padding=0, norm=norm, relu=0.0, bias=False)

        self.dec2 = layers.DECBR2d(16 * nker, 8 * nker, kernel_size=4, stride=2,
                            padding=1, norm=norm, relu=0.0, bias=False)

        self.dec3 = layers.DECBR2d(8 * nker, 4 * nker, kernel_size=4, stride=2,
                            padding=1, norm=norm, relu=0.0, bias=False)

        self.dec4 = layers.DECBR2d(4 * nker, 2 * nker, kernel_size=4, stride=2,
                            padding=1, norm=norm, relu=0.0, bias=False)

        self.dec5 = layers.DECBR2d(2 * nker, 1 * nker, kernel_size=4, stride=2,
                            padding=1, norm=norm, relu=0.0, bias=False)

        self.dec6 = layers.DECBR2d(1 * nker, out_channels, kernel_size=4, stride=2,
                            padding=1, norm=None, relu=None, bias=False)

    def forward(self, x):

        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        x = self.dec6(x)

        x = torch.tanh(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm="bnorm"):
        super(Discriminator, self).__init__()

        self.enc1 = layers.CBR2d(1 * in_channels, 1 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc2 = layers.CBR2d(1 * nker, 2 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc3 = layers.CBR2d(2 * nker, 4 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc4 = layers.CBR2d(4 * nker, 8 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc5 = layers.CBR2d(8 * nker, out_channels, kernel_size=4, stride=2,
                          padding=1, norm=None, relu=None, bias=False)

    def forward(self, x):

        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)

        x = torch.sigmoid(x)

        return x