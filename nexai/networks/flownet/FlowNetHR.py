'''
Portions of this code copyright 2017, Clement Pinard
'''

import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

from .flownet_submodules import *

#def deconv(in_planes, out_planes):
#    return nn.Sequential(
#        nn.Conv2d(in_planes, out_planes*4, kernel_size=1, bias=True),
#        nn.PixelShuffle(2),
#        nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, bias=True),
#        nn.LeakyReLU(0.1,inplace=True)
#    )

class FlowNetHR(nn.Module):
    def __init__(self, args, input_channels = 1, batchNorm=True, partial=False):
        super(FlowNetHR, self).__init__()

        self.batchNorm = batchNorm
        self.conv0 = conv(self.batchNorm, input_channels*2, 32, kernel_size=7, partial=partial)
        self.conv0_1 = conv(self.batchNorm, 32,  32, partial=partial)
        self.conv1   = conv(self.batchNorm, 32,  64, kernel_size=5, stride=2, partial=partial)
        self.conv1_1 = conv(self.batchNorm, 64,  64, partial=partial)
        self.conv2   = conv(self.batchNorm, 64,  128, kernel_size=5, stride=2, partial=partial)   
        self.conv2_1 = conv(self.batchNorm, 128,  128, partial=partial)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2, partial=partial)
        self.conv3_1 = conv(self.batchNorm, 256,  256, partial=partial)

        self.deconv2 = deconv(256, 64)
        self.deconv1 = deconv(194, 32)
        self.deconv0 = deconv(98, 16)

        self.predict_flow3 = predict_flow(256, partial=partial)
        self.predict_flow2 = predict_flow(194, partial=partial)
        self.predict_flow1 = predict_flow(98, partial=partial)
        self.predict_flow0 = predict_flow(50, partial=partial)

        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        #self.upsampled_flow3_to_2 = nn.UpsamplingBilinear2d(scale_factor=2)
        #self.upsampled_flow2_to_1 = nn.UpsamplingBilinear2d(scale_factor=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)
        self.upsample_bilinear = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x1, x2):
        x = torch.cat([x1,x2],1)
        
        out_conv0 = self.conv0_1(self.conv0(x))
        out_conv1 = self.conv1_1(self.conv1(out_conv0)) # 64
        out_conv2 = self.conv2_1(self.conv2(out_conv1)) # 128
        out_conv3 = self.conv3_1(self.conv3(out_conv2)) # 256
        
        flow3       = self.predict_flow3(out_conv3) # 2
        flow3_up    = self.upsample_bilinear(flow3) # 2 
        out_deconv2 = self.deconv2(out_conv3) # 64
        
        concat = torch.cat((out_conv2, flow3_up, out_deconv2), 1) # 128 + 64 + 2 = 194
        flow2 = self.predict_flow2(concat)
        flow2_up = self.upsample_bilinear(flow2)
        out_deconv1 = self.deconv1(concat)
        
        concat = torch.cat((out_conv1, flow2_up, out_deconv1), 1) # 64 + 32 + 2 = 98
        flow1 = self.predict_flow1(concat)
        flow1_up = self.upsample_bilinear(flow1)
        out_deconv0 = self.deconv0(concat)
        
        concat = torch.cat((out_conv0, flow1_up, out_deconv0), 1) # 32 + 16 + 2 = 50
        flow0 = self.predict_flow0(concat)
        
        if self.training:
            return flow0,flow1,flow2,flow3
        else:
            return flow0,

