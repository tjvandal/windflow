import os
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torchvision 

from .base import BaseTrainer
from ..networks import FlowNetS
from ..losses import CharbonnierLoss, L1Loss, L2Loss, RMSVDLoss

from torch.utils.tensorboard import SummaryWriter

class MultiFrameTrainer(BaseTrainer):
    def __init__(self,
                 model, 
                 model_name, 
                 model_path, 
                 lr=1e-4,
                 device=None,
                 distribute=False,
                 rank=0):
        BaseTrainer.__init__(self, model, model_name, model_path, lr=lr, 
                             device=device, distribute=distribute, rank=rank)
        
        # set loss functions
        self.photo_loss = L2Loss(None)
        
    def step(self, inputs, labels, log=False, train=True):
        '''
        Inputs shape: (N, T, C, H, W)
        Flows shape: (N, T, 2, H, W)
        '''
        
        N_pairs = len(inputs)-1
        # for every pair, compute 

class FlownetTrainer(BaseTrainer):
    def __init__(self,
                 model, 
                 model_name, 
                 model_path, 
                 lr=1e-4,
                 device=None,
                 distribute=False,
                 rank=0,
                 loss='L2'):
        BaseTrainer.__init__(self, model, model_name, model_path, lr=lr, 
                             device=device, distribute=distribute, rank=rank)

        # set loss functions
        if loss == 'L2':
            self.photo_loss = L2Loss(None)
        elif loss.lower() == 'rmsvd':
            print("RMSVD Loss")
            self.photo_loss = RMSVDLoss()
        elif loss == 'CHAR':
            self.photo_loss = CharbonnierLoss(alpha=0.50)
        elif loss == 'L1':
            self.photo_loss = L1Loss(None)

        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',
                                                                    #factor=0.5, patience=100)
    #def step(self, I0, I1, UV, log=False, train=True):
    def step(self, inputs, labels, log=False, train=True):
        '''
        Inputs shape: (N, T, C, H, W)
        Flows shape: (N, T, 2, H, W)
        '''
        I0 = inputs[:,0]
        I1 = inputs[:,1]
        x = torch.cat([I0, I1], 1)
        
        #with torch.cuda.amp.autocast():
        #flows = self.model(I0, I1)
        flows = self.model(x)
        #if not isinstance(flows, list):
        #    flows = [flows]

        UV_l = labels[:,0]
        losses = []
        weights = [0.32, 0.16, 0.08, 0.04, 0.02, 0.005]

        for layer, flow_l in enumerate(flows):
            H = flow_l.shape[2]
            scale_factor = flow_l.shape[2] / UV_l.shape[2]
            UV_l = torch.nn.functional.interpolate(UV_l, scale_factor=scale_factor, 
                                                   mode='bilinear', recompute_scale_factor=True)
            losses.append(weights[layer] * self.photo_loss(flow_l, UV_l))
            if log:
                self.log_flow_grid(flow_l, f'flow_level_{layer}', train=train)

        total_loss = sum(losses)

        if train:
            self.optimizer.zero_grad()
            total_loss.backward()
            #self.scaler.scale(total_loss).backward()
            self.optimizer.step()
            #self.scaler.step(self.optimizer)
            #self.scaler.update()
        #else:
        #    self.scheduler.step(total_loss)

        if log and (self.rank == 0):
            self.log_scalar(total_loss, "total_loss", train=train)
            self.log_image_grid(I0, "data/I0", train=train)
            self.log_image_grid(I1, "data/I1", train=train)
            self.log_flow_grid(labels[:,0], "label", train=train)
            for i in range(len(losses)):
                self.log_scalar(losses[i], f"losses/level_{i}", train=train)
                self.log_flow_grid(flows[i], f"level_{i}", train=train)

        if train:
            self.global_step += 1

        return total_loss
