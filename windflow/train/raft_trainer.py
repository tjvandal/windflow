import os
import numpy as np

import torch
import torch.nn as nn
from torch import optim
#from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision 

from .base import BaseTrainer
from ..networks import raft, warper
from ..losses import CharbonnierLoss, L1Loss, L2Loss

from torch.utils.tensorboard import SummaryWriter

MAX_FLOW = 200

def sequence_loss(flow_preds, flow_gt, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds) 
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    #valid = (valid >= 0.5) & (mag < max_flow)
    valid = (mag < max_flow)
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()
        #flow_loss += (i_weight * i_loss).mean()

    return flow_loss


def unsupervised_sequence_loss(I0, I1, flow_pred, gamma=0.8):
    """Loss function unsupervised warping I1 to I0 with predicted flows"""
    warp = warper.BackwardWarp()
    loss = CharbonnierLoss(alpha=0.50)
    
    n_predictions = len(flow_pred)
    flow_loss = 0.0
    
    for i in range(n_predictions):
        I0_i = warp(I1, flow_pred[i])
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = loss(I0, I0_i)
        flow_loss +=  (i_weight * i_loss).mean()

    return flow_loss
    

class RAFTTrainer(BaseTrainer):
    def __init__(self, model, model_path, iters=24, 
                 device=None, lr=1e-5, distribute=False, 
                 clip=1., rank=0):
        BaseTrainer.__init__(self, model, 'raft', model_path, lr=lr, 
                             device=device, distribute=distribute, rank=rank)        
        self.iters = iters
        self.clip = clip
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, 
                                     weight_decay=1e-4, eps=1e-8)

        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, 
                                                       max_lr=lr, 
                                                       total_steps=500000,
                                                       pct_start=0.05, 
                                                       cycle_momentum=False, 
                                                       anneal_strategy='linear')
    
        
    #def step(self, I0, I1, flow_gt, flow_init=None, log=False, train=True):
    def step(self, inputs, labels, flow_init=None, log=False, train=True):
        if train:
            tfwriter = self.tfwriter_train
        else:
            tfwriter = self.tfwriter_valid
 
        with torch.cuda.amp.autocast():
            I0 = inputs[:,0]
            I1 = inputs[:,1]
            flow_predictions = self.model(I0, I1, iters=self.iters, flow_init=flow_init)
            loss = sequence_loss(flow_predictions, labels[:,0])

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

                self.optimizer.step()
                #self.scaler.step(self.optimizer)
                #self.scalar.update()
                self.scheduler.step()
 

            if log and (self.rank == 0):
                self.log_scalar(loss, 'total_loss', train)
                self.log_flow_grid(labels[:,0], 'label', train)
                self.log_image_grid(I0, 'data/I0', train)
                self.log_image_grid(I1, 'data/I1', train)
 
                for i in range(0, self.iters, 2):
                    self.log_flow_grid(flow_predictions[i], f'flows/iter_{i}', train)    
            if train:
                self.global_step = self.global_step + 1 
        return loss

    
class RAFTGuided(BaseTrainer):
    def __init__(self, model, model_path, iters=10, lambda_obs=0.1,
                 device=None, lr=1e-4, distribute=False, clip=1., rank=0):
        BaseTrainer.__init__(self, model, 'raft', model_path, lr=lr, 
                             device=device, distribute=distribute, rank=rank)        
        self.iters = iters
        self.clip = clip
        self.lambda_obs = lambda_obs
        
    def step(self, I0_obs, I1_obs, I0_phys, I1_phys, flow_phys, train=True, log=None):
        if train:
            tfwriter = self.tfwriter_train
        else:
            tfwriter = self.tfwriter_valid
            
        with torch.cuda.amp.autocast():
            flow_predictions_phys = self.model(I0_phys, I1_phys, iters=self.iters)
            loss_guide = sequence_loss(flow_predictions_phys, flow_phys)
            
            #flow_predictions_obs = self.model(I0_obs, I1_obs, iters=self.iters)
            #loss_obs = unsupervised_sequence_loss(I0_obs, I1_obs, flow_predictions_obs)
            print('loss', loss_guide)
            loss = loss_guide # + loss_obs * self.lambda_obs
            
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

                self.optimizer.step()
                #self.scaler.step(self.optimizer)
                #self.scalar.update()
                

            if log and (self.rank == 0):
                self.log_scalar(loss_guide, 'loss_guide', train)
                self.log_scalar(loss_obs, 'loss_obs', train)
                self.log_scalar(loss, 'total_loss', train)
                self.log_flow_grid(flow_gt, 'label', train)
                self.log_image_grid(I0, 'data/I0', train)
                self.log_image_grid(I1, 'data/I1', train)
        
                for i in range(0, self.iters, 2):
                    self.log_flow_grid(flow_predictions[i], f'flows/iter_{i}', train)   
                    
            if train:
                self.global_step = self.global_step + 1
                
        return loss
