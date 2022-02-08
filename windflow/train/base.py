import os, sys

import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter # there is a bug with summarywriter importing

import torchvision

def scale_image(x):
    xmn = torch.min(x)
    xmx = torch.max(x)
    return (x - xmn) / (xmx - xmn)

class BaseTrainer(nn.Module):
    def __init__(self, model, 
                 model_name, 
                 model_path,
                 lr=1e-4,
                 device=None,
                 distribute=False,
                 rank=0):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.model_name = model_name
        self.model_path = model_path
        self.lr = lr
        self.device = device
        self.distribute = distribute
        self.rank = rank
        
        # NEW
        self.scaler = torch.cuda.amp.GradScaler()
            
        self.checkpoint_filepath = os.path.join(model_path, 'checkpoint.pth.tar')
        if (rank == 0) and (not os.path.exists(model_path)):
            os.makedirs(model_path)
        
        self.global_step = 0
        self._set_optimizer()
        self._set_summary_writer()

        
    def _set_optimizer(self):
        # set optimizer
        #if self.rank == 0:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)


    def _set_summary_writer(self):
        self.tfwriter_train = SummaryWriter(os.path.join(self.model_path, 'train', 'tfsummary'))
        self.tfwriter_valid = SummaryWriter(os.path.join(self.model_path, 'valid', 'tfsummary'))

    def load_checkpoint(self):
        filename = self.checkpoint_filepath
        if os.path.isfile(filename):
            print("loading checkpoint %s" % filename)
            checkpoint = torch.load(filename)
            self.global_step = checkpoint['global_step']
            try:
                self.model.module.load_state_dict(checkpoint['model'])
            except:
                self.model.load_state_dict(checkpoint['model'])
                
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (Step {})"
                    .format(filename, self.global_step))
        else:
            print("=> no checkpoint found at '{}'".format(filename))
        
    def save_checkpoint(self):
        if self.distribute:
            state = {'global_step': self.global_step, 
                     'model': self.model.module.state_dict(),
                     'optimizer': self.optimizer.state_dict()}
        else:
            state = {'global_step': self.global_step, 
                     'model': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict()}
        torch.save(state, self.checkpoint_filepath)        

    def log_tensorboard(self):
        pass

    def get_tfwriter(self, train):
        if train:
            return self.tfwriter_train
        else:
            return self.tfwriter_valid
        
    def log_scalar(self, x, name, train=True):
        tfwriter = self.get_tfwriter(train)
        tfwriter.add_scalar(name, x, self.global_step)
    
    def log_image_grid(self, img, name, train=True, N=4):
        '''
        img of shape (N, C, H, W)
        '''
        tfwriter = self.get_tfwriter(train)
        img_grid = torchvision.utils.make_grid(img[:N])
        tfwriter.add_image(name, scale_image(img_grid), self.global_step)
        
    def log_flow_grid(self, flows, name, train=True, N=4):
        tfwriter = self.get_tfwriter(train)
        U_grid = torchvision.utils.make_grid(flows[:N,:1])
        V_grid = torchvision.utils.make_grid(flows[:N,1:])
        intensity = (U_grid ** 2 + V_grid ** 2)**0.5
        tfwriter.add_image(f'{name}/U', scale_image(U_grid), self.global_step)
        tfwriter.add_image(f'{name}/V', scale_image(V_grid), self.global_step)
        tfwriter.add_image(f'{name}/intensity', scale_image(intensity), self.global_step)
        