import sys
import os
import time
import argparse

from torch.utils.tensorboard import SummaryWriter # there is a bug with summarywriter importing
from collections import OrderedDict

import numpy as np
import torch
torch.cuda.empty_cache()

import torch.nn as nn
from torch import optim
from torch.utils import data
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from windflow.datasets import get_dataset
from windflow.networks.models import get_flow_model
from windflow.train.trainers import get_flow_trainer

os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning' 


def setup(rank, world_size, port):
    '''
    Setup multi-gpu processing group
    Args:
        rank: current rank
        world_size: number of processes
        port: which port to connect to
    Returns:
        None
    '''
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'{port}'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_net(params, rank=0):
    # set device
    #if not device:
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = rank #% N_DEVICES
    print(f'in train_net rank {rank} on device {device}')

    if params['ngpus'] > 1:
        distribute = True
    else:
        distribute = False

    dataset_train, dataset_valid = get_dataset(params['dataset'], params['data_path'],
                                              scale_factor=params['scale_input'], 
                                               frames=params['input_frames'])
    
    data_params = {'batch_size': params['batch_size'] // params['ngpus'], 'shuffle': True,
                   'num_workers': 8, 'pin_memory': True}
    training_generator = data.DataLoader(dataset_train, **data_params)
    val_generator = data.DataLoader(dataset_valid, **data_params)

    model = get_flow_model(params['model_name'], small=False)
    if distribute:
        model = DDP(model.to(device), device_ids=[device], find_unused_parameters=True)
    
    trainer = get_flow_trainer(model, params['model_name'], 
                               params['model_path'], 
                               distribute=distribute,
                               rank=rank,
                               lr=params['lr'],
                               loss=params['loss'])

    trainer.model.to(device)
    trainer.load_checkpoint()

    print(f'Start Training {params["model_name"]} on {params["dataset"]}')

    while trainer.global_step < params['max_iterations']:
        running_loss = 0.0
        t0 = time.time()
        for batch_idx, (images, flows) in enumerate(training_generator):
            images = images.to(device)
            flows = flows.to(device)

            log = False
            if (trainer.global_step % params['log_step'] == 0):
                log=True

            train_loss = trainer.step(images, flows, 
                                      log=log, train=True)
            if np.isinf(train_loss.cpu().detach().numpy()):
                print(train_loss, I0.cpu().detach().numpy(), I1.cpu().detach().numpy().flatten())
                return

            if log:
                images, flows = next(iter(val_generator))
                valid_loss = trainer.step(images.to(device), flows.to(device), 
                                          log=log, train=False)
                print(f'Rank {trainer.rank} @ Step: {trainer.global_step-1}, Training Loss: {train_loss:.3f}, Valid Loss: {valid_loss:.3f}')

            if (trainer.global_step % params['checkpoint_step'] == 0) and (rank == 0):
                trainer.save_checkpoint()

    return best_validation_loss

def manual_experiment(args):
    train_net(vars(args))

def train_net_mp(rank, world_size, port, params):
    '''
    Setup and train on node
    '''
    setup(rank, world_size, port)
    train_net(params, rank=rank)

def run_training(args, world_size, port):
    #params['batch_size'] = params['batch_size'] // world_size
    if world_size > 1:
        mp.spawn(train_net_mp,
                 args=(world_size, port, vars(args)),
                 nprocs=world_size,
                 join=True)
        cleanup()
    else:
        train_net(vars(args))

if __name__ == "__main__":
    # Feb 1 2020, Band 1 hyper-parameter search
    # {'w': 0.6490224421024322, 's': 0.22545622639358046, 'batch_size': 128}
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/raft-size_512/", type=str)
    parser.add_argument("--dataset", default="g5nr", type=str)
    parser.add_argument("--data_path", default="data/G5NR_patches/",    type=str)
    parser.add_argument("--input_frames", default=2, type=int)
    parser.add_argument("--model_name", default="raft", type=str)
    #parser.add_argument("--gpus", default="0", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--scale_input", default=None, type=float, help='Bilinear interpolation on input image.')
    parser.add_argument('--max_iterations',type=int, default=2000000, help='Number of training iterations')
    parser.add_argument("--log_step", default=1000, type=int)
    parser.add_argument("--checkpoint_step", default=5000, type=int)
    parser.add_argument("--ngpus", default=1, type=int)
    parser.add_argument("--port", default=9009, type=int)
    parser.add_argument("--loss", default='L1', type=str)

    args = parser.parse_args()
    if torch.cuda.device_count() < args.ngpus:
        print(f"Cannot running training because {args.ngpus} are not available.")

    run_training(args, args.ngpus, args.port)
