import os, sys
import glob
import argparse
import time

from mpi4py import MPI

MPI_COMM = MPI.COMM_WORLD
MPI_RANK = MPI_COMM.Get_rank()
MPI_SIZE = MPI_COMM.Get_size()

N_GPUS = 2

rank_gpu = MPI_RANK % N_GPUS
os.environ['CUDA_VISIBLE_DEVICES'] = f'{rank_gpu}'

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from windflow.datasets.g5nr import g5nr
from windflow.inference.inference_flows import FlowRunner
from windflow.datasets.utils import cartesian_to_speed

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="raft", type=str)
parser.add_argument("--checkpoint", default="models/raft-size_512/checkpoint.pth.tar", type=str)
parser.add_argument("--data_directory", default="data/G5NR/", type=str)
parser.add_argument("--output", default="data/G5NR_Flows/raft/", type=str)
parser.add_argument("--n_files", default=None, type=int)
parser.add_argument("--batch_size", default=2, type=int)

args = parser.parse_args()

# Set model information
model_name = args.model_name

# Set data variables
data_directory = args.data_directory
to_directory = args.output

if (MPI_RANK == 0) and (not os.path.exists(to_directory)):
    os.makedirs(to_directory)

# get file list
files = g5nr.get_files(data_directory, 'test')
print("Number of OSSE test files", len(files))

if args.n_files is None:
    N_files = len(files)
else:
    N_files = args.n_files

N_per_rank = N_files // MPI_SIZE
files = files.iloc[N_per_rank * MPI_RANK: N_per_rank * (MPI_RANK+1) + 1]

# load model
tile_size = 512
overlap = 128

if model_name in ['pwc-net-rmsvd', 'pwc-net-l1']:
    runner = FlowRunner('pwc-net', 
                        tile_size=tile_size,
                        overlap=overlap,
                        batch_size=args.batch_size)
else:
    runner = FlowRunner(model_name.replace('-guided','').replace('-unflow', ''),
                        tile_size=tile_size,
                        overlap=overlap,
                        batch_size=args.batch_size)
 
runner.load_checkpoint(args.checkpoint)

stats = []

# iterate and perform inference, compute test statistics 
ds1 = xr.open_mfdataset(files.iloc[0].values, engine='netcdf4')
for i in range(1, files.shape[0]): 
    # open dataset
    
    ds2 = xr.open_mfdataset(files.iloc[i].values, engine='netcdf4')
 
    f = os.path.basename(files.iloc[i-1]['U']) # get ds1 file
    to_flow_file = os.path.join(to_directory, f.replace('_U_', '_WindFlow_').replace('.nc', '.zarr'))
    if os.path.exists(to_flow_file):
        ds1 = ds2.copy()
        continue 

    t = ds1['time'].values

    output_ds = xr.zeros_like(ds1)
    del output_ds['QV'], output_ds['tpw']

    U_flows = np.zeros(ds1.U.shape)
    V_flows = np.zeros(ds1.V.shape)

    t0 = time.time()
    
    for i, lev in enumerate(ds1.lev):
        qv1_lev = ds1.sel(lev=lev)['QV'].values[0]
        qv2_lev = ds2.sel(lev=lev)['QV'].values[0]
        _, flows_lev = runner.forward(qv1_lev, qv2_lev)
        U_flows[0,i] = flows_lev[0]
        V_flows[0,i] = flows_lev[1]

    output_ds['U'] = output_ds['U'] + U_flows
    output_ds['V'] = output_ds['V'] + V_flows

    output_ds = cartesian_to_speed(output_ds)
 
    output_ds.attrs['Source'] = 'NEX'
    output_ds.attrs['Title'] = 'Optical Flow Feature Tracking'
    output_ds.attrs['Contact'] = 'vandal@baeri.org'
    output_ds.attrs['History'] = 'G5NR outputs from GEOS-5 by gmao processed by NEX optical flow.'
    output_ds.attrs['Model'] = model_name
    output_ds.attrs['Pytorch_Checkpoint'] = args.checkpoint

    #output_ds.to_netcdf(to_flow_file)
    output_ds.to_zarr(to_flow_file)
    #print(f'Wrote to file {to_flow_file}')
    print(f"Wrote to file: {to_flow_file} -- Processing time {time.time()-t0} (seconds)")

    ds1 = ds2.copy()
