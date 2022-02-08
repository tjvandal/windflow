'''
Author: TJ Vandal

Process NOAA L1b Files to Flow files between a range of dates

Inputs: 10-minute NOAA L1b files, Optical Flow Model
Output: 10-minute wind vectors
'''

import os, sys
import time

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

import torch

sys.stdout.write(f"Rank {rank} Size {size} Name {name} CUDA: {os.environ['CUDA_VISIBLE_DEVICES']} Devices count: {torch.cuda.device_count()}\n")


sys.stdout.write(f'number of devices: {torch.cuda.device_count()}\n')
sys.stdout.flush()

device = 0 #'cuda:0'#torch.cuda.current_device() #'cuda:0'# + str(rank % N_GPUS)
#torch.cuda.set_device(device)

import glob
import argparse
import datetime as dt

import xarray as xr
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from windflow.datasets.geostationary import goesr, stats
from windflow.inference import inference_flows

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="raft", type=str)
parser.add_argument("--checkpoint",
                    default="/nobackupp10/tvandal/windflow/models/raft-size_512/checkpoint.pth.tar", type=str)
parser.add_argument('--start_date', type=lambda s: dt.datetime.strptime(s, '%Y-%m-%d'),
                    default='2021-01-01')
parser.add_argument('--end_date', type=lambda s: dt.datetime.strptime(s, '%Y-%m-%d'), default='2021-01-02')
parser.add_argument('--timestep', type=int, default=10)
parser.add_argument('--band', type=int, default=10)
parser.add_argument('--data_path', type=str, default='/nex/datapool/geonex/public/GOES16/NOAA-L1B/')
parser.add_argument('--spatial', type=str, default=None)
parser.add_argument('--upsample', type=float, default=None)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--product', type=str, default='ABI-L1b-RadF')
parser.add_argument('--output_path', type=str, default='data/L1B-WindFlow/')
args = parser.parse_args()


curr_date = args.start_date
dates = []
while curr_date <= args.end_date:
    dates.append(curr_date)
    #curr_date = curr_date + dt.timedelta(minutes=args.timestep)
    curr_date = curr_date + dt.timedelta(hours=1)

rank_idxs = np.arange(rank, len(dates)-1, size)

inference = inference_flows.GeoFlows(args.model_name.replace('-guided', ''), 
                                     args.data_path, 
                                     overlap=128, 
                                     tile_size=512,
                                     channels=[args.band], 
                                     product=args.product,
                                     device=device,
                                     batch_size=args.batch_size, 
                                     timestep=args.timestep,
                                     upsample_data=args.upsample, 
                                     spatial=args.spatial)
inference.load_checkpoint(args.checkpoint)

for idx in rank_idxs:
    t = dates[idx]

    filename = f'GeoNEX_AMV_Flows-{args.product}-B{args.band:02g}-{t.year}{t.month:02g}{t.day:02g}_{t.hour:02g}{t.minute:02g}.{args.model_name}.zarr'
    output_file_path = os.path.join(args.output_path, args.product, f'{t.year}', 
                               f'{t.timetuple().tm_yday:03g}')
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    output_file = os.path.join(output_file_path, filename)
    if os.path.exists(output_file):
        continue

    sys.stdout.write(f"Processing time {t}\n")
    sys.stdout.flush()

    #data, flows = inference.flows_by_time(t)
    try:
        t0 = time.time()
        data = inference.flows_by_time(t, reproject=True)
                
        '''
        img1 = data['Rad'].values
        img1[img1 == 0] = np.nan
        flows[0][img1 != img1] = np.nan
        flows[1][img1 != img1] = np.nan
        '''

        ds_out = xr.Dataset(dict(U=data['U'], V=data['V']))
        ds_out['band'] = args.band
        ds_out['lat'] = data['lat']
        ds_out['lon'] = data['lon']
        ds_out['t'] = t
        #ds_out.attrs['input_dataset_name'] = data.attrs['dataset_name']

        #ds_out.to_netcdf(output_file)
        ds_out.chunk({'lat': 1000, 'lon': 1000}).to_zarr(output_file)
        print(f"Wrote to file: {output_file} -- Processing time {time.time()-t0} (seconds)")
        sys.stdout.write(output_file + '\n')
        sys.stdout.flush()

        del ds_out, data
    except Exception as err:
        print('Exception Raised', err, t)
