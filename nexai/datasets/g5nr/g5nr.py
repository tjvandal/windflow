'''
Author: TJ Vandal

GMAO G5NR files from Will McCarty in June 2021
Script to generating training data for optical flow. 
'''

import os
import glob
import xarray as xr
import numpy as np
import pandas as pd

from mpi4py import MPI

comm = MPI.COMM_WORLD
MPI_RANK = comm.Get_rank()
MPI_SIZE = comm.Get_size()

def get_files(folder, mode):
    pattern = os.path.join(folder, '*.nc4')
    files = sorted(glob.glob(pattern))

    variables = ['QV', 'U', 'V']
    var_files = {v: sorted([f for f in files if f'_{v}_' in f]) for v in variables}
    var_files = pd.DataFrame(var_files)

    N_files = len(var_files)
    if mode == 'train':
        var_files = var_files.iloc[0:int(N_files*0.7)]
    elif mode == 'valid':
        var_files = var_files.iloc[int(N_files*0.7):int(N_files*0.8)]
    elif mode == 'test':
        var_files = var_files.iloc[int(N_files*0.8):]

    return var_files

def make_training_data(input_folder, output_folder, mode='train', patch_size=560, step=300):
    files = get_files(data_folder, mode)

    for i in np.arange(MPI_RANK, len(files), MPI_SIZE):
        data = xr.open_mfdataset(files.values[i])
        data = data.sel(lat=slice(-80, 80))
        n_lats = data.lat.shape[0]
        n_lons = data.lon.shape[0]
        n_lev = data.lev.shape[0]

        time = data.time.values[0]
        timestamp = (time.astype('uint64') / 1e6).astype('uint32')

        for lev in data.lev.values:
            for lat_idx in np.arange(0, n_lats-patch_size, step):
                for lon_idx in np.arange(0, n_lons-patch_size, step):

                    sub_ds = data.isel(lat=slice(lat_idx, lat_idx+patch_size),
                                       lon=slice(lon_idx, lon_idx+patch_size))\
                                     .sel(lev=lev)
                    lat_0 = sub_ds.lat.values[0]
                    lon_0 = sub_ds.lon.values[0]
                    sub_folder = os.path.join(output_folder, f'{int(lev)}_{lat_0}_{lon_0}')
                    sub_file = os.path.join(sub_folder, str(timestamp).zfill(11) + '.nc4')
                    if os.path.exists(sub_file):
                        continue

                    try:
                        if not os.path.exists(sub_folder):
                            os.makedirs(sub_folder)
                    except IOError:
                        pass

                    sub_ds.to_netcdf(sub_file)
                    print(f"Rank: {MPI_RANK}, Saved patch to file {sub_file}")

if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data', metavar='G5NR_7km/', type=str,
                        help='Where to save your data?')
    parser.add_argument('output', metavar='G5NR_7km_training/', type=str,
                        help='Where to save your data?')
    args = parser.parse_args()


    data_folder = args.data
    training_data_folder = args.output
    make_training_data(data_folder, training_data_folder+'train', mode='train')
    make_training_data(data_folder, training_data_folder+'valid', mode='valid')


