import xarray as  xr
import os, sys

sys.path.append('..')

from nexai.datasets.g5nr import g5nr
from dask.distributed import Client


# Set data variables
label_directory = '/nobackupp10/tvandal/nex-ai-opticalflow/data/G5NR_7km/'
label_files = g5nr.get_files(label_directory, 'test').reset_index()

output_directory = '/nobackupp10/tvandal/nex-ai-opticalflow/data/G5NR_7km_zarr/'
print(label_files)

N = len(label_files)
for i, row in label_files.iterrows():
    fname = os.path.basename(row['U'])
    output_zarr = os.path.join(output_directory, fname.split('.')[2] + '.zarr')
    if os.path.exists(output_zarr):
        continue
        
    U = xr.open_mfdataset(row['U'])
    V = xr.open_mfdataset(row['V'])
    ds = xr.open_mfdataset(row['QV']).merge(U).merge(V)
    ds = ds.chunk(dict(time=1, lat=500, lon=500, lev=1))
    ds.to_zarr(output_zarr)
    print(i/N, output_zarr)