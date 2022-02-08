import os
import glob
import numpy as np
import xarray as xr

import torch
from torch.utils import data
import cv2

from .. import flow_transforms
from ..preprocess import image_histogram_equalization

class TileFlows(data.Dataset):
    def __init__(self, directory, scale_factor=None, size=None, 
                 augment=False, frames=2, convert_cartesian=False):
        '''
        7-km, 30-min simulation
        Units: m/s
        Scale to cartesian coordinates: 1800 (seconds) / 7000 (m)
        '''
        self.directory = directory
        self.files = np.array(glob.glob(os.path.join(self.directory, '*.nc4')))
        
        filenames = np.array([int(os.path.basename(f.replace('.nc4',''))) for f in self.files])
        filesort = np.argsort(filenames)
        self.files = self.files[filesort]
        
        self.size = size
        self.scale_factor = scale_factor
        self.augment = augment
        self.frames = frames
        self.convert_cartesian = convert_cartesian
        
        self.rand_transform = flow_transforms.Compose([
            flow_transforms.ToTensor(images_order='CHW', flows_order='CHW'),  # Must come before other transforms
            #flow_transforms.RandomRotate(15), # rotate fills in a lot of the image with zeros
            flow_transforms.RandomCrop((size, size)),
            flow_transforms.RandomHorizontalFlip(),
            flow_transforms.RandomVerticalFlip(),
            #flow_transforms.RandomAdditiveColor(0.01),
            #flow_transforms.RandomMultiplicativeColor(0.9, 1.1)
        ])

    def __len__(self):
        return len(self.files)-self.frames+1

    def __getitem__(self, idx):
        try:
            ds = xr.open_mfdataset(self.files[idx:idx+self.frames])
            h, w = ds.U.isel(time=0).shape
            diff = ds.time.values[1] - ds.time.values[0]
            if diff != np.timedelta64(30,'m'):
                #print("Difference in time not 30 minutes", ds.time)
                return self.__getitem__((idx+1) % len(self))

            if self.scale_factor:
                h, w = int(h * self.scale_factor), int(w * self.scale_factor)
                new_lats = np.linspace(ds.lat.values[0], ds.lat.values[-1], h)
                new_lons = np.linspace(ds.lon.values[0], ds.lon.values[-1], w)
                ds = ds.interp(lat=new_lats, lon=new_lons)

            qv = ds['QV'].values
            u = ds['U'].values
            v = ds['V'].values
        except (KeyError, xr.MergeError) as err:
            print("Error in G5NR Loader __getitem__", err)
            return self.__getitem__((idx+self.frames) % len(self))
        except (AttributeError) as err:
            print(f"Error: {err}")
            print(f"Files", self.files[idx:idx+self.frames])
            return self.__getitem__((idx+self.frames) % len(self))
        
        uv = np.concatenate([u[:,np.newaxis], v[:,np.newaxis]], 1) #/ 7000. * (30 * 60)
        uv[~np.isfinite(uv)] = 0. 
        qv[~np.isfinite(qv)] = 0.

        # pixel size differs by latitude
        # see haversine formula -- compute in lon direction, lats are constant

        if self.convert_cartesian:
            #uv = uv * CARTESIAN_SCALE
            
            lat_rad = np.radians(ds.lat.values)
            lon_rad = np.radians(ds.lon.values)
            a = np.cos(lat_rad)**2 * np.sin((lon_rad[1]-lon_rad[0])/2)**2
            d = 2 * 6378.137 * np.arcsin(a**0.5)
            size_per_pixel = np.repeat(np.expand_dims(d, -1), len(lon_rad), axis=1) # kms

            uv = uv / size_per_pixel / 1000 * 1800
        
        qv = image_histogram_equalization(qv)
        
        images = [q[np.newaxis] for q in qv]
        flows = [_uv for _uv in uv]
        images_tensor, flow_tensor = self.rand_transform(images, flows)        
        return images_tensor, flow_tensor
    
    
class G5NRFlows(data.ConcatDataset):
    def __init__(self, directory, scale_factor=None, size=None, augment=False, frames=2, convert_cartesian=True):
        self.directory = directory
        tiles = os.listdir(self.directory)
        tile_paths = [os.path.join(self.directory, t) for t in tiles]
        data.ConcatDataset.__init__(self, [TileFlows(p, scale_factor=scale_factor, size=size, augment=augment, frames=frames, convert_cartesian=convert_cartesian) for p in tile_paths])
        
        
if __name__ == '__main__':
    dataset = G5NRFlows('/nobackupp10/tvandal/nex-ai-opticalflow/data/G5NR_7km_training/train/', size=128)
    print(dataset[0][0].shape)
