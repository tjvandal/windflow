import os
import xarray as xr
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms

from . import goesr
from . import stats

from .. import flow_transforms
from ..preprocess import image_histogram_equalization

class L1bPatches(data.Dataset):
    def __init__(self, data_directory, time_step=1, size=512, bands=[9,], mode='train'):
        self.patch_folders = [os.path.join(data_directory, f) for f in os.listdir(data_directory)]
        
        self.patches = [L1bPatch(f, time_step=time_step, size=size, bands=bands, mode=mode) for f in self.patch_folders]
        self.patches = data.ConcatDataset(self.patches)
        self.N = len(self.patches)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.patches[idx]

class L1bPatch(data.Dataset):
    def __init__(self, data_directory, time_step=1, size=512, bands=[9,], mode='train'):
        self.files = sorted([os.path.join(data_directory, f) for f in os.listdir(data_directory)])
        
        N = len(self.files)
        if mode == 'train':
            self.files = self.files[:int(N*0.8)]
        elif mode == 'valid':
            self.files = self.files[int(N*0.8):int(N*0.9)]
        elif mode == 'test':
            self.files = self.files[int(N*0.9):]
        
        self.N = len(self.files)-time_step
        self.time_step = time_step
        self.bands = bands
        self.rand_transform = transforms.Compose([
            transforms.RandomCrop((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        sample_files = [self.files[idx], self.files[idx+self.time_step]]
        N_samples = len(sample_files)
        N_bands = len(self.bands)
        
        ds = [xr.open_dataset(f) for f in sample_files]
        
        x = np.concatenate([d['Rad'].sel(band=self.bands).values for d in ds], 0)
        x[~np.isfinite(x)] = 0.
        
        mask = torch.Tensor((x[0] == x[0]).copy()).float()
        
        x = image_histogram_equalization(x)
        x = self.rand_transform(torch.Tensor(x))
        x = x.unsqueeze(1)
        #x = [x[i:i+N_bands] for i in range(N_samples)]
        return x, mask

class L1bResized(data.Dataset):
    def __init__(self, data_directory, time_step=1, new_size = (1024, 1024), jitter=6, band=9):
        self.files = sorted([os.path.join(data_directory, f) for f in os.listdir(data_directory)])
        self.N = len(self.files)-time_step
        self.time_step = time_step
        self.jitter = jitter
        self.new_size = new_size
        jitter_size = (new_size[0] + jitter, new_size[1] + jitter)
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize(jitter_size),
                                            transforms.ToTensor()])
        self.mu, self.sd = stats.get_sensor_stats("ABI")        
        self.mu = self.mu[band-1]
        self.sd = self.sd[band-1]
        
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        sample_files = [self.files[idx], self.files[idx+self.time_step]]
        print(sample_files)
        samples = []
        for f in sample_files:
            if (f[-3:] == '.nc') or (f[-4:] == '.nc4'):
                data = xr.open_dataset(f)
                x = data['Rad'].values.astype(np.float32)
                x = np.copy(x)
            elif f[-3:] == 'npy':
                x = np.load(f).astype(np.float32)
            x = (x - self.mu) / self.sd
            x[~np.isfinite(x)] = 0.
            x = self.transform(x)
            samples.append(x)

        #if np.random.uniform() > 0.5:
        #    samples = [torch.flipud(s) for s in samples]
        #if np.random.uniform() > 0.5:
        #    samples = [torch.fliplr(s) for s in samples]
        #if np.random.uniform() > 0.5:
        #    samples = [torch.rot90(s, 1, [1, 2]) for s in samples]        
    
        if self.jitter > 0:
            ix = np.random.choice(range(self.jitter))
            iy = np.random.choice(range(self.jitter))
            samples = [s[:,ix:ix+self.new_size[0],iy:iy+self.new_size[1]] for s in samples]
    
        #factor1 = np.random.uniform(0.9,1.1)
        #factor2 = np.random.uniform(0.9,1.1)
        #samples = [s*factor1 for s in samples]
        #samples[1] = samples[1]*factor2
        mask = (samples[0] != 0).float()
        return samples, mask


