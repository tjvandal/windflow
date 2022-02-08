import xarray as xr
import numpy as np
import datetime as dt

class NOAAAMV(object):
    def __init__(self, file):
        self.file = file
        self.dataset = xr.open_dataset(self.file)
        self.dataset = self.dataset.isel(nMeasures=np.where(np.isfinite(self.dataset.wind_speed.values))[0])
        self.time = dt.datetime(2000, 1, 1, 12) + dt.timedelta(seconds=self.dataset['time_bounds'].values[0], minutes=0)        
        self.wind_speeds =  self.dataset.wind_speed.values
        self.direction = self.dataset.wind_direction.values
        self.lats = self.dataset['lat'].values
        self.lons = self.dataset['lon'].values
        self.vs = np.sin(self.direction / 180 * np.pi) * self.wind_speeds
        self.us = np.cos(self.direction / 180 * np.pi) * self.wind_speeds
