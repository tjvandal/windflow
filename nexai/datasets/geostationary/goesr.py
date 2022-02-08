import os, sys
import glob
import xarray as xr
import pandas as pd
import numpy as np
import dask as da
import datetime as dt
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pyproj
import pyresample
from .. import utils

def get_filename_metadata(f):
    channel = int(f.split('_')[1][-2:])
    spatial = f.split('-')[2]
    t1 = f.split('_')[3]
    year = int(t1[1:5])
    dayofyear = int(t1[5:8])
    hour = int(t1[8:10])
    minute = int(t1[10:12])
    second = int(t1[12:15])
    return dict(channel=channel, year=year, dayofyear=dayofyear, hour=hour,
                minute=minute, second=second, spatial=spatial)

class L1bBand(object):
    def __init__(self, fpath):
        self.fpath = fpath
        meta = get_filename_metadata(self.fpath)
        self.band = meta['channel']
        self.year = meta['year']
        self.dayofyear = meta['dayofyear']
        self.hour = meta['hour']
        self.minute = meta['minute']
        self.second = meta['second']
        self.spatial = meta['spatial']
        self.datetime = dt.datetime(self.year, 1, 1, self.hour, self.minute, self.second//10) +\
                dt.timedelta(days=self.dayofyear-1)

    def open_dataset(self, rescale=True, force=False, chunks=None):
        if (not hasattr(self, 'data')) or force:
            ds = xr.open_dataset(self.fpath, chunks=chunks)
            ds = ds.where(ds.DQF.isin([0, 1]))
            band = ds.band_id[0]
            # normalize radiance
            if rescale:
                radiance = ds['Rad']
                if band <= 6:
                    ds['Rad'] = ds['Rad'] * ds.kappa0
                else:
                    fk1 = ds.planck_fk1.values
                    fk2 = ds.planck_fk2.values
                    bc1 = ds.planck_bc1.values
                    bc2 = ds.planck_bc2.values
                    tmp = fk1 / ds["Rad"] + 1
                    tmp = np.where(tmp > 0, tmp, 1)
                    T = (fk2/(np.log(tmp))-bc1)/bc2
                    radiance.values = T
                ds['Rad'] = radiance
            self.data = ds
        return self.data

    def plot(self, ax=None, cmap=None, norm=None):
        if not hasattr(self, 'data'):
            self.open_dataset()

        # Satellite height
        sat_h = self.data['goes_imager_projection'].perspective_point_height
        # Satellite longitude
        sat_lon = self.data['goes_imager_projection'].longitude_of_projection_origin

        # The geostationary projection
        x = self.data['x'].values * sat_h
        y = self.data['y'].values * sat_h
        if ax is None:
            fig = plt.figure(figsize=[10,10])
            ax = fig.add_subplot(111)
            
        m = Basemap(projection='geos', lon_0=sat_lon, resolution='i',
                     rsphere=(6378137.00,6356752.3142),
                     llcrnrx=x.min(),llcrnry=y.min(),
                     urcrnrx=x.max(),urcrnry=y.max(),
                     ax=ax)
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()
        ax.set_title('GOES-16 -- Band {}'.format(self.band), fontweight='semibold', loc='left')
        ax.set_title('%s' % self.datetime.strftime('%H:%M UTC %d %B %Y'), loc='right')
        return m.imshow(self.data['Rad'].values[::-1], cmap=cmap, norm=norm)
        #return m
        
    def plot_infrared(self, ax=None, colortable='ir_drgb_r', colorbar=False, cmin=180, cmax=270):
        from metpy.plots import colortables
        ir_norm, ir_cmap = colortables.get_with_range('WVCIMSS', cmin, cmax)
        # Use a colortable/colormap available from MetPy
        # ir_norm, ir_cmap = colortables.get_with_range(colortable, 190, 350)
        im = self.plot(ax, cmap=ir_cmap, norm=ir_norm)
        if colorbar:
            plt.colorbar(im, pad=0.01, aspect=50, ax=ax, shrink=0.85, 
                         ticks=range(cmin,cmax,10), label='Temperature (K)')
        return im
    
    def interp(self, scale):
        if not hasattr(self, 'data'):
            self.open_dataset()

        new_x = np.linspace(self.data.x.values[0], self.data.x.values[-1], 
                            int(self.data.x.values.shape[0] * scale))
        new_y = np.linspace(self.data.y.values[0], self.data.y.values[-1], 
                            int(self.data.y.values.shape[0] * scale))
        self.data = self.data.interp(x=new_x, y=new_y)
        
    def latlon(self):
        if not hasattr(self, 'lats'):
            if not hasattr(self, 'data'):
                self.open_dataset()
            # Satellite height
            sat_h = self.data['goes_imager_projection'].perspective_point_height
            # Satellite longitude
            sat_lon = self.data['goes_imager_projection'].longitude_of_projection_origin
            sat_sweep= self.data['goes_imager_projection'].sweep_angle_axis
            p = pyproj.Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)
            X = self.data['x'].values * sat_h
            Y = self.data['y'].values * sat_h
            XX, YY = np.meshgrid(X, Y)
            lons, lats = p(XX, YY, inverse=True)
            self.lats = lats
            self.lons = lons
            NANs = np.isnan(self.data['Rad'].values)
            self.lats[NANs] = np.nan
            self.lons[NANs] = np.nan
            self.lats[~np.isfinite(self.lats)] = np.nan
            self.lons[~np.isfinite(self.lons)] = np.nan
        return self.lats, self.lons

    def latlon_lookup(self, lat, lon):
        self.latlon()
        if (lat > self.lats.min()) and (lat < self.lats.max()) and (lon > self.lons.min()) and (lon < self.lons.max()):
            dist = ((self.lats - lat)**2 + (self.lons - lon)**2)**0.5
            ix, iy = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
            return ix, iy
        
        
    def reproject_to_latlon(self):
        data = self.open_dataset()
        lats, lons = self.latlon()
        sat_lon = data['goes_imager_projection'].longitude_of_projection_origin
        
        if self.band in [1,3,4,5,6]:
            s = 0.01
        elif self.band == 2:
            s = 0.005
        else:
            s = 0.02
        
        lat_min = np.around(max(np.nanmin(lats), -60), 3)
        lat_max = np.around(min(np.nanmax(lats), 60), 3)
        lon_min = np.around(max(np.nanmin(lons), sat_lon-60), 3)
        lon_max = np.around(min(np.nanmax(lons), sat_lon+60), 3)
        lats_new = np.arange(lat_min, lat_max, s)
        lons_new = np.arange(lon_min, lon_max, s)
        lons_new, lats_new = np.meshgrid(lons_new, lats_new)
        
        source_def = pyresample.geometry.SwathDefinition(lats=lats, lons=lons)
        target_def = pyresample.geometry.GridDefinition(lons=lons_new, lats=lats_new)
        result = pyresample.kd_tree.resample_nearest(source_def, 
                                                     data['Rad'].values,
                                                     target_def, 
                                                     radius_of_influence=50000, 
                                                     epsilon=0.05)                                                              
        data_new = xr.DataArray(result, coords=dict(lat=lats_new[:,0], lon=lons_new[0,:]), dims=('lat', 'lon'))
        return xr.Dataset(dict(Rad=data_new))


class GroupBandTemporal(object):
    def __init__(self, data_list):
        self.data = data_list
        self.open_dataset()

    def open_dataset(self):
        for b in self.data:
            if not hasattr(b, 'data'):
                b.open_dataset()

    def get_radiances(self, indices=None):
        self.open_dataset()
        xequal = np.array_equal(self.data[0].data.x.values,
                       self.data[-1].data.x.values)
        yequal = np.array_equal(self.data[0].data.y.values,
                       self.data[-1].data.y.values)
        if (not xequal) or (not yequal):
            return

        if indices is None:
            indices = range(len(self.data))
        data = xr.concat([self.data[i].data['Rad'] for i in indices], 'time')
        return data

    def get_radiance_patches(self, patch_size):
        data = self.get_radiances()
        if data is None:
            return

        data = utils.block_array(data, 2,
                             size=patch_size,
                             stride=patch_size)
        data = utils.block_array(data, 2,
                                 size=patch_size,
                                 stride=patch_size)
        data= data.reshape(-1, len(self.data),
                                patch_size,
                                patch_size)
        return data

    def add(self, band):
        self.data.append(band)

    def __len__(self):
        return len(self.data)

    def timeseries(self, ix, iy):
        self.open_dataset()
        data = np.array([b.data['Rad'].isel(x=ix, y=iy).values for b in self.data])
        x = [b.datetime for b in self.data]
        return x, data

    def timeseries_latlon(self, lat, lon):
        print("Lat: {}, Lon: {}".format(lat, lon))
        self.open_dataset()
        indices = [b.latlon_lookup(lat, lon) for b in self.data]
        data = []
        x = []
        for i, b in enumerate(self.data):
            if indices[i] is None:
                print("Data bounds: ({}, {}), ({}, {})".format(b.lats.min(), b.lats.max(), b.lons.min(),
                                               b.lons.max()))
                continue
            data.append(b.data['Rad'].isel(x=indices[i][0], y=indices[i][1]).values)
            x.append(b.datetime)
        data = np.array(data)
        x = np.array(x)
        return x, data

class GOESL1b(object):
    def __init__(self, product='ABI-L1b-RadF',
                       channels=range(1,17),
                       data_directory='/nex/datapool/geonex/public/GOES16/NOAA-L1B/'):
        self.product = product
        self.channels = channels
        self.data_directory = os.path.join(data_directory, product)

    def local_files(self, year=None, dayofyear=None, hour=None, spatial=None):
        data = []
        base_dir = self.data_directory

        if year is None:
            year = "*"
        else:
            year = '%04i' % year
        if dayofyear is None:
            dayofyear = "*"
        else:
            dayofyear = '%03i' % dayofyear
        if hour is None:
            hour = "*"
        else:
            hour = '%02i' % hour 

        for c in self.channels:
            path_pattern = os.path.join(self.data_directory, year, dayofyear, hour,
                                        'OR_ABI-L1b-*C%02i_*.nc' % c)
            channel_files = glob.glob(path_pattern)
            for f in channel_files:

                meta = get_filename_metadata(os.path.basename(f))
                meta['file'] = f
                data.append(meta)

        data = pd.DataFrame(data)
        if (len(data) > 0) and (spatial is not None):
            data = data[data['spatial'] == spatial]

        if len(data) > 0:
            new_index = ['year', 'dayofyear', 'hour', 'minute', 'second', 'spatial']
            data = data.set_index(new_index)
            data = data.pivot(columns='channel').drop_duplicates()

        return data

    def snapshot_file(self, year, dayofyear, hour, minute, spatial=None):
        t = dt.datetime(year, 1, 1, hour, minute) + dt.timedelta(days=dayofyear-1)
        t1 = t + dt.timedelta(hours=1)
        curr_time = t
        files = []
        while curr_time <= t1:
            files.append(self.local_files(curr_time.year, curr_time.timetuple().tm_yday, 
                                          curr_time.hour, spatial=spatial))
            curr_time = curr_time + dt.timedelta(hours=1)
        files = pd.concat(files)

        if spatial is None:
            spatial = files.index.get_level_values('spatial')[0]

        files = files.reset_index()
        
        def to_dt(row):
            tt = dt.datetime(row['year'], 1, 1, row['hour'], row['minute'], row['second']/10.) + dt.timedelta(days=row['dayofyear'].values[0]-1)
            return tt
            
        times = files.apply(to_dt, axis=1)
        
        idx = np.argmin(np.abs(times-t))
        return files.iloc[idx]['file']        

    def snapshot_files(self, year, dayofyear, hour, minute, spatial=None):
        hour_files = self.local_files(year, dayofyear, hour, spatial=spatial)

        minutes = hour_files.index.get_level_values('minute')
        if spatial is None:
            spatial = hour_files.index.get_level_values('spatial')[0]

        idx = np.argmin(np.abs(minutes-minute))
        
        minute_sel = minutes[idx]
        return hour_files.loc[year, dayofyear, hour, minute_sel]['file']        

    
    def open_snapshot(self, year, dayofyear, hour, minute, spatial=None):
        snapshot_files = self.snapshot_files(year, dayofyear, hour, minute, spatial=spatial)
        rads = []
        regrid = utils.regrid_1km
        for c in self.channels:
            band_file = snapshot_files[c].iloc[0]
            l1b = L1bBand(band_file)
            data = l1b.open_dataset()
            rad_c = data['Rad']
            rad_c = rad_c.expand_dims(dict(band=[c]))
            rad_c_regrid = regrid(rad_c, c)
            rads.append(rad_c_regrid)
            rads[-1] = rads[-1].assign_coords(x=rads[0].x.values,
                                              y=rads[0].y.values)

        rad = xr.concat(rads, 'band')
        #rad = rad.swapaxes(0, 1) # put time in front
        return rad
