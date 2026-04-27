import os
import re
import glob
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import xbatcher

import torch
from torch.utils import data

from .. import flow_transforms
from ..preprocess import image_histogram_equalization


_FNAME_RE = re.compile(r'_(QV|U|V)_Nv\.(\d{8}_\d{4}z)\.nc4$')

# Sidecar caching the full (timestamp, QV, U, V) index. Saves a tree-walk
# over NCCS layouts (~minutes for 2 years × 365 × 48 × 3 = 105k files).
_INDEX_CACHE_NAME = '.g5nr_file_index.parquet'


def _scan_files(directory):
    """Walk `directory` once and return a DataFrame keyed by timestamp."""
    files = glob.glob(os.path.join(
        directory, 'inst30mn_3d_*_Nv', 'Y*', 'M*', 'D*', '*.nc4'))
    if not files:
        files = glob.glob(os.path.join(directory, '*.nc4'))

    rows = {}
    for f in files:
        m = _FNAME_RE.search(os.path.basename(f))
        if not m:
            continue
        var, ts = m.group(1), m.group(2)
        rows.setdefault(ts, {})[var] = f
    if not rows:
        return pd.DataFrame(columns=['timestamp', 'QV', 'U', 'V'])
    df = pd.DataFrame.from_dict(rows, orient='index').sort_index()
    df = df.dropna(subset=['QV', 'U', 'V'])
    return (df.rename_axis('timestamp').reset_index()
              [['timestamp', 'QV', 'U', 'V']])


def _index_files_by_timestamp(directory, years=None, cache_path=None):
    """Discover paired (QV, U, V) .nc4 files under `directory`.

    Supports two layouts:
      - flat:   directory/c1440_NR.inst30mn_3d_{VAR}_Nv.{YYYYMMDD}_{HHMM}z.nc4
      - NCCS:   directory/inst30mn_3d_{VAR}_Nv/Y{YYYY}/M{MM}/D{DD}/c1440_NR.*.nc4
        e.g. /css/g5nr/Ganymed/7km/c1440_NR/DATA/0.0625_deg/inst

    `years`: optional iterable of int years (e.g. [2005, 2006]) to filter
    after scanning. Filter is applied in-memory so a single cache covers any
    year subset.

    `cache_path`: parquet sidecar to read/write. Default
    `<directory>/.g5nr_file_index.parquet`. Pass ``False`` to disable caching;
    if the path is on a read-only filesystem the cache is silently skipped.
    Delete the file to force a rescan.
    """
    if cache_path is None:
        cache_path = os.path.join(directory, _INDEX_CACHE_NAME)

    df = None
    if cache_path and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
        except Exception as exc:
            warnings.warn(
                f'Could not read file index cache {cache_path!r}: {exc}; '
                f'will rescan.')
            df = None

    if df is None:
        df = _scan_files(directory)
        if cache_path and len(df) > 0:
            try:
                df.to_parquet(cache_path)
            except (OSError, PermissionError) as exc:
                warnings.warn(
                    f'Could not write file index cache to {cache_path!r}: '
                    f'{exc}; loader will rescan on each invocation.')

    if len(df) == 0:
        return pd.DataFrame(columns=['QV', 'U', 'V'])

    if years:
        wanted = {int(y) for y in years}
        df = df[df['timestamp'].str[:4].astype(int).isin(wanted)]
    return df[['QV', 'U', 'V']].reset_index(drop=True)


class G5NRXBatcherFlows(data.Dataset):
    """xbatcher-backed G5NR loader that reads spatial patches lazily from
    raw global QV/U/V .nc4 files. No offline patch preprocessing required.

    Yields the same tensor contract as TileFlows:
        images: [frames, 1, size, size]   (histogram-equalized QV)
        flows : [frames, 2, size, size]   (U, V)
    """

    def __init__(self, directory, mode='train', size=128, frames=2,
                 scale_factor=None, convert_cartesian=True,
                 lat_bounds=(-80, 80), overlap=0, levels=None, years=None):
        self.directory = directory
        self.mode = mode
        self.size = size
        self.frames = frames
        self.scale_factor = scale_factor
        self.convert_cartesian = convert_cartesian
        self.lat_bounds = lat_bounds
        self.levels = levels
        self.years = years

        df = _index_files_by_timestamp(directory, years=years)
        if len(df) == 0:
            raise RuntimeError(f'No QV/U/V .nc4 triples found in {directory}')

        n = len(df)
        if mode == 'train':
            df = df.iloc[:int(n * 0.7)]
        elif mode == 'valid':
            df = df.iloc[int(n * 0.7):int(n * 0.8)]
        elif mode == 'test':
            df = df.iloc[int(n * 0.8):]
        else:
            raise ValueError(f'mode must be train/valid/test, got {mode!r}')
        self.df = df.reset_index(drop=True)

        if len(self.df) < frames:
            raise RuntimeError(
                f'mode={mode!r} has {len(self.df)} timesteps but frames={frames}')

        probe = xr.open_dataset(self.df.iloc[0]['QV'], engine='h5netcdf')
        full_lat = probe.lat.values
        self.lev_values = probe.lev.values.copy()
        n_lev_total = probe.sizes['lev']
        probe.close()

        if self.levels is None:
            self._lev_indices = list(range(n_lev_total))
        else:
            self._lev_indices = [int(i) for i in self.levels]
            for i in self._lev_indices:
                if i < 0 or i >= n_lev_total:
                    raise ValueError(
                        f'level {i} out of range [0, {n_lev_total})')
        self.n_lev = len(self._lev_indices)

        if lat_bounds is not None:
            lat_keep = (full_lat >= lat_bounds[0]) & (full_lat <= lat_bounds[1])
            lat_idx = np.where(lat_keep)[0]
            self._lat_start = int(lat_idx[0])
            self._lat_stop = int(lat_idx[-1]) + 1
            self.lat_values = full_lat[self._lat_start:self._lat_stop].copy()
        else:
            self._lat_start = 0
            self._lat_stop = len(full_lat)
            self.lat_values = full_lat.copy()
        h = self._lat_stop - self._lat_start

        probe = xr.open_dataset(self.df.iloc[0]['QV'], engine='h5netcdf')
        self.lon_values = probe.lon.values.copy()
        w = probe.sizes['lon']
        probe.close()

        self._handles = None

        template = xr.DataArray(
            np.zeros((h, w), dtype=np.float32),
            dims=('lat', 'lon'),
            coords={'lat': self.lat_values, 'lon': self.lon_values},
        )
        bgen = xbatcher.BatchGenerator(
            template,
            input_dims={'lat': size, 'lon': size},
            input_overlap={'lat': overlap, 'lon': overlap},
        )
        self.patch_slices = []
        for patch in bgen:
            lat0 = int(np.searchsorted(self.lat_values, patch.lat.values[0]))
            lon0 = int(np.searchsorted(self.lon_values, patch.lon.values[0]))
            self.patch_slices.append((lat0, lon0))

        self.n_pairs = len(self.df) - frames + 1

        self.aug = flow_transforms.Compose([
            flow_transforms.ToTensor(images_order='CHW', flows_order='CHW'),
            flow_transforms.RandomHorizontalFlip(),
            flow_transforms.RandomVerticalFlip(),
        ])

    def __len__(self):
        return self.n_pairs * self.n_lev * len(self.patch_slices)

    def _decode_index(self, idx):
        n_patches = len(self.patch_slices)
        per_pair = self.n_lev * n_patches
        pair_idx, rem = divmod(idx, per_pair)
        lev_idx, patch_idx = divmod(rem, n_patches)
        return pair_idx, lev_idx, patch_idx

    def _ensure_handles(self):
        if self._handles is not None:
            return
        self._handles = {
            'QV': [xr.open_dataset(f, engine='h5netcdf')['QV'] for f in self.df['QV']],
            'U':  [xr.open_dataset(f, engine='h5netcdf')['U']  for f in self.df['U']],
            'V':  [xr.open_dataset(f, engine='h5netcdf')['V']  for f in self.df['V']],
        }

    def _read_var(self, var, pair_idx, lev_idx, lat0, lon0):
        lat_start = self._lat_start + lat0
        lat_stop = lat_start + self.size
        out = []
        for f in range(self.frames):
            arr = self._handles[var][pair_idx + f].isel(
                time=0, lev=lev_idx,
                lat=slice(lat_start, lat_stop),
                lon=slice(lon0, lon0 + self.size),
            ).values
            out.append(arr)
        return np.stack(out)

    def __getitem__(self, idx):
        self._ensure_handles()
        pair_idx, lev_idx, patch_idx = self._decode_index(idx)
        phys_lev = self._lev_indices[lev_idx]
        lat0, lon0 = self.patch_slices[patch_idx]

        try:
            qv = self._read_var('QV', pair_idx, phys_lev, lat0, lon0)
            u = self._read_var('U', pair_idx, phys_lev, lat0, lon0)
            v = self._read_var('V', pair_idx, phys_lev, lat0, lon0)

            t0 = self._handles['QV'][pair_idx]['time'].values[0]
            t1 = self._handles['QV'][pair_idx + 1]['time'].values[0]
            if t1 - t0 != np.timedelta64(30, 'm'):
                return self.__getitem__((idx + 1) % len(self))

            lat_vals = self.lat_values[lat0:lat0 + self.size]
            lon_vals = self.lon_values[lon0:lon0 + self.size]

            if self.scale_factor:
                # Simple bilinear via xarray on a tiny dataset; rare path.
                ds_tmp = xr.Dataset(
                    {'QV': (('time', 'lat', 'lon'), qv),
                     'U':  (('time', 'lat', 'lon'), u),
                     'V':  (('time', 'lat', 'lon'), v)},
                    coords={'lat': lat_vals, 'lon': lon_vals,
                            'time': np.arange(self.frames)},
                )
                hh = int(self.size * self.scale_factor)
                ww = int(self.size * self.scale_factor)
                new_lats = np.linspace(lat_vals[0], lat_vals[-1], hh)
                new_lons = np.linspace(lon_vals[0], lon_vals[-1], ww)
                ds_tmp = ds_tmp.interp(lat=new_lats, lon=new_lons)
                qv = ds_tmp['QV'].values
                u = ds_tmp['U'].values
                v = ds_tmp['V'].values
                lat_vals = new_lats
                lon_vals = new_lons
        except (KeyError, AttributeError) as err:
            print(f'G5NRXBatcherFlows __getitem__ error: {err}')
            return self.__getitem__((idx + len(self.patch_slices)) % len(self))

        uv = np.concatenate([u[:, np.newaxis], v[:, np.newaxis]], 1)
        uv[~np.isfinite(uv)] = 0.
        qv[~np.isfinite(qv)] = 0.

        if self.convert_cartesian:
            lat_rad = np.radians(lat_vals)
            lon_rad = np.radians(lon_vals)
            a = np.cos(lat_rad) ** 2 * np.sin((lon_rad[1] - lon_rad[0]) / 2) ** 2
            d = 2 * 6378.137 * np.arcsin(a ** 0.5)
            size_per_pixel = np.repeat(np.expand_dims(d, -1), len(lon_rad), axis=1)
            uv = uv / size_per_pixel / 1000 * 1800

        qv = image_histogram_equalization(qv)

        images = [q[np.newaxis] for q in qv]
        flows = [_uv for _uv in uv]
        return self.aug(images, flows)


if __name__ == '__main__':
    ds = G5NRXBatcherFlows('/home/ubuntu/windflow/data/g5nr',
                           mode='train', size=128, frames=2)
    print('len:', len(ds))
    imgs, flows = ds[0]
    print('images:', imgs.shape, imgs.dtype)
    print('flows :', flows.shape, flows.dtype)
