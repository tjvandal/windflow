import torch
import torch.nn.functional as F


class Padder:
    """Pads images so spatial dims are divisible by `factor` (twins=32, etc.)."""

    def __init__(self, dims, mode='sintel', factor=32):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht + 8) // factor) + 1) * factor - self.ht
        pad_wd = (((self.wd + 8) // factor) + 1) * factor - self.wd
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2,
                         pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, x):
        return F.pad(x, self._pad, mode='constant', value=0)

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(
        torch.arange(ht, device=device),
        torch.arange(wd, device=device),
        indexing='ij',
    )
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """grid_sample wrapper using pixel coordinates."""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        m = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, m.float()
    return img
