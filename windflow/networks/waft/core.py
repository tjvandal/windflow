"""Upstream WAFTv2 model.

Vendored from WAFT/model/waft_a2.py with imports rewired to local modules
and dav2/dinov3 encoder branches removed (twins-only). Behaviour of the
WAFTv2 forward pass is unchanged; the channel/scale adaptation for
windflow's single-channel inputs lives in waft.py (the LightningModule
wrapper) so this file stays a faithful port.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .twins import TwinsFeatureEncoder
from .vit import VisionTransformer, MODEL_CONFIGS
from .utils import coords_grid, Padder, bilinear_sampler


class _ResConv(nn.Module):
    def __init__(self, inp, oup, k=3, s=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(inp, oup, kernel_size=k, stride=s, padding=k // 2, bias=True),
            nn.GELU(),
            nn.Conv2d(oup, oup, kernel_size=3, stride=1, padding=1, bias=True),
        )
        if inp != oup or s != 1:
            self.skip_conv = nn.Conv2d(inp, oup, kernel_size=1, stride=s, padding=0, bias=True)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.skip_conv(x)


class _ResNet18Deconv(nn.Module):
    def __init__(self, inp, oup):
        super().__init__()
        self.feature_dims = [64, 128, 256, 512]
        self.ds1 = _ResConv(inp, 64, k=7, s=2)
        self.conv1 = _ResConv(64, 64, k=3, s=1)
        self.conv2 = _ResConv(64, 128, k=3, s=2)
        self.conv3 = _ResConv(128, 256, k=3, s=2)
        self.conv4 = _ResConv(256, 512, k=3, s=2)
        self.up_4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, bias=True)
        self.proj_3 = _ResConv(256, 256, k=3, s=1)
        self.up_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, bias=True)
        self.proj_2 = _ResConv(128, 128, k=3, s=1)
        self.up_2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, bias=True)
        self.proj_1 = _ResConv(64, oup, k=3, s=1)

    def forward(self, x):
        out_1 = self.ds1(x)
        out_1 = self.conv1(out_1)
        out_2 = self.conv2(out_1)
        out_3 = self.conv3(out_2)
        out_4 = self.conv4(out_3)
        out_3 = self.proj_3(out_3 + self.up_4(out_4))
        out_2 = self.proj_2(out_2 + self.up_3(out_3))
        out_1 = self.proj_1(out_1 + self.up_2(out_2))
        return [out_1, out_2, out_3, out_4]


class WAFTv2(nn.Module):
    """Twins-only WAFTv2.

    `args` must expose: feature_encoder ('twins'), iterative_module
    (vits/vitb/vitl/vitt), iters (int), var_min, var_max.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.feature_encoder != 'twins':
            raise ValueError(
                f'windflow WAFT port supports feature_encoder=twins only, '
                f'got {args.feature_encoder!r}'
            )
        self.encoder = TwinsFeatureEncoder(frozen=True)
        self.factor = 32

        self.pretrain_dim = self.encoder.output_dim
        self.fnet = _ResNet18Deconv(3, self.pretrain_dim)
        self.iter_dim = MODEL_CONFIGS[args.iterative_module]['features']
        self.refine_net = VisionTransformer(args.iterative_module, self.iter_dim, patch_size=8)
        self.fmap_conv = nn.Conv2d(self.pretrain_dim * 2, self.iter_dim, kernel_size=1)
        self.hidden_conv = nn.Conv2d(self.iter_dim * 2, self.iter_dim, kernel_size=1)
        self.warp_linear = nn.Conv2d(3 * self.iter_dim + 2, self.iter_dim, 1)
        self.refine_transform = nn.Conv2d(self.iter_dim // 2 * 3, self.iter_dim, 1)
        self.upsample_weight = nn.Sequential(
            nn.Conv2d(self.iter_dim, 2 * self.iter_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * self.iter_dim, 4 * 9, 1),
        )
        self.flow_head = nn.Sequential(
            nn.Conv2d(self.iter_dim, 2 * self.iter_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * self.iter_dim, 6, 1),
        )

    def upsample_data(self, flow, info, mask):
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 2, 2, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(2 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, 2 * H, 2 * W), up_info.reshape(N, C, 2 * H, 2 * W)

    @staticmethod
    def normalize_image(img):
        """Inputs (B,C,H,W) in [0,255], RGB; outputs ImageNet-normalized."""
        tf = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
        return tf(img / 255.0).contiguous()

    def forward(self, image1, image2, iters=None, flow_gt=None):
        if iters is None:
            iters = self.args.iters
        image1 = self.normalize_image(image1)
        image2 = self.normalize_image(image2)
        padder = Padder(image1.shape, factor=self.factor)
        image1 = padder.pad(image1)
        image2 = padder.pad(image2)

        N, _, H, W = image1.shape
        fmap1_pretrain = self.encoder(image1)
        fmap2_pretrain = self.encoder(image2)
        fmap1_img = self.fnet(image1)[0]
        fmap2_img = self.fnet(image2)[0]
        fmap1_2x = self.fmap_conv(torch.cat([fmap1_pretrain, fmap1_img], dim=1))
        fmap2_2x = self.fmap_conv(torch.cat([fmap2_pretrain, fmap2_img], dim=1))
        net = self.hidden_conv(torch.cat([fmap1_2x, fmap2_2x], dim=1))
        flow_2x = torch.zeros(N, 2, H // 2, W // 2, device=image1.device)

        flow_predictions, info_predictions = [], []
        for _ in range(iters):
            flow_2x = flow_2x.detach()
            coords2 = (coords_grid(N, H // 2, W // 2, device=image1.device) + flow_2x).detach()
            warp_2x = bilinear_sampler(fmap2_2x, coords2.permute(0, 2, 3, 1))
            refine_inp = self.warp_linear(torch.cat([fmap1_2x, warp_2x, net, flow_2x], dim=1))
            refine_outs = self.refine_net(refine_inp)
            net = self.refine_transform(torch.cat([refine_outs['out'], net], dim=1))
            flow_update = self.flow_head(net)
            weight_update = 0.25 * self.upsample_weight(net)
            flow_2x = flow_2x + flow_update[:, :2]
            info_2x = flow_update[:, 2:]
            flow_up, info_up = self.upsample_data(flow_2x, info_2x, weight_update)
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)

        for i in range(len(info_predictions)):
            flow_predictions[i] = padder.unpad(flow_predictions[i])
            info_predictions[i] = padder.unpad(info_predictions[i])

        if flow_gt is not None:
            nf_predictions = []
            for i in range(len(info_predictions)):
                raw_b = info_predictions[i][:, 2:]
                log_b = torch.zeros_like(raw_b)
                weight = info_predictions[i][:, :2]
                log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=self.args.var_max)
                log_b[:, 1] = torch.clamp(raw_b[:, 1], min=self.args.var_min, max=0)
                term2 = ((flow_gt - flow_predictions[i]).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
                term1 = weight - math.log(2) - log_b
                nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(
                    term1.unsqueeze(1) - term2, dim=2)
                nf_predictions.append(nf_loss)
            return {'flow': flow_predictions, 'info': info_predictions, 'nf': nf_predictions}

        return {'flow': flow_predictions, 'info': info_predictions}
