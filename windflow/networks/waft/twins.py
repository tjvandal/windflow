"""Frozen pretrained Twins SVT-Large feature encoder for WAFT.

Vendored from WAFT/model/backbone/twins.py without behavioural change.
"""

import torch.nn as nn
import torch.nn.functional as F
import timm


class ResidualConvUnit(nn.Module):
    def __init__(self, features, activation, bn):
        super().__init__()
        self.bn = bn
        self.groups = 1
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        super().__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = 1
        self.expand = expand
        out_features = features // 2 if expand else features
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        output = xs[0]
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)
        if size is None and self.size is None:
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}
        output = F.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features, nn.ReLU(False), deconv=False, bn=use_bn,
        expand=False, align_corners=True, size=size,
    )


class TwinsFeatureEncoder(nn.Module):
    def __init__(self, frozen=True):
        super().__init__()
        self.backbone = timm.create_model(
            'twins_svt_large',
            pretrained=True,
            features_only=True,
        )
        if frozen:
            self.backbone = self._freeze(self.backbone)

        self.out_channels = [128, 256, 512, 1024]
        self.features = 128
        self.output_dim = self.features // 2
        self.scratch = nn.ModuleList([
            nn.Conv2d(self.out_channels[i], self.features, kernel_size=3,
                      stride=1, padding=1, bias=False)
            for i in range(len(self.out_channels))
        ])
        self.refine = nn.ModuleList([
            _make_fusion_block(self.features, use_bn=False, size=None)
            for _ in range(len(self.out_channels))
        ])
        self.final = nn.ConvTranspose2d(
            self.features, self.features // 2,
            kernel_size=4, stride=2, padding=1, bias=True,
        )

    @staticmethod
    def _freeze(model):
        model = model.eval()
        for p in model.parameters():
            p.requires_grad = False
        for p in model.buffers():
            p.requires_grad = False
        return model

    def forward(self, x):
        out = self.backbone(x)
        out_rn = [self.scratch[i](out[i]) for i in range(len(out))]
        for i in range(1, len(out_rn) + 1):
            if i == 1:
                out_rn[-i] = self.refine[-i](out_rn[-i], size=out_rn[-i].shape[2:])
            else:
                up_feat = F.interpolate(out_rn[-i + 1], scale_factor=2, mode='bilinear', align_corners=True)
                out_rn[-i] = self.refine[-i](out_rn[-i], up_feat, size=out_rn[-i].shape[2:])
        return self.final(out_rn[0])
