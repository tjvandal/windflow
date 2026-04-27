import argparse
import math

import torch
import torch.nn as nn

from .corr import CorrBlock
from .extractor import ResNetFPN
from .layer import conv3x3
from .loss import mol_sequence_loss
from .update import BasicUpdateBlock
from .utils import InputPadder, coords_grid

from ...train.base import BaseTrainer


class SEARAFT(BaseTrainer):
    """SEA-RAFT (Princeton-VL) adapted for windflow's single-channel inputs.

    Inputs are expected in roughly [0, 1] (e.g. histogram-equalized QV); they
    are remapped to [-1, 1] inside forward(). Returns the upstream output
    contract: dict(final, flow, info, nf).
    """

    def __init__(
        self,
        log_step=100,
        iters=4,
        dim=128,
        num_blocks=2,
        radius=4,
        corr_levels=4,
        initial_dim=64,
        block_dims=(64, 128, 256),
        pretrain="resnet18",
        use_var=True,
        var_min=0,
        var_max=10,
        gamma_seq=0.85,
        max_flow=200,
        lr=4e-4,
        scheduler_total_steps=500_000,
    ):
        super().__init__(lr=lr)

        self.log_step = log_step
        self.iters = iters
        self.gamma_seq = gamma_seq
        self.max_flow = max_flow
        self.scheduler_total_steps = scheduler_total_steps

        args = argparse.Namespace()
        args.dim = dim
        args.num_blocks = num_blocks
        args.radius = radius
        args.corr_levels = corr_levels
        args.corr_radius = radius
        args.corr_channel = corr_levels * (radius * 2 + 1) ** 2
        args.iters = iters
        args.pretrain = pretrain
        args.initial_dim = initial_dim
        args.block_dims = list(block_dims)
        args.use_var = use_var
        args.var_min = var_min
        args.var_max = var_max
        self.args = args

        self.cnet = ResNetFPN(
            args, input_dim=2, output_dim=2 * dim, norm_layer=nn.BatchNorm2d
        )
        self.init_conv = conv3x3(2 * dim, 2 * dim)
        self.upsample_weight = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, 64 * 9, 1, padding=0),
        )
        self.flow_head = nn.Sequential(
            nn.Conv2d(dim, 2 * dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * dim, 6, 3, padding=1),
        )
        if iters > 0:
            self.fnet = ResNetFPN(
                args, input_dim=1, output_dim=2 * dim, norm_layer=nn.BatchNorm2d
            )
            self.update_block = BasicUpdateBlock(args, hdim=dim, cdim=dim)

    def initialize_flow(self, img):
        N, _, H, W = img.shape
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords2 = coords_grid(N, H // 8, W // 8, device=img.device)
        return coords1, coords2

    def upsample_data(self, flow, info, mask):
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = torch.nn.functional.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = torch.nn.functional.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, 8 * H, 8 * W), up_info.reshape(N, C, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=None, flow_gt=None, test_mode=False):
        N, _, H, W = image1.shape
        if iters is None:
            iters = self.args.iters
        if flow_gt is None:
            flow_gt = torch.zeros(N, 2, H, W, device=image1.device)

        # windflow inputs are ~[0, 1]; map to ~[-1, 1] like upstream SEA-RAFT
        image1 = (2.0 * image1 - 1.0).contiguous()
        image2 = (2.0 * image2 - 1.0).contiguous()

        flow_predictions = []
        info_predictions = []

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        N, _, H, W = image1.shape
        dilation = torch.ones(N, 1, H // 8, W // 8, device=image1.device)

        cnet = self.cnet(torch.cat([image1, image2], dim=1))
        cnet = self.init_conv(cnet)
        net, _context = torch.split(cnet, [self.args.dim, self.args.dim], dim=1)

        # iter 0: context-only init
        flow_update = self.flow_head(net)
        weight_update = 0.25 * self.upsample_weight(net)
        flow_8x = flow_update[:, :2]
        info_8x = flow_update[:, 2:]
        flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
        flow_predictions.append(flow_up)
        info_predictions.append(info_up)

        if self.args.iters > 0:
            fmap1_8x = self.fnet(image1)
            fmap2_8x = self.fnet(image2)
            corr_fn = CorrBlock(fmap1_8x, fmap2_8x, self.args)

        for _ in range(iters):
            Nb, _, Hf, Wf = flow_8x.shape
            flow_8x = flow_8x.detach()
            coords2 = (coords_grid(Nb, Hf, Wf, device=image1.device) + flow_8x).detach()
            corr = corr_fn(coords2, dilation=dilation)
            net = self.update_block(net, _context, corr, flow_8x)
            flow_update = self.flow_head(net)
            weight_update = 0.25 * self.upsample_weight(net)
            flow_8x = flow_8x + flow_update[:, :2]
            info_8x = flow_update[:, 2:]
            flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)

        for i in range(len(info_predictions)):
            flow_predictions[i] = padder.unpad(flow_predictions[i])
            info_predictions[i] = padder.unpad(info_predictions[i])

        if test_mode:
            return {
                "final": flow_predictions[-1],
                "flow": flow_predictions,
                "info": info_predictions,
                "nf": None,
            }

        nf_predictions = []
        for i in range(len(info_predictions)):
            if not self.args.use_var:
                var_max = var_min = 0
            else:
                var_max = self.args.var_max
                var_min = self.args.var_min

            raw_b = info_predictions[i][:, 2:]
            log_b = torch.zeros_like(raw_b)
            weight = info_predictions[i][:, :2]
            log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
            log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)
            term2 = (
                (flow_gt - flow_predictions[i]).abs().unsqueeze(2)
            ) * (torch.exp(-log_b).unsqueeze(1))
            term1 = weight - math.log(2) - log_b
            nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(
                term1.unsqueeze(1) - term2, dim=2
            )
            nf_predictions.append(nf_loss)

        return {
            "final": flow_predictions[-1],
            "flow": flow_predictions,
            "info": info_predictions,
            "nf": nf_predictions,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=1e-4, eps=1e-8
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.scheduler_total_steps,
            pct_start=0.05,
            cycle_momentum=False,
            anneal_strategy="linear",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def step(self, batch, batch_idx):
        I0 = batch[0][:, 0]
        I1 = batch[0][:, 1]
        labels = batch[1]
        flow_gt = labels[:, 0]
        output = self.forward(I0, I1, iters=self.iters, flow_gt=flow_gt, test_mode=False)
        loss = mol_sequence_loss(
            output, flow_gt, gamma=self.gamma_seq, max_flow=self.max_flow
        )

        if self.global_rank == 0:
            self.log_scalar(loss, "total_loss")
            if self.global_step % self.log_step == 0:
                self.log_flow_grid(flow_gt, "label")
                self.log_image_grid(I0, "data/I0")
                self.log_image_grid(I1, "data/I1")
                flows = output["flow"]
                stride = max(1, len(flows) // 4)
                for i in range(0, len(flows), stride):
                    self.log_flow_grid(flows[i], f"flows/iter_{i}")
        return loss
