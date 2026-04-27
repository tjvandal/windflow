"""WAFT (Princeton-VL) adapted for windflow's single-channel inputs.

The upstream WAFTv2 expects 3-channel RGB in [0, 255] and ImageNet-
normalises internally. windflow inputs are single-channel histogram-
equalised QV in ~[0, 1]; this wrapper replicates the channel 3x and
scales to [0, 255] so the frozen pretrained twins encoder receives a
valid (R=G=B) input. The wrapper otherwise preserves the upstream
forward contract: dict(flow, info, nf).
"""

import argparse

import torch
import torch.nn as nn

from .core import WAFTv2
from .loss import waft_sequence_loss

from ...train.base import BaseTrainer


class WAFT(BaseTrainer):
    def __init__(
        self,
        log_step=100,
        iters=4,
        feature_encoder='twins',
        iterative_module='vits',
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
        args.feature_encoder = feature_encoder
        args.iterative_module = iterative_module
        args.iters = iters
        args.var_min = var_min
        args.var_max = var_max
        self.args = args

        self.core = WAFTv2(args)

    def _prep_inputs(self, image):
        """[N,1,H,W] in ~[0,1] -> [N,3,H,W] in [0,255] (R=G=B)."""
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        return image * 255.0

    def forward(self, image1, image2, iters=None, flow_gt=None):
        image1 = self._prep_inputs(image1)
        image2 = self._prep_inputs(image2)
        return self.core(image1, image2, iters=iters, flow_gt=flow_gt)

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
            anneal_strategy='linear',
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }

    def step(self, batch, batch_idx):
        I0 = batch[0][:, 0]
        I1 = batch[0][:, 1]
        labels = batch[1]
        flow_gt = labels[:, 0]
        output = self.forward(I0, I1, iters=self.iters, flow_gt=flow_gt)
        loss = waft_sequence_loss(
            output, flow_gt, gamma=self.gamma_seq, max_flow=self.max_flow
        )

        if self.global_rank == 0:
            self.log_scalar(loss, 'total_loss')
            if self.global_step % self.log_step == 0:
                self.log_flow_grid(flow_gt, 'label')
                self.log_image_grid(I0, 'data/I0')
                self.log_image_grid(I1, 'data/I1')
                flows = output['flow']
                stride = max(1, len(flows) // 4)
                for i in range(0, len(flows), stride):
                    self.log_flow_grid(flows[i], f'flows/iter_{i}')
        return loss
