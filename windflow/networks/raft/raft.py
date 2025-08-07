import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock, SmallUpdateBlock
from .extractor import BasicEncoder, SmallEncoder
from .corr import CorrBlock, AlternateCorrBlock
from .utils.utils import coords_grid, upflow8
from ...train.base import BaseTrainer
from ...train.raft_trainer import sequence_loss

try:
    autocast = torch.cuda.amp.autocast
except AttributeError:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class RAFT(BaseTrainer):
    def __init__(
        self,
        log_step=100,
        iters=12,
        small=False,
        lr=1e-4,
        dropout=0.0,
        alternate_corr=False,
    ):
        super().__init__(lr=lr)
        self.dropout = dropout
        self.log_step = log_step
        self.iters = iters
        # super(RAFT, self).__init__()
        # self.args = args

        if small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            self.corr_levels = 4
            self.corr_radius = 3

        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            self.corr_levels = 4
            self.corr_radius = 4

        self.alternate_corr = alternate_corr

        # feature network, context network, and update block
        if small:
            self.fnet = SmallEncoder(
                input_dim=1, output_dim=128, norm_fn="instance", dropout=self.dropout
            )
            self.cnet = SmallEncoder(
                input_dim=1,
                output_dim=hdim + cdim,
                norm_fn="none",
                dropout=self.dropout,
            )
            self.update_block = SmallUpdateBlock(
                corr_radius=self.corr_radius,
                corr_levels=self.corr_levels,
                hidden_dim=hdim,
            )
        else:
            self.fnet = BasicEncoder(
                input_dim=1, output_dim=256, norm_fn="instance", dropout=self.dropout
            )
            self.cnet = BasicEncoder(
                input_dim=1,
                output_dim=hdim + cdim,
                norm_fn="batch",
                dropout=self.dropout,
            )
            self.update_block = BasicUpdateBlock(
                corr_radius=self.corr_radius,
                corr_levels=self.corr_levels,
                hidden_dim=hdim,
            )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(
        self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False
    ):
        """Estimate optical flow between pair of frames"""

        # image1 = 2 * (image1 / 255.0) - 1.0
        # image2 = 2 * (image2 / 255.0) - 1.0

        # image1 = image1.contiguous()
        # image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        # with autocast(enabled=self.args['mixed_precision']):
        fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        # run the context network
        # with autocast(enabled=self.args['mixed_precision']):
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)
        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1.float())  # index correlation volume

            flow = coords1 - coords0
            # with autocast(enabled=self.args['mixed_precision']):
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return (flow_up,)

        return flow_predictions

    def configure_optimizers(self):
        # set optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=1e-4, eps=1e-8
        )
        return optimizer

    # def step(self, inputs, labels, flow_init=None, log=False, train=True):
    def step(self, batch, batch_idx):
        I0 = batch[0][:, 0]
        I1 = batch[0][:, 1]
        labels = batch[1]
        flow_predictions = self.forward(I0, I1, iters=self.iters, flow_init=None)
        loss = sequence_loss(flow_predictions, labels[:, 0])

        # self.scheduler.step()
        if self.global_rank == 0:
            self.log_scalar(loss, "total_loss")
            if self.global_step % self.log_step == 0:
                self.log_flow_grid(labels[:, 0], "label")
                self.log_image_grid(I0, "data/I0")
                self.log_image_grid(I1, "data/I1")

                for i in range(0, self.iters, 2):
                    self.log_flow_grid(flow_predictions[i], f"flows/iter_{i}")
        return loss
