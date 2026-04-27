import torch


def mol_sequence_loss(output, flow_gt, gamma=0.85, max_flow=200):
    """Sequence loss over SEA-RAFT's per-iteration MoL negative log-likelihoods.

    Mirrors upstream SEA-RAFT/core/loss.py:sequence_loss but synthesises the
    validity mask from flow magnitude (windflow datasets do not ship a sparse
    valid channel — see windflow/train/raft_trainer.py).
    """
    n_pred = len(output["flow"])
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid = mag < max_flow

    flow_loss = 0.0
    for i in range(n_pred):
        i_weight = gamma ** (n_pred - i - 1)
        nf_i = output["nf"][i]
        finite = (~torch.isnan(nf_i.detach())) & (~torch.isinf(nf_i.detach()))
        mask = valid[:, None] & finite
        denom = mask.sum().clamp(min=1)
        flow_loss = flow_loss + i_weight * (mask * nf_i).sum() / denom
    return flow_loss
