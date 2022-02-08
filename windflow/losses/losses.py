import torch
import torch.nn as nn

class SmoothnessLoss(nn.Module):
    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def forward(self, flows, mask=None):
        gradx = torch.abs(flows[:,:,:,:-1] - flows[:,:,:,1:])
        grady = torch.abs(flows[:,:,:-1,:] - flows[:,:,1:,:])
        normalize = 1.
        if isinstance(mask, torch.Tensor):
            gradx *= mask[:,:,:,:-1]
            grady *= mask[:,:,:-1,:]
            normalize = torch.mean(mask)
        loss_smooth = (torch.mean(gradx) + torch.mean(grady)) / normalize
        return loss_smooth 

class CharbonnierLoss(nn.Module):
    def __init__(self, alpha, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, I0, I1, mask=None):
        err = ((I0 - I1)**2 + self.eps)**self.alpha
        norm = 1.
        if isinstance(mask, torch.Tensor):
            err *= mask
            norm = torch.mean(mask)
        return torch.mean(err) / norm
    
def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean()

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()

    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue

class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue

class L1Loss(nn.Module):
    def __init__(self, args):
        super(L1Loss, self).__init__()
        self.args = args
        self.loss = L1()
        self.loss_labels = ['L1', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        #epevalue = EPE(output, target)
        return lossvalue
        #return [lossvalue, epevalue]
    
class L2Loss(nn.Module):
    def __init__(self, args):
        super(L2Loss, self).__init__()
        self.args = args
        self.loss = L2()

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        #epevalue = EPE(output, target)
        return lossvalue #+ epevalue
    
    
class RMSVDLoss(nn.Module):
    def __init__(self):
        super(RMSVDLoss, self).__init__()
    
    def forward(self, output, target):
        # output and target shape (N, 2, H, W)
        u_err = (output[:,0] - target[:,0])**2
        v_err = (output[:,1] - target[:,1])**2
        rmsvd = (u_err.mean() + v_err.mean())**0.5
        return rmsvd
