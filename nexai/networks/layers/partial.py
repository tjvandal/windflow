import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d_partial(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, partial=False):
        super(Conv2d_partial, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.partial = partial

    def forward(self, input):
        if self.partial:
            self.padding = 0

            pad_val = (self.kernel_size[0] - 1) // 2
            if pad_val > 0:
                if (self.kernel_size[0] - self.stride[0]) % 2 == 0:
                    pad_top = pad_val
                    pad_bottom = pad_val
                    pad_left = pad_val
                    pad_right = pad_val
                else:
                    pad_top = pad_val
                    pad_bottom = self.kernel_size[0] - self.stride[0] - pad_top
                    pad_left = pad_val
                    pad_right = self.kernel_size[0] - self.stride[0] - pad_left

                p0 = torch.ones_like(input)
                p0 = p0.sum()

                input = F.pad(input, (pad_left, pad_right, pad_top, pad_bottom) , mode='constant', value=0)

                p1 = torch.ones_like(input)
                p1 = p1.sum()

                ratio = torch.div(p1, p0 + 1e-8) 
                input = torch.mul(input, ratio)  
            
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
