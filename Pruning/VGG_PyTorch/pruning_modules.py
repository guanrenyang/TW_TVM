import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import  _pair
from torch.nn import init

def _reverse_repeat_tuple(t, n):
    r"""Reverse the order of `t` and repeat each element for `n` times.
    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in reversed(t) for _ in range(n))


class Pruner_autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask):
        return input * mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class Pruner(nn.Module):
    def __init__(self):
        super(Pruner, self).__init__()

        self.mask = None
        self.is_enable = False
        self.name = None
        self.sparsity = 0
        self.weight = None

    def load_mask(self, mask_value, name):
        with torch.no_grad():
            self.mask = torch.tensor(mask_value, dtype=torch.float).cuda()
            self.name = name
            self.is_enable = True

    def dump_mask(self):
        with torch.no_grad():
            if self.mask == None:
                self.mask = torch.ones_like(self.weight).cuda()
            return self.mask.cpu().numpy()

    def dump_weight(self):
        with torch.no_grad():
            return self.weight.cpu().numpy()
    
    def _prune(self, tensor):
        return Pruner_autograd.apply(tensor, self.mask)
        # return tensor * self.mask

    def _forward(self, tensor):
        if self.is_enable:
            tensor = self._prune(tensor)
        return tensor

class TensorPruner(Pruner):
    def __init__(self):
        super(TensorPruner, self).__init__()

    def forward(self, tensor):
        return self._forward(tensor)

# class Conv2dPruner(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size,
#         stride=1,
#         padding=0,
#         dilation=1,
#         groups=1,
#         bias=True,
#         padding_mode='zeros',
#         ):
#         super(Conv2dPruner, self).__init__()

#         kernel_size = _pair(kernel_size)
#         stride = _pair(stride)
#         padding = _pair(padding)
#         dilation = _pair(dilation)
        
#         if in_channels % groups != 0:
#             raise ValueError('in_channels must be divisible by groups')
#         if out_channels % groups != 0:
#             raise ValueError('out_channels must be divisible by groups')
#         valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
#         if padding_mode not in valid_padding_modes:
#             raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
#                 valid_padding_modes, padding_mode))
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups
#         self.padding_mode = padding_mode
#         self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
        
#         self.weight = nn.Parameter(torch.Tensor(
#             out_channels, in_channels // groups, *kernel_size))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#         self.prune_weight = TensorPruner()

#     def reset_parameters(self):
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(self.bias, -bound, bound)

#     def extra_repr(self):
#         s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
#              ', stride={stride}')
#         if self.padding != (0,) * len(self.padding):
#             s += ', padding={padding}'
#         if self.dilation != (1,) * len(self.dilation):
#             s += ', dilation={dilation}'
#         if self.groups != 1:
#             s += ', groups={groups}'
#         if self.bias is None:
#             s += ', bias=False'
#         if self.padding_mode != 'zeros':
#             s += ', padding_mode={padding_mode}'
#         return s.format(**self.__dict__)

#     def __setstate__(self, state):
#         super(_ConvNd, self).__setstate__(state)
#         if not hasattr(self, 'padding_mode'):
#             self.padding_mode = 'zeros'
    
#     def _conv_forward(self, input, weight):
#         if self.padding_mode != 'zeros':
#             return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
#                             weight, self.bias, self.stride,
#                             _pair(0), self.dilation, self.groups)
#         return F.conv2d(input, weight, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)

#     def forward(self, input):
#         weight = prune_weight._forward(self.weight)
#         return self._conv_forward(input, weight)


class Conv2dPruner(nn.Module):
    """
    Class to quantize given convolutional layer
    """
    def __init__(self):
        super(Conv2dPruner, self).__init__()
        self.prune_weight = TensorPruner()

    def set_param(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight = nn.Parameter(conv.weight.data.clone())
        self.prune_weight.weight = self.weight
        try:
            self.bias = nn.Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None

    def _conv_forward(self, input, weight):
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        weight = self.prune_weight(self.weight)
        return self._conv_forward(input, weight)

class LinearPruner(nn.Module):
    """
    Class to quantize given linear layer
    """
    def __init__(self, mode=None, wbit=None, abit=None, args=None):
        super(LinearPruner, self).__init__()
        self.prune_weight = TensorPruner()


    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.data.clone())
        self.prune_weight.weight = self.weight
        try:
            self.bias = nn.Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, input): 
        weight = self.prune_weight(self.weight) 
        return F.linear(input, weight, self.bias)