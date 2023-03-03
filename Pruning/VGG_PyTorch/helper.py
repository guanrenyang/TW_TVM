import shutil
import torch
import torch.nn as nn
import os
import copy
from pathlib import Path
import torchvision.models as models
from pruning_modules import Pruner, Conv2dPruner, LinearPruner
from multihead_attention import MultiheadAttentionPruner
import pruning_algo
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, filename='alex_checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'alex_model_best.pth')


def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def root_dir() -> Path:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return Path(current_dir)


def ensure_dir(path: Path) -> Path:
    path = os.path.abspath(path)
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except FileExistsError:
            pass
    return Path(path)


def load_mask(model, mask):
    id = 0
    for name, module in model.named_modules():
        if isinstance(module, Pruner):
            module.load_mask(mask[id], name)
            id += 1
        
def pruning_info(model):
    id = 0
    non_zeros = 0
    lengths = 0
    for name, module in model.named_modules():
        if isinstance(module, Pruner):
            mask = module.dump_mask()
            mask = np.array(mask.flatten().tolist())
            non_zero = np.sum(mask)

            print("  %d %s   " %(id, name), end="\t")
            print("    %d" %(len(mask) - non_zero), end="\t")
            print("    %d" %(non_zero), end="\t")
            print("    %d" %(len(mask)), end="\t")
            print("    %f" %(non_zero/len(mask)), end="\t")
            print(mask.shape, end="\t")
            print()

            non_zeros += non_zero
            lengths += len(mask)
            id += 1
            
    print("ALL Density  : %f " %(non_zeros/lengths))
    print("ALL Sparsity : %f " %(1- non_zeros/lengths))

def update_mask(model, sparsity, pruning_type, masks_now = None):
    id = 0
    scores = []
    names = []
    for name, module in model.named_modules():
        if isinstance(module, Pruner):
            if id in masks_now:
                scores.append(module.dump_weight())
            id += 1

    new_accumulated_scores = pruning_algo.img2col_forward(scores)
    new_mask_values = pruning_algo.pruning_fun(pruning_type)(new_accumulated_scores, sparsity)
    new_mask_values = pruning_algo.img2col_back_ward(new_mask_values, scores) 

    mask_id = 0
    id = 0
    for name, module in model.named_modules():
        if isinstance(module, Pruner):
            if id in masks_now:
                module.load_mask(new_mask_values[mask_id], name)
                mask_id += 1
            id += 1


def dump_mask(model):
    masks = []
    for name, module in model.named_modules():
        if isinstance(module, Pruner):
            masks.append(module.dump_mask())
        
    return masks

def get_model(arch):
    if arch == "inception_v3":
        return models.inception_v3(aux_logits=False,weights=True)
    else:
        return models.__dict__[arch](weights=True)


def pruning_model(model):
    """
    Recursively quantize a pretrained single-precision model to int8 quantized model
    model: pretrained single-precision model
    """
    # quantize convolutional and linear layers to 8-bit
    if type(model) == nn.Conv2d:
        quant_mod = Conv2dPruner()
        quant_mod.set_param(model)
        return quant_mod
    elif type(model) == nn.Linear:
        quant_mod = LinearPruner()
        quant_mod.set_param(model)
        return quant_mod
    elif type(model) == nn.MultiheadAttention:
        quant_mod = MultiheadAttentionPruner()
        quant_mod.set_param(model)
        return quant_mod
    # recursively use the quantized module to replace the single-precision module
    elif type(model) == nn.Sequential:
        mods = []
        for n, m in model.named_children():
            mods.append(pruning_model(m))
        return nn.Sequential(*mods)
    elif type(model) == nn.ModuleList:
        mods = []
        for n, m in model.named_children():
            mods.append(pruning_model(m))
        return nn.Sequential(*mods)
    elif isinstance(model, nn.Sequential):
        mods = []
        for n, m in model.named_children():
            mods.append(pruning_model(m))
        return nn.Sequential(*mods)
    else:
        q_model = copy.deepcopy(model)
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module):
                setattr(q_model, attr, pruning_model(mod))
        return q_model