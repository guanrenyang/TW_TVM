import pickle
from helper import *
import torch

sparsity_stages = [25,  50,  60,  65,  70,   75,  80, 85,  90, 92, 94, 96]

arch = "resnet50"
pruning_type = "ew"
now = "2022-06-19-14_15_50"
model_dir = root_dir() / "train" / arch / pruning_type / now

for stage in range(len(sparsity_stages)):
    stage_dir = model_dir / ("sparsity_stage_" + str(sparsity_stages[stage]))
    all_layer_dir = stage_dir / "all_layers"
    masks_dir = all_layer_dir / "masks"
    print(sparsity_stages[stage], end=", ")
    with open(masks_dir / ("mask_" + str(sparsity_stages[stage]) + ".pkl"), "rb") as file:
        all_mask_values = pickle.load(file)
        for layer in all_mask_values:
            layer = torch.tensor(layer)
            zero_num = (layer == 0).sum()
            all_num = layer.view(-1).shape[0]
            sparsity = (zero_num / all_num * 100).item()
            print('%0.2f'  %sparsity, end=", ")
        print()