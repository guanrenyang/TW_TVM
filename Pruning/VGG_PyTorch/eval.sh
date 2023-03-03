#!/bin/bash
DATASET_DIR="~/imagenet-raw-data/"
GPU_ID=${1}
pruning_type="ew"
batch_size=1024
finetune_steps=10000
mini_finetune_steps=10000
init_checkpoint="/home/cguo/cguo/TileSparsity2.0/Pruning/VGG_PyTorch/train/resnet18/ew/2022-06-10-23_57_48/sparsity_stage_50/all_layers/ckpt_50.pth"
pre_masks_dir="/home/cguo/cguo/TileSparsity2.0/Pruning/VGG_PyTorch/train/resnet18/ew/2022-06-10-23_57_48/sparsity_stage_50/all_layers/masks/good_mask_50.pkl"
comment=${2}
hostname=`hostname`
# arch="vgg16"
arch="resnet18"
# arch="resnet50"
# arch="vit_b_16"

log_output_dir="./${arch}_${hostname}_${pruning_type}_${batch_size}_${finetune_steps}_${mini_finetune_steps=10000}_${score_type}_gpu_${GPU_ID}_${comment}.log"
    command="
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -u main.py ${DATASET_DIR} -a ${arch} -p --eval\
        --lr=5e-5 \
        --batch-size=$batch_size \
        --finetune_steps=${finetune_steps} \
        --mini_finetune_steps=${mini_finetune_steps} \
        --pruning_type=${pruning_type} \
        --resume=${init_checkpoint} \
        --pre_masks_dir=${pre_masks_dir} \
        2>&1 | tee ${log_output_dir}
    "

        # --score_type=${score_type} \
echo $command
echo $log_output_dir
eval $command