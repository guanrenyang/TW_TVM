#!/bin/bash
DATASET_DIR= "/home/fcxue/pycode/TileSparsity-master/Pruning/VGG/imagenet/"
#DATASET_DIR= "/home/cguo/SCTest/Pruning/VGG/imagenet/"
#GPU_ID=${0}
GPU_ID=0
pruning_type="ew"
batch_size=64
finetune_steps=5000
mini_finetune_steps=5000
score_type="weight"
#score_type="taylor"
init_checkpoint=""
pre_masks_dir=""
# init_checkpoint="/home/fcxue/pycode/TileSparsity-master/Pruning/VGG/init_checkpoint"
# pre_masks_dir="/home/fcxue/pycode/TileSparsity-master/Pruning/VGG/pre_masks_dir"
comment=${2}
hostname=`hostname`

log_output_dir="./${hostname}_${pruning_type}_${batch_size}_${finetune_steps}_${mini_finetune_steps=10000}_${score_type}_gpu_${GPU_ID}_${comment}.log"
    command="
    CUDA_VISIBLE_DEVICES=${GPU_ID} python pruning.py \
        --batch_size=$batch_size \
        --finetune_steps=${finetune_steps} \
        --mini_finetune_steps=${mini_finetune_steps} \
        --score_type=${score_type} \
        --pruning_type=${pruning_type} \
        --init_checkpoint=${init_checkpoint} \
        --pre_masks_dir=${pre_masks_dir} \
        --data_dir=${DATASET_DIR} \
        2>&1 | tee ${log_output_dir}
    "
echo $command
echo $log_output_dir
eval $command
