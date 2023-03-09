#!/bin/bash
# DATASET_DIR="/home/cguo/imagenet-raw-data/"
GPU_ID=${1}
pruning_type="tw1"
batch_size=256
finetune_steps=10000
mini_finetune_steps=10000
# score_type="weight"
# score_type="taylor"
init_checkpoint=""
pre_masks_dir=""
comment=${2}
hostname=`hostname`

# architecture and dataset 
arch="resnet18"
dataset="cifar10"

log_output_dir="./${arch}_${dataset}_${pruning_type}_${hostname}_${batch_size}_${finetune_steps}_${mini_finetune_steps=10000}_gpu_${comment}.log"
    command="
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -u pruner.py \
        -a ${arch} \
        -d ${dataset} \
        --pruning_type=${pruning_type} \
        --pretrained \
        --lr=5e-5 \
        --batch-size=$batch_size \
        --finetune_steps=${finetune_steps} \
        --mini_finetune_steps=${mini_finetune_steps} \
        --resume=${init_checkpoint} \
        --pre_masks_dir=${pre_masks_dir} \
        2>&1 | tee ${log_output_dir}
    "
        # --score_type=${score_type} \
echo $command
echo $log_output_dir
eval $command