#!/bin/bash
DATASET_DIR="/home/cguo/imagenet-raw-data/"
GPU_ID=${1}
pruning_type="tw1"
batch_size=192
finetune_steps=10000
mini_finetune_steps=10000
# score_type="weight"
# #score_type="taylor"
init_checkpoint=""
pre_masks_dir=""
comment=${2}
hostname=`hostname`
# arch="vgg16"
# arch="resnet18"
arch="resnet50"
# arch="vit_b_16"

log_output_dir="./${arch}_${hostname}_${pruning_type}_${batch_size}_${finetune_steps}_${mini_finetune_steps=10000}_${score_type}_gpu_${GPU_ID}_${comment}.log"
    command="
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -u main.py ${DATASET_DIR} -a ${arch} -p --prune\
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