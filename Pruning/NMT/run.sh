#!/usr/bin/env bash

for type in "tw1" "twvw"; do

task_name="nmt_${type}"
cuda_device=0
granularity=128
STARTTIME=$( date '+%F')
previous_spar=0
num_train_epochs=1
pruning_type=${type}


if [[ "${pruning_type}" = "vw" ]];
then 
    granularity=16
fi
if [[ "${pruning_type}" = "bw" ]];
then 
    granularity=64
fi

for spar in 25 37.5 50 62.5 75 87.5 93.75; do

    output_dir="./${task_name}_output/pruning_${spar}%_granularity_${granularity}_type_${pruning_type}_${STARTTIME}"

    if [[ "${previous_spar}" = "0" ]];
    then
        init_dir="./envi_model_1"		
    else
        init_dir="./${task_name}_output/pruning_${previous_spar}%_granularity_${granularity}_type_${pruning_type}_${STARTTIME}/best_bleu/"
    fi

    if [[ "${pruning_type}" = "vw" ]];
    then
        init_dir="./envi_model_1"
    fi

    if [[ "${pruning_type}" = "ew" ]];
    then
        init_dir="./envi_model_1"
    fi

    mkdir -p ${output_dir}

    command="
        CUDA_VISIBLE_DEVICES=${cuda_device} python -m nmt.nmt \
        --src=en --tgt=vi \
        --ckpt=${init_dir} \
        --hparams_path=nmt/standard_hparams/iwslt15.json \
        --vocab_prefix=./data_set/vocab \
        --train_prefix=./data_set/train \
        --dev_prefix=./data_set/tst2012 \
        --test_prefix=./data_set/tst2013 \
        --out_dir=${output_dir}/ \
        --override_loaded_hparams=true \
        --sparsity=${spar} \
        --pruning_type=${pruning_type} \
        2>&1 | tee ${output_dir}/log.txt 
    "
    echo $command | tee ${output_dir}/cmd.txt
    eval $command
    
    previous_spar=${spar}
done
done