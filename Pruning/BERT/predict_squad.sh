#!/usr/bin/env bash
# mode: pretrain or prune
# cuda_device: use which GPU, 0 or 1 or ...
bert_base_dir='./Model/uncased_L-12_H-768_A-12'
SQUAD_DIR="./squad_data"
task_name='SQUAD'
mode="export"
cuda_device=0
granularity=128
pruning_type=1000
pretrained_path=./Model/${task_name}_tmp
STARTTIME=$( date '+%F')
STARTTIME="2022-05-31"
previous_spar=75
number=10
dim=64
g1percent=0
if [[ "${cuda_device}" = "0" || "${cuda_device}" = "1" || "${cuda_device}" = "2" || "${cuda_device}" = "3" ]]
then    
		output_dir=${pretrained_path}/model_0
        init_dir="${task_name}_pruning_${previous_spar}%_granularity_${granularity}_taylor_score_pruning_type_${pruning_type}_${STARTTIME}_${number}_g1p_${g1percent}/model_0"
        mkdir -p ${output_dir}

        command="
        CUDA_VISIBLE_DEVICES=${cuda_device} python run_squad.py --task_name=${task_name}  \
        --vocab_file=$bert_base_dir/vocab.txt \
        --bert_config_file=$bert_base_dir/bert_config.json \
        --init_checkpoint=${init_dir} \
        --do_train=False \
        --do_predict=True \
        --train_file=$SQUAD_DIR/train-v1.1.json \
        --predict_file=$SQUAD_DIR/dev-v1.1.json \
        --train_batch_size=32 \
        --learning_rate=2e-5 \
        --num_train_epochs=1.0 \
        --max_seq_length=128 \
        --mode=${mode} \
        --output_dir="${output_dir}" \
        2>&1 | tee ${output_dir}/log.txt
        "
        echo $command | tee ${output_dir}/cmd.txt
        eval $command    
else
    echo "No use any GPU. Need to use one GPU."
fi