#!/usr/bin/env bash
# mode: pretrain or prune
# cuda_device: use which GPU, 0 or 1 or ...
bert_base_dir='./Model/uncased_L-12_H-768_A-12'
SQUAD_DIR="./squad_data"
task_name='SQUAD'
mode=${1}
cuda_device=0
granularity=64
pruning_type=998
pretrained_path=./Model/${task_name}_pretrained_model
STARTTIME=$( date '+%F')
# STARTTIME="2022-06-01"
previous_spar=0
number=10
dim=64
g1percent=0
if [[ "${cuda_device}" = "0" || "${cuda_device}" = "1" || "${cuda_device}" = "2" || "${cuda_device}" = "3" ]]
then    
    if [ "$mode" = "pretrain" ]
    then

        is_pruning_mode=false
		output_dir=${pretrained_path}/model_${run}
        init_dir=$bert_base_dir/bert_model.ckpt
        mkdir -p ${output_dir}

        command="
        CUDA_VISIBLE_DEVICES=${cuda_device} python run_squad.py --task_name=${task_name}  \
        --vocab_file=$bert_base_dir/vocab.txt \
        --bert_config_file=$bert_base_dir/bert_config.json \
        --init_checkpoint=${init_dir} \
        --do_train=True \
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

    elif [ "$mode" = "prune" ]
    then
        for spar in  25 50 65 75 85 95; do
        # for spar in 20	30	37	45	50  55  61 68 75; do
        # for spar in 26 36; do
            #for spar in 25 50 62.5 75 87.5 93.75; do 
            for run in 0  ; do
                is_pruning_mode=true
                output_dir="${task_name}_pruning_${spar}%_granularity_${granularity}_taylor_score_pruning_type_${pruning_type}_${STARTTIME}_${number}_g1p_${g1percent}/model_${run}"
                if [[ "${previous_spar}" = "0" ]]
                then
                    init_dir=${pretrained_path}/model_${run}
                else
                    init_dir="${task_name}_pruning_${previous_spar}%_granularity_${granularity}_taylor_score_pruning_type_${pruning_type}_${STARTTIME}_${number}_g1p_${g1percent}/model_${run}"
                fi
                mkdir -p ${output_dir}
                cp ${init_dir}/*.tf_record ${output_dir}
                if [[ "${previous_spar}" != "0" ]]
                then
                    cp ${init_dir}/model.* ${init_dir}/checkpoint ${output_dir}/
                fi

                command="
                CUDA_VISIBLE_DEVICES=${cuda_device} python run_squad.py --task_name=${task_name}  \
                --vocab_file=$bert_base_dir/vocab.txt \
                --bert_config_file=$bert_base_dir/bert_config.json \
                --init_checkpoint=${init_dir} \
                --do_train=True \
                --do_predict=True \
                --train_file=$SQUAD_DIR/train-v1.1.json \
                --predict_file=$SQUAD_DIR/dev-v1.1.json \
                --train_batch_size=12 \
                --learning_rate=1e-5 \
                --num_train_epochs=2.0 \
                --max_seq_length=384 \
                --sparsity=${spar} \
                --mode=${mode} \
                --granularity=${granularity} \
                --pruning_type=${pruning_type} \
                --output_dir="${output_dir}" \
                --block_remain=${number} \
                --skip_block_dim=${dim} \
                --g1percent=${g1percent} \
                2>&1 | tee ${output_dir}/log.txt
                "
                echo $command | tee ${output_dir}/cmd.txt
                eval $command
            done 
            previous_spar=${spar}
        done
    fi
else
    echo "No use any GPU. Need to use one GPU."
fi