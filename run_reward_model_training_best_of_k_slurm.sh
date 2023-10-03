#!/bin/bash
# scripts to train reward model by observing the difference towards ground truth
#SBATCH --mail-user=chenghao@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/net/scratch/chenghao/fm2/slurm_output/%A_%a.%N.stdout
#SBATCH --error=/net/scratch/chenghao/fm2/slurm_output/%A_%a.%N.stderr
#SBATCH --chdir=/net/scratch/chenghao/fm2/slurm_output
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:4
#SBATCH --job-name=run_decsum_agent_training
#SBATCH --nodes=1
#SBATCH --mem=100gb
#SBATCH --exclude=g[006,009]
#SBATCH --ntasks=4
#SBATCH --time=11:00:00
#SBATCH --signal=SIGUSR1@120

#export PATH="/home/chenghao/miniconda3/bin:$PATH"
#export HF_HOME=/net/scratch/chenghao/transformers
#export TRANSFORMERS_CACHE=/net/scratch/chenghao/transformers
#alias sgpu="bash ~/slurm_interactive.sh"
#alias scpu="bash ~/slurm_interactive_cpu.sh"
#export PATH="/home/chenghao/miniconda3/bin:$PATH"
#conda init bash
echo $PATH
source ~/miniconda3/etc/profile.d/conda.sh
cd /net/scratch/chenghao/fm2
conda activate /home/chenghao/env37
root_dir="/net/scratch/chenghao/fm2/pragsum_dataset/AgentTraining_domain_wise"
readarray -t domains < <(find "${root_dir}" -maxdepth 1 -type d ! -path "${root_dir}" -exec basename {} \;)
#printf "%s\n" "${domains[@]}"
#slurm_id=2
#echo "test-${domains[${slurm_id}]}"
domain=${domains[${SLURM_ARRAY_TASK_ID}]}
output_root_dir=${root_dir}
epoch=3
#for domain in "Toronto" "Phoenix"
#for domain in "Las_Vegas" "Phoenix" "Toronto"
#for domain in "Montreal" "Champaign" "Pittsburgh" "Las_Vegas" "Toronto"
#for domain_dir in $root_dir/*
#do
#    #for suffix in "_wo_summary" "_w_summary"
#    domain=$(basename -- "${domain_dir}")
for suffix in "_wo_summary"
do
    if [[ ${suffix} == "_wo_summary" ]]
    then
        text_column="text"
    else
        text_column="summary_longt5"
    fi
#domain="Toronto"
#--text_column text \
#--train_file ${root_dir}/${domain}/train.json \
#--validation_file ${root_dir}/${domain}/dev.json \
#--test_file ${root_dir}/${domain}/test.json \
#--text_column summary_longt5 \
#--train_file ${root_dir}/${domain}/train_w_summary.json \
#--validation_file ${root_dir}/${domain}/dev_w_summary.json \
#--test_file ${root_dir}/${domain}/test_w_summary.json \
      #--text_column ${text_column} \
      #--train_file ${root_dir}/${domain}/train_w_summary.json \
      #--validation_file ${root_dir}/${domain}/dev_w_summary.json \
      #--test_file ${root_dir}/${domain}/test_w_summary.json \
      #--label_column rank \
      #--output_dir ${output_dir}/${domain}/agent_training_output_epoch3_${suffix}
      #--overwrite_output_dir \
          #--model_name_or_path google/bigbird-roberta-large \
      #--output_dir ./agent_training_output_epoch3_filter_chain_rest_train${suffix}/city/${domain}
          #--train_file ${root_dir}/${domain}/agent_difference_dataset/train_difference.json \
          #--label_column rank \
    output_dir=${output_root_dir}/${domain}/agent_difference_dataset/validation_set_dbt_agent_training_output_epoch${epoch}${suffix}
    if [[ ! -f ${output_dir}/pytorch_model.bin ]]; then
        deepspeed --master_port 6100${SLURM_ARRAY_TASK_ID} agent_training_slurm.py \
          --model_name_or_path microsoft/deberta-v3-large \
          --do_train \
          --do_eval \
          --do_predict \
          --deepspeed ./ds_config_zero2.json \
          --bf16 \
          --label_column difference \
          --train_file ${root_dir}/${domain}/agent_difference_dataset/dev_difference.json \
          --validation_file ${root_dir}/${domain}/agent_difference_dataset/dev_difference.json \
          --test_file ${root_dir}/${domain}/agent_difference_dataset/test_difference.json \
          --max_seq_length 2048 \
          --max_eval_samples 2000 \
          --save_steps 2000 \
          --eval_steps 2000 \
          --per_device_train_batch_size 2 \
          --learning_rate 2e-5 \
          --num_train_epochs ${epoch} \
          --save_total_limit 5 \
          --load_best_model_at_end True \
          --evaluation_strategy steps \
          --output_dir ${output_dir}
    else
        echo "${output_dir} already exists, skipping.."
    fi
done
#done
