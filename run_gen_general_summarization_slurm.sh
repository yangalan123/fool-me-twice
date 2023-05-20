#!/bin/bash
#SBATCH --mail-user=chenghao@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/net/scratch/chenghao/yelp/slurm_output/%A_%a.%N.stdout
#SBATCH --error=/net/scratch/chenghao/yelp/slurm_output/%A_%a.%N.stderr
#SBATCH --chdir=/net/scratch/chenghao/yelp/slurm_output
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:4
#SBATCH --job-name=run_decsum_agent_training
#SBATCH --nodes=1
#SBATCH --mem=100gb
#SBATCH --exclude=g[006,009]
#SBATCH --ntasks=4
#SBATCH --time=11:59:00
#SBATCH --signal=SIGUSR1@120
echo $PATH
source ~/miniconda3/etc/profile.d/conda.sh
cd /net/scratch/chenghao/yelp/50reviews
conda activate /home/chenghao/env37
#root_dir="/data/yelp/50reviews/pragsum_dataset/city"
root_dir="/net/scratch/chenghao/yelp/processed_data_all_business_filter_values_category/latest--1reviews-ranking--1/pragsum_filter_chain_filter_values_category_final_review_level_mutual_exclusive/custom"
readarray -t domains < <(find "${root_dir}" -maxdepth 1 -type d ! -path "${root_dir}" -exec basename {} \;)
domain=${domains[${SLURM_ARRAY_TASK_ID}]}
output_root_dir=${root_dir}
num_output_sample=1
num_output_sample_per_batch=1
suffix="summary"
#for domain in "Toronto" "Phoenix"
#for domain in "Las_Vegas" "Phoenix" "Toronto"
#for domain in "Montreal" "Champaign" "Pittsburgh" "Scarborough"
##for domain in "Toronto"
#do
#model="google/long-t5-tglobal-xl"
#model="google/flan-t5-xl"
#model="google/flan-t5-base"
#model_type=${model##*/}
model="/net/scratch/chenghao/yelp/processed_data_all_business_filter_values_category/latest--1reviews-ranking--1/pragsum_filter_chain_filter_values_category_final_review_level_mutual_exclusive/custom/${domain}/rl_agent_threshold_reward_epoch1_wo_summary"
model_type=${model##*/}_w_prefix
# split the model name by / and get last item
for split in 'train' 'dev' 'test'
#for split in 'test'
do
  output_dir=${output_root_dir}/${domain}/${epoch}${suffix}
  deepspeed run_summarization.py \
    --model_name_or_path ${model} \
    --do_predict \
    --deepspeed ./ds_config_zero3.json \
    --bf16 \
    --train_file ${root_dir}/${domain}/train.json \
    --validation_file ${root_dir}/${domain}/dev.json \
    --test_file ${root_dir}/${domain}/${split}.json \
    --text_column "text" \
    --summary_column "text" \
    --source_prefix "summarize: " \
    --num_output_sample ${num_output_sample} \
    --max_target_length 300 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 40 \
    --num_output_sample_per_batch ${num_output_sample_per_batch} \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --save_total_limit 5 \
    --load_best_model_at_end True \
    --evaluation_strategy steps \
    --predict_with_generate \
    --output_dir ./general_summary_${model_type}/custom/${domain}/${split}_sample_${num_output_sample}
done
#done
