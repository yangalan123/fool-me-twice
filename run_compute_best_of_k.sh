#!/bin/bash
#SBATCH --mail-user=chenghao@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/net/scratch/chenghao/fm2/slurm_output/%A_%a.%N.stdout
#SBATCH --error=/net/scratch/chenghao/fm2/slurm_output/%A_%a.%N.stderr
#SBATCH --chdir=/net/scratch/chenghao/fm2/slurm_output
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=run_decsum_best_of_k
#SBATCH --nodes=1
#SBATCH --mem=100gb
#SBATCH --exclude=g[006,009]
#SBATCH --ntasks=4
#SBATCH --time=11:00:00
#SBATCH --signal=SIGUSR1@120
##SBATCH --gres=gpu:a40:1

echo $PATH
source ~/miniconda3/etc/profile.d/conda.sh
cd /net/scratch/chenghao/fm2
conda activate /home/chenghao/env37

target_summary_root_dir="/net/scratch/chenghao/fm2/general_summary_longt5/GeneralContext"
#root_dir="/net/scratch/chenghao/yelp/processed_data_all_business_filter_values_category/latest--1reviews-ranking--1/pragsum_filter_chain_filter_values_category_final_review_level_mutual_exclusive/custom"
root_dir="/net/scratch/chenghao/fm2/pragsum_dataset/AgentTraining_domain_wise"
readarray -t domains < <(find "${root_dir}" -maxdepth 1 -type d ! -path "${root_dir}" -exec basename {} \;)
#SLURM_ARRAY_TASK_ID=1
domain=${domains[${SLURM_ARRAY_TASK_ID}]}
target_summary_dir=${target_summary_root_dir}
output_root_dir=${root_dir}
origin_epoch=10
suffix="_wo_summary"
#reward_model_dir="${output_root_dir}/${domain}/dbt_agent_training_output_epoch${origin_epoch}${suffix}"
#reward_model_dir="${output_root_dir}/{}/dbt_agent_training_output_epoch${origin_epoch}${suffix}"
#reward_model_dir="${output_root_dir}/{}/agent_difference_dataset/dbt_agent_training_output_epoch${origin_epoch}${suffix}"
model_root_dir="/net/scratch/chenghao/fm2/pragsum_dataset/AgentTraining_w_gold_evidence"
#as we will have a separate reward model, we need agent model
#agent_model_dir="${model_root_dir}/${domain}/dbt_agent_training_output_epoch${origin_epoch}${suffix}"
agent_model_dir="${model_root_dir}/dbt_agent_training_output_epoch${origin_epoch}"
rm_epoch=3
#reward_model_dir="${root_dir}/{}/agent_difference_dataset/validation_set_dbt_agent_training_output_epoch${rm_epoch}${suffix}"
reward_model_dir="${root_dir}/${domain}/agent_difference_dataset/validation_set_dbt_agent_training_output_epoch${rm_epoch}${suffix}"

echo ${reward_model_dir}
# as we use validation file to do training, we have to test on the test set
    #--validation_file ${root_dir}/${domain}/dev.json \
#    --negate_reward True \
    #--reset_cache True \
# rm is set to fit negative log-like, so we do not need to negate reward
python compute_best_of_k_performance.py \
    --target_summary_dir ${target_summary_dir} \
    --agent_model ${agent_model_dir} \
    --reward_model ${reward_model_dir} \
    --train_file ${root_dir}/${domain}/train.json \
    --validation_file ${root_dir}/${domain}/test.json \
    --output_dir ${target_summary_dir}/${domain} \
    --test_file ${root_dir}/${domain}/test.json


