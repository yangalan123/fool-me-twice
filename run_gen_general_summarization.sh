#!/bin/bash
#SBATCH --mail-user=chenghao@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/net/scratch/chenghao/fm2/slurm_output/%A_%a.%N.stdout
#SBATCH --error=/net/scratch/chenghao/fm2/slurm_output/%A_%a.%N.stderr
#SBATCH --chdir=/net/scratch/chenghao/fm2/slurm_output
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:4
#SBATCH --job-name=run_gen_general_summarization
#SBATCH --nodes=1
#SBATCH --mem=100gb
#SBATCH --exclude=g[006,009]
#SBATCH --ntasks=4
#SBATCH --time=11:00:00
#SBATCH --signal=SIGUSR1@120
##SBATCH --gres=gpu:a40:1

#for domain in "Las_Vegas" "Phoenix" "Toronto"
#root_dir="/data/yelp/50reviews/pragsum_dataset/city"
echo $PATH
source ~/miniconda3/etc/profile.d/conda.sh
cd /net/scratch/chenghao/fm2
conda activate /home/chenghao/env37
root_dir="/net/scratch/chenghao/fm2/pragsum_dataset"
# for normal
#num_output_sample=1
#num_output_sample_per_batch=1
# for best-of-k
num_output_sample=40
num_output_sample_per_batch=8
#for domain in "Toronto" "Phoenix"
#for domain in "Las_Vegas" "Phoenix" "Toronto"
#for domain in "Montreal" "Champaign" "Pittsburgh" "Scarborough"
#for domain in "Toronto"
#do
#for split in 'train' 'dev' 'test'
for split in 'train'
#for split in 'test'
do
  deepspeed run_summarization.py \
    --model_name_or_path google/long-t5-tglobal-xl \
    --do_predict \
    --deepspeed ./ds_config_zero3.json \
    --bf16 \
    --train_file ${root_dir}/dummy.json \
    --validation_file ${root_dir}/dummy.json \
    --test_file ${root_dir}/GeneralContext.json \
    --text_column "text" \
    --seed 2023 \
    --summary_column "text" \
    --source_prefix "summarize: " \
    --num_output_sample ${num_output_sample} \
    --max_target_length 300 \
    --per_device_train_batch_size 1 \
    --num_output_sample_per_batch ${num_output_sample_per_batch} \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --save_total_limit 5 \
    --load_best_model_at_end True \
    --evaluation_strategy steps \
    --predict_with_generate \
    --output_dir ./general_summary_longt5/GeneralContext/${split}_sample_${num_output_sample}
done
#done
