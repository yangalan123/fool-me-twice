#!/bin/bash
#SBATCH --mail-user=chenghao@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/net/scratch/chenghao/fm2/slurm_output/%A_%a.%N.stdout
#SBATCH --error=/net/scratch/chenghao/fm2/slurm_output/%A_%a.%N.stderr
#SBATCH --chdir=/net/scratch/chenghao/fm2/slurm_output
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --job-name=run_decsum_agent_training
#SBATCH --nodes=1
#SBATCH --mem=100gb
#SBATCH --exclude=g[006,009]
#SBATCH --ntasks=4
#SBATCH --time=11:00:00
#SBATCH --signal=SIGUSR1@120

echo $PATH
source ~/miniconda3/etc/profile.d/conda.sh
cd /net/scratch/chenghao/fm2
conda activate /home/chenghao/env37

#root_dir="/net/scratch/chenghao/yelp/processed_data_all_business_filter_values_category/latest--1reviews-ranking--1/pragsum_filter_chain_filter_values_category_final_review_level_mutual_exclusive/custom"
root_dir="/net/scratch/chenghao/fm2/pragsum_dataset/AgentTraining_domain_wise"
readarray -t domains < <(find "${root_dir}" -maxdepth 1 -type d ! -path "${root_dir}" -exec basename {} \;)
#printf "%s\n" "${domains[@]}"
#slurm_id=2
#echo "test-${domains[${slurm_id}]}"
#SLURM_ARRAY_TASK_ID=1
domain=${domains[${SLURM_ARRAY_TASK_ID}]}
output_root_dir=${root_dir}
origin_epoch=1
epoch=3
#epoch=4
#for domain in "Toronto" "Phoenix"
#for domain in "Las_Vegas" "Phoenix" "Toronto"
#for domain in "Montreal" "Champaign" "Pittsburgh" "Las_Vegas" "Toronto"
#export WANDB_MODE=disabled
#reward_model_dir=${output_root_dir}/${domain}/dbt_agent_training_output_epoch${origin_epoch}${suffix}
agent_model_dir="/net/scratch/chenghao/fm2/pragsum_dataset/AgentTraining_w_gold_evidence/dbt_agent_training_output_epoch10"
for suffix in "_wo_summary"
do
    if [[ ${suffix} == "_wo_summary" ]]
    then
        text_column="text"
    else
        text_column="summary_longt5"
    fi
#      model_name="microsoft/deberta-v3-large"
    #model_name="google/flan-t5-xl"
              #--tracker_project_name PragSumRL_slurm_${model_type}_normal_shared_layers_${num_shared_layers} \
    #model_name="google/flan-t5-base"
    #model_name="gpt2"
#      for model_name in "gpt2-xl" "google/flan-t5-base" "gpt2" "google/flan-t5-xl"
    for model_name in "google/flan-t5-base"
    do
#        model_name="gpt2-xl"
      if [[ ${model_name} == "gpt2-xl" ]]
      then
        num_shared_layers=47
      elif [[ ${model_name} == "google/flan-t5-base" || ${model_name} == "gpt2" ]]
      then
        #num_shared_layers=11
        num_shared_layers=7
      else
        num_shared_layers=23
      fi
      num_samples_per_instance=16
      batch_size=16
      mini_batch_size=4
      model_type=${model_name##*/}
      #output_dir=${output_root_dir}/${domain}/validation_set_rl_agent_${model_type}_threshold_001_reward_epoch${epoch}${suffix}_num_shared_layers_${num_shared_layers}
      #output_dir=${output_root_dir}/${domain}/validation_set_rl_agent_${model_type}_random_sample_${num_samples_per_instance}_epoch${epoch}${suffix}_num_shared_layers_${num_shared_layers}_bsz_${batch_size}_minibsz_${mini_batch_size}
      #output_dir=${output_root_dir}/${domain}/reward_reform_rl_agent_${model_type}_random_sample_${num_samples_per_instance}_epoch${epoch}${suffix}_shared_layers_${num_shared_layers}_bsz_${batch_size}_mbsz_${mini_batch_size}
      #output_dir=${output_root_dir}/${domain}/rl_agent_${model_type}_threshold_reward_epoch${epoch}${suffix}_lora
      output_dir=${root_dir}/${domain}/agent_difference_dataset
      # get the basename of output_dir into exp_name
      exp_name=$(basename "${output_dir}")
                #--deepspeed ./ds_config_zero2.json \
                #--mini_batch_size ${batch_size} \
                #--use_lora \
      if [[ ! -f ${output_dir}/test_difference.json ]]; then
          if [[ -d ${agent_model_dir} ]]; then
              python create_dataset_for_best_k.py \
                --do_train \
                --do_eval \
                --do_predict \
                --generator_model_name ${model_name} \
                --agent_model ${agent_model_dir} \
                --gradient_accumulation_steps 1 \
                --pad_to_max_length False \
                --bf16 \
                --text_column text \
                --train_file ${root_dir}/${domain}/train.json \
                --validation_file ${root_dir}/${domain}/dev.json \
                --test_file ${root_dir}/${domain}/test.json \
                --save_steps 2000 \
                --eval_steps 2000 \
                --max_seq_length 2048 \
                --per_device_train_batch_size 4 \
                --learning_rate 2e-5 \
                --num_train_epochs ${epoch} \
                --save_total_limit 5 \
                --load_best_model_at_end True \
                --evaluation_strategy steps \
                --output_dir ${output_dir}
          fi
      else
          echo "${output_dir} already exists, skipping.."
      fi
    done
done
