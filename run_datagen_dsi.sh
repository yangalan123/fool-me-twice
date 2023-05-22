#!/bin/bash
#SBATCH --mail-user=chenghao@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/net/scratch/chenghao/t-zero/evaluation/slurm_output/%j.%N.stdout
#SBATCH --error=/net/scratch/chenghao/t-zero/evaluation/slurm_output/%j.%N.stderr
#SBATCH --chdir=/net/scratch/chenghao/t-zero/evaluation/slurm_output
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:4
#SBATCH --job-name=run_tzero_data_collection
#SBATCH --nodes=1
#SBATCH --mem=100gb
#SBATCH --ntasks=4
#SBATCH --time=4:00:00
#for model in T0pp T0_3B

#do
#python run_eval.py \
    #--dataset_name tau/scrolls \
    #--dataset_config_name quality \
    #--template_name "Answer Given options" \
    #--model_name_or_path bigscience/${model} \
    #--output_dir ./output_scroll_quality/${model} \
    #--parallelize
#done
#for agent in {0..19..2}
#for agent in {0..19..5}
                        #--dataset_config_name2 "dpr-rest-"$((5*(19-agent)))"%-maxlen-"${maxlen1} \
#for agent in {3..19..5}
                #file=./output_chrome_quality/first_${maxlen0}_rest_${maxlen1}/${model}/agent_${agent}/results.json
                        #--output_dir ./output_chrome_quality/first_${maxlen0}_rest_${maxlen1}/${model}/agent_${agent} \
                        #--dataset_config_name2 "dpr-rest-0%-maxlen-"${maxlen1} \
                        #--dataset_config_name2 "dpr-first-0%-maxlen-"${maxlen1} \
                        #--output_dir ./output_chrome_quality/first_${maxlen0}_rest_fulldpr_${maxlen1}/${model}/agent_${agent} \
                #file=./output_chrome_quality/first_${maxlen0}_rest_fulldpr_${maxlen1}/${model}/agent_${agent}/results.json
                        #--dataset_config_name2 "dpr-rest-"$((5*(19-agent)))"%-maxlen-"${maxlen1} \
                        #--dataset_config_name "dpr-first-"$((5*agent))"%-maxlen-"${maxlen0} \
                #root_dir="/data/chenghao/t-zero/evaluation/random_summary_full_rand_200_${maxlen1}"
                #root_dir="/data/chenghao/t-zero/evaluation/random_summary_agent_${agent}_maxlen_${maxlen0}_rest_rand_3_${maxlen1}"
cd /net/scratch/chenghao/fm2
#agent=5
#for maxlen0 in 150 300
#for maxlen0 in 150 300
#for maxlen0 in 300
##for agent in {0..19}
#do
    ##for maxlen0 in 150 300 500
    #for flag in "rest" "full"
    #do
        ##for maxlen1 in 300 150
        #for maxlen1 in 300
        ##for maxlen in 25
        #do
#for model_name in bigscience/T0pp
for model_name in "bigscience/T0pp" "google/flan-t5-xxl"
do
#file=./output_chrome_quality/first_${maxlen0}_rest_${maxlen1}/${model}/agent_${agent}/results.json
#save_key=random_summary_full_${maxlen1}
#save_key=random_summary_agent_${agent}_maxlen_${maxlen0}_${flag}_rand_3_${maxlen1}
save_key=gold_evidence_agent
#root_dir="/net/scratch/chenghao/fm2/evaluation/${save_key}"
root_dir="/net/scratch/chenghao/fm2/pragsum_dataset/AgentTraining_zeroshot_qa_t0"
#save_root=./output_gold_evidence/${save_key}/${model}
model_type=${model_name##*/}
save_root=./output_gold_evidence/${save_key}/${model_type}
#save_root=./output_gold_evidence/${save_key}/${model_type}_int8
file=${save_root}/validation_predictions.p
if test -d "$file"; then
    echo "$file exists. skipping.."
else
    echo "$file not exists. running.."
    #python run_eval_ours_slurm.py \
    python run_t0pp_slurm.py \
        --dataset_name custom \
        --dataset_config_name ${root_dir}\
        --dataset_name2 "None" \
        --dataset_config_name2 "" \
        --template_name "Answer Given options" \
        --model_name_or_path ${model_name} \
        --output_dir ${save_root} \
        --preserve_meta_info
        #--preserve_meta_info \
        #--parallelize
fi
done
