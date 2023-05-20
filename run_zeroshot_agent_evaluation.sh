#root_dir=/net/scratch/chenghao/fm2/pragsum_dataset/AgentTraining
root_dir=/net/scratch/chenghao/fm2/pragsum_dataset/AgentTraining_zeroshot
#root_dir=/net/scratch/chenghao/fm2/fever/AgentTraining_zeroshot
epoch=10
#output_dir=${root_dir}/rbt_offtheshelf_ynie_agent_training_output_epoch${epoch}
output_dir=${root_dir}/dbt_offtheshelf_moritz_agent_training_output_epoch${epoch}
#output_dir=${root_dir}/debug
  #--do_train \
  #--task_name mnli \
  #--model_name_or_path MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli \
  #--model_name_or_path ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli \
  #--dataset_name pietrolesci/nli_fever \
deepspeed agent_training_slurm_updated_labels.py \
  --model_name_or_path MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli \
  --do_eval \
  --do_predict \
  --train_file ${root_dir}/train.json \
  --validation_file ${root_dir}/dev.json \
  --test_file ${root_dir}/dev.json \
  --deepspeed ./ds_config_zero3.json \
  --bf16 \
  --label_column label \
  --save_steps 2000 \
  --eval_steps 2000 \
  --max_seq_length 2048 \
  --per_device_train_batch_size 1 \
  --learning_rate 2e-5 \
  --num_train_epochs ${epoch} \
  --save_total_limit 5 \
  --load_best_model_at_end True \
  --evaluation_strategy steps \
  --overwrite_output_dir \
  --output_dir ${output_dir}
