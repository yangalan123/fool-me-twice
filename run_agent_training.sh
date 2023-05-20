root_dir=/data/chenghao/fool-me-twice/pragsum_dataset/AgentTraining
epoch=10
output_dir=${root_dir}/dbt_agent_training_output_epoch${epoch}
deepspeed agent_training_slurm.py \
  --model_name_or_path microsoft/deberta-v3-large \
  --do_train \
  --do_eval \
  --do_predict \
  --deepspeed ./ds_config_zero2.json \
  --bf16 \
  --label_column label \
  --train_file ${root_dir}/train.json \
  --validation_file ${root_dir}/dev.json \
  --test_file ${root_dir}/test.json \
  --save_steps 2000 \
  --eval_steps 2000 \
  --max_seq_length 2048 \
  --per_device_train_batch_size 1 \
  --learning_rate 2e-5 \
  --num_train_epochs ${epoch} \
  --save_total_limit 5 \
  --load_best_model_at_end True \
  --evaluation_strategy steps \
  --output_dir ${output_dir}
