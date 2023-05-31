root_dir=/data/chenghao/fool-me-twice/pragsum_dataset/AgentTraining
epoch=10
output_root_dir=${root_dir}/dbt_agent_training_output_epoch${epoch}
model_dir=${output_root_dir}
output_dir=${output_root_dir}/eval_wo_gold_evidence
  #--do_train \
  #--train_file ${root_dir}_w_summary/train.json \
  #--validation_file ${root_dir}_w_summary/dev.json \
  #--test_file ${root_dir}_w_summary/test.json \
  #--output_dir ${output_dir}/replace_evidence_with_wikisummary
  #--train_file ${root_dir}_no_evidence/train.json \
  #--validation_file ${root_dir}_no_evidence/dev.json \
  #--test_file ${root_dir}_no_evidence/test.json \
  #--output_dir ${output_dir}/replace_evidence_with_none
deepspeed agent_training_slurm.py \
  --model_name_or_path ${model_dir} \
  --do_eval \
  --do_predict \
  --deepspeed ./ds_config_zero3.json \
  --bf16 \
  --label_column label \
  --train_file ${root_dir}_w_longt5_summary/train.json \
  --validation_file ${root_dir}_w_longt5_summary/dev.json \
  --test_file ${root_dir}_w_longt5_summary/test.json \
  --save_steps 2000 \
  --eval_steps 2000 \
  --max_seq_length 2048 \
  --per_device_train_batch_size 1 \
  --learning_rate 2e-5 \
  --num_train_epochs ${epoch} \
  --save_total_limit 5 \
  --load_best_model_at_end True \
  --evaluation_strategy steps \
  --output_dir ${output_dir}/replace_evidence_with_longt5_general_summary
