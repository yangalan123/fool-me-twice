#for domain in "Las_Vegas" "Phoenix" "Toronto"
#root_dir="/data/yelp/50reviews/pragsum_dataset/city"
#root_dir="/data/yelp/50reviews/pragsum_dataset_filter_chain_restaurant/city"
#root_dir="/data/yelp/processed_data/latest-10reviews-ranking-50/pragsum_dataset_filter_chain_restaurant_rank/city"
#root_dir="/data/yelp/processed_data/latest-10reviews-ranking-10/pragsum_dataset_filter_chain_restaurant_rank/city"
#root_dir="/data/yelp/processed_data/latest-10reviews-ranking--1/pragsum_filter_chain_filter_values_country_review_level/city"
root_dir="/data/yelp/processed_data/latest--1reviews-ranking--1/pragsum_filter_chain_filter_values_country_review_level_mutual_exclusive/city"
#root_dir="/data/yelp/processed_data/latest-10reviews-ranking--1/pragsum_dataset_filter_chain_restaurant_rank/city"
output_root_dir=${root_dir}
epoch=1
#for domain in "Toronto" "Phoenix"
#for domain in "Las_Vegas" "Phoenix" "Toronto"
#for domain in "Montreal" "Champaign" "Pittsburgh" "Las_Vegas" "Toronto"
for domain_dir in $root_dir/*
do
    #for suffix in "_wo_summary" "_w_summary"
    domain=$(basename -- "${domain_dir}")
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
              #--label_column rank \
        output_dir=${output_root_dir}/${domain}/dbt_agent_training_output_epoch${epoch}${suffix}
        if [[ ! -d ${output_dir} ]]; then
            deepspeed agent_training.py \
              --model_name_or_path microsoft/deberta-v3-large \
              --do_train \
              --do_eval \
              --do_predict \
              --deepspeed ./ds_config_zero2.json \
              --bf16 \
              --text_column text \
              --train_file ${root_dir}/${domain}/train.json \
              --validation_file ${root_dir}/${domain}/dev.json \
              --test_file ${root_dir}/${domain}/test.json \
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
        fi
    done
done
