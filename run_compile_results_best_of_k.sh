target_summary_root_dir="/net/scratch/chenghao/fm2/general_summary_longt5/GeneralContext"
python compile_results_cross_agent_best_of_k.py \
    --target_summary_dir ${target_summary_root_dir} \
    --subdir train_sample_16 \
    --all_mses_suffix "_rm_trained_for_difference"
#python compile_results_cross_agent_best_of_k.py \
    #--subdir test_sample_16 \
    #--exp_name "best-of-16-validation-agent-as-rm" \
    #--all_mses_suffix "agent_as_rm"
