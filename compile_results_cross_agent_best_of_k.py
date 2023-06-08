import json
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import copy
import glob, os
import argparse
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--domain_root_dir", default="/net/scratch/chenghao/yelp/processed_data_all_business_filter_values_category/latest--1reviews-ranking--1/pragsum_filter_chain_filter_values_category_final_review_level_mutual_exclusive", type=str)
    # target_summary_root_dir = "/net/scratch/chenghao/yelp/50reviews/general_summary_flan-t5-base_w_prefix/custom"
    parser.add_argument("--target_summary_root_dir", default="/net/scratch/chenghao/yelp/50reviews/general_summary_flan-t5-base_w_prefix/custom", type=str)
    parser.add_argument("--subdir", default='dev_sample_16', type=str)
    parser.add_argument("--exp_name", default='best-of-16-validation-trained-rm-test-test', type=str)
    parser.add_argument("--all_mses_suffix", default='', type=str)
    args = parser.parse_args()
    domain_dirs = glob.glob(f"/net/scratch/chenghao/fm2/pragsum_dataset/AgentTraining_domain_wise/*")
    domain_values = [os.path.basename(domain_dir) for domain_dir in domain_dirs]
    domain_values.sort()
    for key, explanation in zip(['fitness_min', "summary_min", "delta-best-avg", "delta-mse-best-mse-original"], ["best_summary v.s. model score", "best_summary v.s. human score",
                                                                          "Delta Figure (best_summary - average summary)",
                                                                          "Delta Figure (MSE(best-human), MSE(original-human))"]):
        res = []
        # split = args.subdir.split("_")[0]
        split = "test"
        all_exists = True
        print("Now Computing: ", key)
        for domain_value1 in domain_values:
            res.append([domain_value1, ])
            for domain_value2 in domain_values:
                # [Attention]: Here for FM2, as we change the logic for loading summary in compute_best_of_k_performance.py as well as the storage logic, so we need to change the order of subdir and domain_value2
                eval_target = os.path.join(args.target_summary_root_dir, args.subdir, domain_value2, f"all_mses_{split}_{domain_value1}{args.all_mses_suffix}.pkl")
                if os.path.exists(eval_target):
                    data = pickle.load(open(eval_target, "rb"))
                    # performance = 100 * data['avg']
                    if key in data:
                        performance = 1 * data[key]
                    else:
                        if key == "delta-best-avg":
                            performance = 1 * (data['summary_min'] - data['summary_mean'])
                        elif key == "delta-mse-best-mse-original":
                            performance = 1 * (data['summary_min'] - data['normal'])
                    # performance = 100 * data['summary_min']
                    performance = f"{performance:.2f}"
                    res[-1].append(performance)
                else:
                    # print(f"{eval_target} does not exist")
                    res[-1].append("Running")
                    all_exists = False
                    # exit()

        print("\t&\t".join(["Model Trained on", ] + domain_values))
        for line in res:
            print("\t&\t".join(line) + "\\\\")

        if all_exists:
            res_for_heatmap = [x[1:] for x in res]
            # draw heatmap
            plt.figure(figsize=(10, 10))
            df = pd.DataFrame(res_for_heatmap, columns=domain_values)
            df = df.set_index([pd.Index(domain_values)])
            df = df.astype(float)
            df = df.round(2)
            # sns.set(font_scale=1.5)
            heatmap_cmaps = ["YlGnBu", "YlOrRd", "Blues", "Reds", "Greens", "Purples", "Greys", "Oranges", "PuBu", "PuBuGn", "PuRd", "OrRd", "PuOr", "RdPu", "BuPu", "GnBu", "PuBu", "YlGn", "binary", "gist_yarg", "gist_gray", "gray", "bone", "pink", "spring", "summer", "autumn", "winter", "cool", "Wistia", "hot", "afmhot", "gist_heat", "copper"]
            # ax = sns.heatmap(df, annot=True, fmt='.2f', cmap=random.sample(heatmap_cmaps, 1)[0])
            # ax = sns.heatmap(df, annot=True, fmt='.2f', cmap=heatmap_cmaps[2])
            # fix color scale using min_val and max_val
            # ax = sns.heatmap(df, annot=True, fmt='.2f', cmap=heatmap_cmaps[2], vmin=min_val, vmax=max_val)
            ax = sns.heatmap(df, annot=True, fmt='.2f', cmap=heatmap_cmaps[2] if "delta" not in key else heatmap_cmaps[3])
            ax.xaxis.tick_top()
            # plt.title(f"100 * MSE of DecSum on {exp_name} setting (Percentile Regression)")
            plt.title(f"binary-cross-entropy on Best-of-K setting {explanation}")
            plt.xlabel("Model Trained on")
            plt.ylabel("Model Evaluated on")
            exp_name = args.exp_name + "_" + args.subdir
            visualization_dir = os.path.join("visualization", exp_name + "_{}".format(key))
            os.makedirs(visualization_dir, exist_ok=True)
            plt.savefig(os.path.join(visualization_dir, "bxent_heatmap.png"))
            plt.show()
            # clear figure
            plt.clf()


        # for domain_dir in domain_dirs:
        #     domain_name = os.path.basename(domain_dir)
        #     split = args.subdir.split("_")[0]
        #     source_model_res = os.path.join(domain_dir, args.subdir, f"all_mses_{split}_do")
