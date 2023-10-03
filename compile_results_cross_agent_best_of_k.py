import json
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import copy
import glob, os
import argparse
import random
import torch
import numpy as np
import csv

def error_analysis(domain_values, args, output_dir="error_analysis"):
    for domain_value1 in domain_values:
        error_dir = os.path.join(output_dir, domain_value1)
        os.makedirs(error_dir, exist_ok=True)
        res.append([domain_value1, ])
        # for domain_value2 in domain_values:
            # [Attention]: Here for FM2, as we change the logic for loading summary in compute_best_of_k_performance_pipe.py as well as the storage logic, so we need to change the order of subdir and domain_value2
        eval_target = os.path.join(args.target_summary_root_dir, args.subdir, domain_value1,
                                   f"summary_data_{split}{args.all_mses_suffix}_{args.criterion}.pt")
        buf = []
        all_num_summaries = []
        all_delta_gt_scores = []
        all_delta_original_scores = []
        if os.path.exists(eval_target):
            # data = pickle.load(open(eval_target, "rb"))
            summary_data = torch.load(eval_target)
            for item in summary_data:
                original_score = item['original_score']
                best_score = item['performance'][np.argmin(item['reward'])]
                best_summary = item['summary'][np.argmin(item['reward'])]
                gt_score = np.min(item['performance'])
                gt_summary = item['summary'][np.argmin(item['performance'])]
                category = item['category']
                label = item['label']
                claim = item['sentence1']
                evidence = item['sentence2']
                num_summaries = len(item['summary'])
                prediction_id = np.argmax([x['score'] for x in item['output_distribution']])
                prediction_label = item['output_distribution'][prediction_id]['label']
                if best_score > original_score and abs(gt_score - original_score) > 0.01:
                    buf.append({
                        'original_score': original_score,
                        'rm_score': best_score,
                        'rm_picked_summary': best_summary,
                        'gt_score': gt_score,
                        'gt_summary': gt_summary,
                        'category': category,
                        'label': label,
                        'claim': claim,
                        'evidence': evidence,
                        'num_summaries': num_summaries,
                        'prediction': prediction_label
                    })
                all_num_summaries.append(num_summaries)
                all_delta_gt_scores.append(gt_score - original_score)
                all_delta_original_scores.append(best_score - original_score)
        # output buf to csv files for further analysis
        # df = pd.DataFrame(buf)
        # df.to_csv(os.path.join(error_dir, f"error_analysis_{split}.csv"), index=False)
        file_name = os.path.join(error_dir, f"error_analysis_{split}_{args.criterion}.csv")


        # Open the file in write mode
        def format_value(value):
            if isinstance(value, float):
                return "{:.3f}".format(value)
            return value

        # Open the file in write mode
        with open(file_name, 'w', newline='') as output_file:
            # Create a CSV DictWriter
            writer = csv.DictWriter(output_file, fieldnames=buf[0].keys())

            # Write the header
            writer.writeheader()

            # Write the data rows
            for row in buf:
                formatted_row = {k: format_value(v) for k, v in row.items()}
                writer.writerow(formatted_row)
        pickle.dump(buf, open(os.path.join(error_dir, f"error_analysis_{split}.pkl"), "wb"))
        # plot histogram for all_num_summaries
        for name, array in zip(["num_summaries", "delta_gt_vs_original", "delta_rm_picked_vs_original"], [all_num_summaries, all_delta_gt_scores, all_delta_original_scores]):
            plt.figure()
            plt.hist(array, bins=100)
            plt.title(f"{name} histogram")
            plt.xlabel(name)
            plt.ylabel("count")
            plt.savefig(os.path.join(error_dir, f"{name}_{split}_{args.criterion}.png"))
            # clear plt
            plt.clf()
            plt.close()


            # performance = 100 * data['avg']

            # if key in data:
            #     performance = 100 * data[key]
            # else:
            #     if key == "delta-best-avg":
            #         performance = 100 * (data['summary_min'] - data['summary_mean'])
            #     elif key == "delta-best-original":
            #         performance = 100 * (data['summary_min'] - data['normal'])
            #     elif key == "delta-gt-best-rm-best":
            #         performance = 100 * (data['ground_truth_min'] - data['summary_min'])
            #     elif key == "delta-gt-best-avg":
            #         performance = 100 * (data['ground_truth_min'] - data['summary_mean'])
            #     elif key == "delta-gt-original":
            #         performance = 100 * (data['ground_truth_min'] - data['normal'])
            #     elif key == "delta-mean-original":
            #         performance = 100 * (data['summary_mean'] - data['normal'])
            # performance = 100 * data['summary_min']
        #     performance = f"{performance:.2f}"
        #     res[-1].append(performance)
        # else:
        #     # print(f"{eval_target} does not exist")
        #     res[-1].append("Running")
        #     all_exists = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--domain_root_dir", default="/net/scratch/chenghao/yelp/processed_data_all_business_filter_values_category/latest--1reviews-ranking--1/pragsum_filter_chain_filter_values_category_final_review_level_mutual_exclusive", type=str)
    # target_summary_root_dir = "/net/scratch/chenghao/yelp/50reviews/general_summary_flan-t5-base_w_prefix/custom"
    parser.add_argument("--target_summary_root_dir",
                        default="/net/scratch/chenghao/yelp/50reviews/general_summary_flan-t5-base_w_prefix/custom",
                        type=str)
    parser.add_argument("--subdir", default='dev_sample_16', type=str)
    parser.add_argument("--exp_name", default='best-of-16-validation-trained-rm-test-test', type=str)
    parser.add_argument("--all_mses_suffix", default='', type=str)
    parser.add_argument("--criterion", default='mse', type=str)
    args = parser.parse_args()
    domain_dirs = glob.glob(f"/net/scratch/chenghao/fm2/pragsum_dataset/AgentTraining_domain_wise/*")
    domain_values = [os.path.basename(domain_dir) for domain_dir in domain_dirs]
    domain_values.sort()
    for key, explanation in zip(
            ['fitness_min', "summary_mean", "summary_min", "delta-best-avg", "delta-best-original",
             "ground_truth_min", "delta-gt-best-rm-best", "delta-gt-best-avg", "delta-gt-original",
             "delta-mean-original"], ["best_summary v.s. model score (fitness)",
                                                                                 "average summary (gt loss)",
                                                                                 "best_summary (gt loss)",
                                                                                 "Delta Figure (best_summary - average summary)",
                                                                                 "Delta Figure (Loss(best-human), Loss(original-human))",
                                                                                 "oracle rm (gt loss)",
                                                                                 "Delta Figure (oracle rm - best_summary)",
                                                                                 "Delta Figure (oracle rm - average summary)",
                                      "Delta Figure (oracle rm - original summary)", "Delta Figure (average summary - original summary)"]):
        res = []
        # split = args.subdir.split("_")[0]
        split = "test"
        all_exists = True
        print("Now Computing: ", key)
        for domain_value1 in domain_values:
            res.append([domain_value1, ])
            for domain_value2 in domain_values:
                # [Attention]: Here for FM2, as we change the logic for loading summary in compute_best_of_k_performance_pipe.py as well as the storage logic, so we need to change the order of subdir and domain_value2
                eval_target = os.path.join(args.target_summary_root_dir, args.subdir, domain_value1,
                                           f"all_{args.criterion}s_{split}_{domain_value2}{args.all_mses_suffix}.pkl")
                if os.path.exists(eval_target):
                    data = pickle.load(open(eval_target, "rb"))
                    # performance = 100 * data['avg']
                    if key in data:
                        performance = 100 * data[key]
                    else:
                        if key == "delta-best-avg":
                            performance = 100 * (data['summary_min'] - data['summary_mean'])
                        elif key == "delta-best-original":
                            performance = 100 * (data['summary_min'] - data['normal'])
                        elif key == "delta-gt-best-rm-best":
                            performance = 100 * (data['ground_truth_min'] - data['summary_min'])
                        elif key == "delta-gt-best-avg":
                            performance = 100 * (data['ground_truth_min'] - data['summary_mean'])
                        elif key == "delta-gt-original":
                            performance = 100 * (data['ground_truth_min'] - data['normal'])
                        elif key == "delta-mean-original":
                            performance = 100 * (data['summary_mean'] - data['normal'])
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
            heatmap_cmaps = ["YlGnBu", "YlOrRd", "Blues", "Reds", "Greens", "Purples", "Greys", "Oranges", "PuBu",
                             "PuBuGn", "PuRd", "OrRd", "PuOr", "RdPu", "BuPu", "GnBu", "PuBu", "YlGn", "binary",
                             "gist_yarg", "gist_gray", "gray", "bone", "pink", "spring", "summer", "autumn", "winter",
                             "cool", "Wistia", "hot", "afmhot", "gist_heat", "copper"]
            # ax = sns.heatmap(df, annot=True, fmt='.2f', cmap=random.sample(heatmap_cmaps, 1)[0])
            # ax = sns.heatmap(df, annot=True, fmt='.2f', cmap=heatmap_cmaps[2])
            # fix color scale using min_val and max_val
            # ax = sns.heatmap(df, annot=True, fmt='.2f', cmap=heatmap_cmaps[2], vmin=min_val, vmax=max_val)
            ax = sns.heatmap(df, annot=True, fmt='.2f',
                             cmap=heatmap_cmaps[2] if "delta" not in key else heatmap_cmaps[3])
            ax.xaxis.tick_top()
            # plt.title(f"100 * MSE of DecSum on {exp_name} setting (Percentile Regression)")
            plt.title(f"binary-cross-entropy (*100) on Best-of-K setting {explanation}")
            plt.xlabel("Model Evaluated on")
            plt.ylabel("Model Trained on")
            exp_name = args.exp_name + "_" + args.subdir
            visualization_dir = os.path.join("visualization_after_debug_0719", exp_name + f"_{args.all_mses_suffix}" + "_{}".format(key))
            os.makedirs(visualization_dir, exist_ok=True)
            plt.savefig(os.path.join(visualization_dir, f"bxent_heatmap_{args.criterion}.png"))
            plt.show()
            # clear figure
            plt.clf()
            error_analysis(domain_values, args)

        # for domain_dir in domain_dirs:
        #     domain_name = os.path.basename(domain_dir)
        #     split = args.subdir.split("_")[0]
        #     source_model_res = os.path.join(domain_dir, args.subdir, f"all_mses_{split}_do")
