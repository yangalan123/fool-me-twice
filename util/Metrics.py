import os
import pickle
import json
import numpy as np
from tqdm import trange
import torch


def validate_length(reward_outputs, sent_outputs, sent_original_outputs):
    return len(reward_outputs) == len(sent_outputs) == len(sent_original_outputs)


def initialize_scores():
    return {
        "normal_scores": {},
        "summary_mean_scores": {},
        "summary_max_scores": {},
        "summary_min_scores": {},
        "fitness_mean_scores": {},
        "fitness_max_scores": {},
        "fitness_min_scores": {},
        "ground_truth_max_scores": {},
        "ground_truth_min_scores": {},
        "labels": {}
    }


def process_summary_data(i, summary_data, sent_outputs, sent_original_outputs, reward_outputs, summary_pointers, func_get_score_from_output,
                         script_args):
    gt_label = summary_data[i]['label'].lower()
    category = summary_data[i]['category']

    summary_data[i]['original_score'] = func_get_score_from_output(sent_original_outputs[i], gt_label)

    output = sent_outputs[summary_pointers[i][0]: summary_pointers[i][1]]
    reward_output = reward_outputs[summary_pointers[i][0]: summary_pointers[i][1]]

    summary_data[i]["performance"] = [func_get_score_from_output(x, gt_label) for x in output]
    summary_data[i]["output_distribution"] = sent_original_outputs[i]

    summary_data[i]["reward"] = [x[0]['score'] if len(x) == 1 else max([y['score'] for y in x]) for x in reward_output]
    if script_args.negate_reward:
        summary_data[i]['reward'] = [-x for x in summary_data[i]['reward']]

    summary_data[i]['performance'] = np.array(summary_data[i]['performance'])
    summary_data[i]['fitness'] = np.abs(summary_data[i]['performance'] - summary_data[i]['original_score'])

    return category, summary_data[i]


def update_scores(scores, category, data_i):
    for key in scores:
        if category not in scores[key]:
            scores[key][category] = []

    scores["normal_scores"][category].append(data_i['original_score'])
    scores["summary_mean_scores"][category].append(np.mean(data_i["performance"]))
    scores["summary_max_scores"][category].append(data_i["performance"][np.argmax(data_i["reward"])])
    scores["summary_min_scores"][category].append(data_i["performance"][np.argmin(data_i["reward"])])
    scores["fitness_mean_scores"][category].append(np.mean(data_i["fitness"]))
    scores["fitness_max_scores"][category].append(data_i["fitness"][np.argmax(data_i["reward"])])
    scores["fitness_min_scores"][category].append(data_i["fitness"][np.argmin(data_i["reward"])])
    scores["labels"][category].append(data_i["label"])
    scores["ground_truth_max_scores"][category].append(np.max(data_i['performance']))
    scores["ground_truth_min_scores"][category].append(np.min(data_i['performance']))

    return scores


def save_scores(summary_data, output_eval_dir, split, script_args, scores, _raw_dataset, file_path, processed_summary_dump_path):
    for category in scores["normal_scores"]:
        all_mses = {}

        for output_filename, output_scores_array in zip(
                [
                    f"eval_{split}_normal.pkl", f"eval_{split}_summary_mean.pkl", f"eval_{split}_summary_max.pkl",
                    f"eval_{split}_summary_min.pkl", f"eval_{split}_fitness_mean.pkl", f"eval_{split}_fitness_max.pkl",
                    f"eval_{split}_fitness_min.pkl", f"eval_{split}_ground_truth_max.pkl",
                    f"eval_{split}_ground_truth_min.pkl"
                ],
                [
                    scores["normal_scores"][category], scores["summary_mean_scores"][category],
                    scores["summary_max_scores"][category],
                    scores["summary_min_scores"][category], scores["fitness_mean_scores"][category],
                    scores["fitness_max_scores"][category], scores["fitness_min_scores"][category],
                    scores["ground_truth_max_scores"][category],
                    scores["ground_truth_min_scores"][category]
                ]
        ):
            output_filename = output_filename.replace("eval", f"eval_{script_args.criterion}")

            with open(os.path.join(output_eval_dir, output_filename), "wb") as f:
                pickle.dump([output_scores_array, scores["labels"]], f)

            json_filename = output_filename.replace(".pkl", ".json")
            with open(os.path.join(output_eval_dir, json_filename), "w") as f:
                res = {
                    "num_samples": len(_raw_dataset),
                    "eval_source_filename": file_path,
                    'avg': np.mean(output_scores_array)
                }
                if "normal" not in json_filename:
                    res['num_summaries'] = len(summary_data[0]["summary"])
                    res["eval_summary_dir"] = processed_summary_dump_path

                json.dump(res, f, indent=4)

            all_mses[output_filename.replace(".pkl", "").replace(f"eval_{script_args.criterion}_{split}_", "")] = res[
                'avg']

        all_mses['num_summaries'] = len(summary_data[0]["summary"])
        pickle.dump(all_mses, open(
            os.path.join(output_eval_dir, f"all_{script_args.criterion}s_{split}_{category}{script_args.exp_name}.pkl"),
            "wb"))

        print(all_mses)

    torch.save(summary_data,
               os.path.join(output_eval_dir, f"summary_data_{split}{script_args.exp_name}_{script_args.criterion}.pt"))


def compute_and_save_metrics(reward_outputs, sent_outputs, sent_original_outputs, script_args, summary_data,
                             func_get_score_from_output, summary_pointers, split, _raw_dataset, file_path,
                             output_eval_dir, processed_summary_dump_path):
    if not validate_length(reward_outputs, sent_outputs, sent_original_outputs):
        raise ValueError("Mismatched lengths for reward_outputs, sent_outputs, and sent_original_outputs.")

    scores = initialize_scores()

    for i in trange(len(summary_data), desc="evaluating"):
        category, data_i = process_summary_data(i, summary_data, sent_outputs, sent_original_outputs, reward_outputs, summary_pointers,
                                                func_get_score_from_output, script_args)
        scores = update_scores(scores, category, data_i)

    save_scores(summary_data, output_eval_dir, split, script_args, scores, _raw_dataset, file_path, processed_summary_dump_path)


# Placeholder variables
script_args = None
summary_data = None
func_get_score_from_output = None
sent_kwargs = None
num_of_summaries_per_example = None
split = None
_raw_dataset = None
file_path = None
output_eval_dir = None
processed_summary_dump_path = None
summary_pointers = None

# Call the refactor code function (this is just a dummy call for now)
# refactor_code(...)

