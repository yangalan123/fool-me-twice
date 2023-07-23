# please first use run_gen_general_summarization_slurm.sh to get the generated summaries
import copy
import hashlib
import pickle
from sklearn.metrics import mean_squared_error
import numpy as np
import torch
import glob
import traceback
from datasets import load_dataset
from tqdm import tqdm
import datasets, transformers
from transformers import AutoTokenizer, HfArgumentParser, pipeline, TrainingArguments
import json
from tqdm import trange
from transformers.pipelines.pt_utils import KeyDataset, KeyPairDataset
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from trl import AutoModelForSeq2SeqLMWithValueHead, AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
from subprocess import call
import signal
from datasets import Dataset
import threading
import time

from util.DecSumDataset import get_score_from_output, process_func, loadDecSumdataset

logger = logging.getLogger(__name__)


def is_on_slurm():
    return os.environ.get("SLURM_JOB_ID") is not None


def schedule_death(seconds, verbose=False):
    logger.info(f"scheduling death after {seconds}s")

    def f():
        death = time.time() + seconds
        while time.time() < death:
            if verbose:
                logger.info(f"Beep...")
            sleep_interval = max(0, min(600, death - time.time()))
            time.sleep(sleep_interval)

        logger.info(f"time to die...")
        logging.shutdown()
        os.kill(os.getpid(), signal.SIGUSR1)

    threading.Thread(target=f, daemon=True).start()


def slurm_sigusr1_handler_fn(signum, frame) -> None:
    logger.info(f"received signal {signum}")
    job_id = os.environ["SLURM_JOB_ID"]
    cmd = ["scontrol", "requeue", job_id]
    logger.info(f"requeing job {job_id}...")
    try:
        result = call(cmd)
    except FileNotFoundError:
        joint_cmd = [str(x) for x in cmd]
        result = call(" ".join(joint_cmd), shell=True)
    if result == 0:
        logger.info(f"requeued exp {job_id}")
    else:
        logger.info("requeue failed")


def setup_slurm():
    if not is_on_slurm():
        logger.info("not running in slurm, this job will run until it finishes.")
        return
    logger.info("running in slurm, ready to requeue on SIGUSR1.")
    signal.signal(signal.SIGUSR1, slurm_sigusr1_handler_fn)
    # slurm not sending the signal, so sending it myself
    time_to_live = 14300  # just a bit less than 4 hrs
    schedule_death(time_to_live)


def generate_sha256_hash(str_array):
    # Concatenate the strings
    combined_str = "\t\n".join(str_array)

    # Create a new SHA256 hash object
    sha_signature = hashlib.sha256(combined_str.encode()).hexdigest()

    return sha_signature


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        # if self.task_name is not None:
        #     self.task_name = self.task_name.lower()
        #     # if self.task_name not in task_to_keys.keys():
        #     #     raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        # elif self.dataset_name is not None:
        #     pass
        if self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv",
                                       "json"], f"`train_file`({train_extension}) should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                    validation_extension == train_extension
            ), f"`validation_file` should have the same extension (csv or json, now {validation_extension}!={train_extension}) as `train_file`."


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    target_summary_dir: Optional[str] = field(
        metadata={"help": "The root directory where the target summary is stored (before train_sample_xx)."},
    )
    target_summary_suffix: Optional[str] = field(
        default="_sample_16",
        metadata={"help": "The suffix of the target summary directory (e.g., _sample_16)."},
    )
    exp_name: Optional[str] = field(
        default="_rm_trained_for_difference",
        metadata={"help": "The name of the experiment, used to name the output directory."},
    )
    reward_model: Optional[str] = field(default=None, metadata={"help": "the reward model name"})
    negate_reward: Optional[bool] = field(default=False, metadata={"help": "negate the reward"})
    agent_model: Optional[str] = field(default="lvwerra/distilbert-imdb", metadata={"help": "the agent model name"})
    reset_cache: Optional[bool] = field(default=False, metadata={"help": "reset the cache"})

    def __post_init__(self):
        if self.reward_model is None:
            assert self.agent_model is not None, "agent_model must be specified if reward_model is not specified"
            self.reward_model = self.agent_model
    # def __post_init__(self):
    #     assert self.batch_size >= self.mini_batch_size, "batch_size must be >= mini_batch_size"
    #     if self.use_lora:
    #         if self.num_shared_layers != 0:
    #             print("WARNING: num_shared_layers is ignored when using lora")


if __name__ == '__main__':
    tqdm.pandas()
    parser = HfArgumentParser((ScriptArguments, DataTrainingArguments, TrainingArguments))
    script_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {data_args}")

    # last_checkpoint = None
    # if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
    #     if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
    #         raise ValueError(
    #             f"Output directory ({training_args.output_dir}) already exists and is not empty. "
    #             "Use --overwrite_output_dir to overcome."
    #         )
    #     elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    #         logger.info(
    #             f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
    #             "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
    #         )

    setup_slurm()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain_index = None
    # if "{}" in script_args.reward_model:
    #     reward_model_dirs = glob.glob(script_args.reward_model.format("*"))
    #     # get domain index by checking where is "{}" in the reward_model
    #     domain_index = script_args.reward_model.split("/").index("{}")
    # else:
    #     reward_model_dirs = [script_args.reward_model, ]
    #     domain_index = -2
    # assert len(reward_model_dirs) >= 1, f"We need at least one reward model found for {script_args.reward_model}"
    # loss = torch.nn.CrossEntropyLoss()

    assert "{}" not in script_args.agent_model, "We don't support multiple agent models"
    agent_model_dir = script_args.agent_model

    # for reward_model_dir in reward_model_dirs:
    reward_model_dir = script_args.reward_model
    domain_index = -3
    rm_domain_info = reward_model_dir.split("/")[domain_index]
    logger.info(f"Using reward model {reward_model_dir}")
    reward_pipe = pipeline(task="text-classification", model=reward_model_dir, device=0)
    sentiment_pipe = pipeline(task="text-classification", model=agent_model_dir, device=0)

    sent_kwargs = {"return_all_scores": True, "function_to_apply": "softmax", "batch_size": 16}
    rm_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}
    # model = AutoModelForSequenceClassification.from_pretrained(script_args.reward_model).to(device)
    # tokenizer = AutoTokenizer.from_pretrained(script_args.reward_model)
    data_files = {"train": data_args.train_file, "dev": data_args.validation_file, "test": data_args.test_file}
    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
    )
    # validation_dataset = raw_datasets["validation"]
    # train_dataset = raw_datasets["train"]
    # trained on validation set, now evaluate on test set
    for split, file_path in zip(['test'], [data_args.test_file]):
        # for split, file_path in zip(['train', 'dev'], [data_args.train_file, data_args.validation_file]):
        ds_domain_info = file_path.split("/")[-2]
        output_summary_dir = os.path.join(script_args.target_summary_dir,
                                          "train" + script_args.target_summary_suffix + "_domain_decomp",
                                          ds_domain_info)
        if not os.path.exists(output_summary_dir):
            os.makedirs(output_summary_dir)
        _raw_dataset = raw_datasets[split]
        # fixed using train_sample_xxx in post_processing_summary.py
        processed_summary_dump_path = os.path.join(os.path.dirname(data_args.test_file),
                                                   f"{split}_with_summary_longt5.ds")
        processed_summary_dump = datasets.Dataset.load_from_disk(processed_summary_dump_path)
        # output_prediction_files = glob.glob(os.path.join(output_summary_dir, "generated_predictions_*"))
        # create evaluation output dir in the parent dir of the file_path
        output_eval_dir = os.path.join(os.path.dirname(file_path),
                                       f"evaluation_non_pragmatic_{rm_domain_info}{script_args.exp_name}")
        os.makedirs(output_eval_dir, exist_ok=True)
        summary_data = []
        summary_pointer_left = 0
        summary_pointer_right = 0
        summary_pointers = []
        for i in range(len(_raw_dataset)):
            assert _raw_dataset[i]['text'] == processed_summary_dump["text"][i]
            assert _raw_dataset[i]['label'] == processed_summary_dump["label"][i]
            summary_pointer_right += len(processed_summary_dump["summary_longt5"][i])
            item = {"summary": processed_summary_dump["summary_longt5"][i], "original": _raw_dataset[i]["text"],
                    "label": _raw_dataset[i]["label"],
                    "performance": [], "sentence1": _raw_dataset[i]['sentence1'],
                    "sentence2": _raw_dataset[i]['sentence2'],
                    # for Popular Culture -> Popular_Culture
                    "category": _raw_dataset[i]['category'].replace(" ", "_"), }
            summary_data.append(copy.deepcopy(item))
            summary_pointers.append((summary_pointer_left, summary_pointer_right))
            summary_pointer_left = summary_pointer_right
        logger.info("Loaded {} examples from {}".format(len(summary_data), file_path))
        all_summaries = []
        all_claims = []
        all_original_evidences = []
        # num_of_summaries_per_example = 0
        # after we have deduplicate the summaries in (post_processing_summary.py), we cannot use the number of summaries per example to
        # determine the number of summaries per example

        for summary_i in range(len(_raw_dataset)):
            # summary_data[summary_i]["summary"].extend(summaries[summary_i * num_of_summaries_per_example: (summary_i + 1) * num_of_summaries_per_example])
            num_of_summaries_per_example = len(summary_data[summary_i]["summary"])
            all_claims.extend([_raw_dataset[summary_i]["sentence1"], ] * num_of_summaries_per_example)
            all_summaries.extend([_raw_dataset[summary_i]['sentence2'] + x for x in summary_data[summary_i]["summary"]])
            all_original_evidences.extend([_raw_dataset[summary_i]['sentence2'], ] * num_of_summaries_per_example)
        #     # all_summaries.extend(summaries)
        # assert len(all_summaries) == len(_raw_dataset) * len(output_prediction_files) * num_of_summaries_per_example, \
        #     f"bug: \nnumber of summaries ({len(all_summaries)}) must be equal to number of examples ({len(_raw_dataset)}) * number of summaries per example ({num_of_summaries_per_example})"
        dataset = Dataset.from_dict({
            "claim": all_claims,
            "evidence": all_summaries,
        })
        # all_original_text = [x['original'] for x in summary_data]
        original_test_ds = Dataset.from_dict({
            "claim": all_claims,
            "evidence": all_original_evidences
        })
        # cache_path = os.path.join(output_eval_dir, f"best_of_{script_args.target_summary_suffix}_cache_{split}_{rm_domain_info}_outputs.pt")
        cache_path = os.path.join(output_eval_dir,
                                  f"best_of_{script_args.target_summary_suffix}_cache_{split}_{rm_domain_info}_outputs_domain_wise_decomp.pt")
        SHAkey = generate_sha256_hash(
            ["Agent: " + agent_model_dir, " Reward: " + reward_model_dir, " Data: " + file_path])
        try:
            if script_args.reset_cache:
                logger.info("Resetting cache")
                raise Exception("Resetting cache")
            logger.info("Trying to load cached sentiment outputs")
            sent_outputs, sent_original_outputs, reward_outputs = torch.load(cache_path)[SHAkey]
            logger.info("Loaded cached sentiment outputs")
        except:
            logger.info("Failed to load cached sentiment outputs, running sentiment analysis")
            sent_outputs = list(sentiment_pipe(KeyPairDataset(dataset, "claim", "evidence"), **sent_kwargs))
            reward_outputs = list(reward_pipe(KeyPairDataset(dataset, "claim", "evidence"), **rm_kwargs))
            sent_original_outputs = list(
                sentiment_pipe(KeyPairDataset(original_test_ds, "claim", "evidence"), **sent_kwargs))
            if os.path.exists(cache_path):
                cache_data = torch.load(cache_path)
                if not script_args.reset_cache:
                    assert SHAkey not in cache_data, "bug: SHAkey already exists in cache, and you do not want to reset cache"
            else:
                cache_data = {}
            cache_data[SHAkey] = [sent_outputs, sent_original_outputs, reward_outputs]
            try:
                # torch.save([sent_outputs, sent_original_outputs, reward_outputs], cache_path)
                torch.save(cache_data, cache_path)
                logger.info("Saved sentiment outputs")
            except Exception as e:
                print(e)
                traceback.print_exc()
                print(sent_outputs[:10])
                print(sent_original_outputs[:10])
            # for summary_i in range(len(summaries)):
            #     original_index = summary_i // len(_raw_dataset)
            #     summary_data[original_index]["summary"].append(summaries[summary_i])

        normal_scores = dict()
        summary_mean_scores = dict()
        summary_max_scores = dict()
        summary_min_scores = dict()
        fitness_mean_scores = dict()
        fitness_max_scores = dict()
        fitness_min_scores = dict()
        ground_truth_max_scores = dict()
        ground_truth_min_scores = dict()
        labels = dict()
        if "agent_as_rm" not in script_args.exp_name:
            # in this case, reward model must output a scalar
            assert len(reward_outputs[0]) == 1, "bug: reward model must output a scalar, now output: {}".format(
                reward_outputs[0])
        for i in trange(len(summary_data), desc="evaluating"):
            # summary_data[i]['original_score'] = sent_original_outputs[i][0]['score']
            gt_label = summary_data[i]['label'].lower()
            category = summary_data[i]['category']
            # summary_data[i]['original_score'] = [process_func(x['score']) for x in sent_original_outputs[i] if x['label'].lower() == gt_label][0]
            summary_data[i]['original_score'] = get_score_from_output(sent_original_outputs[i], gt_label)
            # output = sentiment_pipe(summary_data[i]["summary"], **sent_kwargs)
            # output = sent_outputs[i * num_of_summaries_per_example: (i + 1) * num_of_summaries_per_example]
            output = sent_outputs[summary_pointers[i][0]: summary_pointers[i][1]]
            # reward_output = reward_outputs[i * num_of_summaries_per_example: (i + 1) * num_of_summaries_per_example]
            reward_output = reward_outputs[summary_pointers[i][0]: summary_pointers[i][1]]
            # print(output)
            # summary_data[i]["performance"] = [x[0]['score'] for x in output]
            summary_data[i]["performance"] = [get_score_from_output(x, gt_label) for x in output]
            # summary_data[i]["reward"] = [x[0]['score'] for x in reward_output]
            summary_data[i]["reward"] = [x[0]['score'] if len(x) == 1 else max([y['score'] for y in x]) for x in
                                         reward_output]
            if script_args.negate_reward:
                summary_data[i]['reward'] = [-x for x in summary_data[i]['reward']]
            summary_data[i]['performance'] = np.array(summary_data[i]['performance'])
            # summary_data[i]['performance'] = np.abs(summary_data[i]['performance'] - summary_data[i]['label'])
            summary_data[i]['fitness'] = np.abs(summary_data[i]['performance'] - summary_data[i]['original_score'])
            # note that as FM2 is a binary classification task, we can no longer use the following code to compute MSE
            # summary_data[i]['performance'] = np.abs(summary_data[i]['performance'] - summary_data[i]['label'])
            # normal_scores.append(abs(summary_data[i]['original_score'] - summary_data[i]['label']))
            # normal_scores.append(summary_data[i]['original_score'])
            if category not in normal_scores:
                normal_scores[category] = []
                summary_mean_scores[category] = []
                summary_max_scores[category] = []
                summary_min_scores[category] = []
                fitness_mean_scores[category] = []
                fitness_max_scores[category] = []
                fitness_min_scores[category] = []
                labels[category] = []
                ground_truth_max_scores[category] = []
                ground_truth_min_scores[category] = []
            normal_scores[category].append(summary_data[i]['original_score'])
            # normal_scores.append(abs(summary_data[i]['original_score'] - summary_data[i]['label']))
            # summary_mean_scores.append(np.mean(summary_data[i]["performance"]))
            summary_mean_scores[category].append(np.mean(summary_data[i]["performance"]))
            # summary_max_scores.append(np.max(summary_data[i]["performance"]))
            # summary_max_scores.append(summary_data[i]["performance"][np.argmax(summary_data[i]["reward"])])
            summary_max_scores[category].append(summary_data[i]["performance"][np.argmax(summary_data[i]["reward"])])
            # summary_min_scores.append(np.min(summary_data[i]["performance"]))
            # summary_min_scores.append(summary_data[i]["performance"][np.argmin(summary_data[i]["reward"])])
            summary_min_scores[category].append(summary_data[i]["performance"][np.argmin(summary_data[i]["reward"])])
            ground_truth_max_scores[category].append(np.max(summary_data[i]['performance']))
            ground_truth_min_scores[category].append(np.min(summary_data[i]['performance']))
            # fitness_mean_scores.append(np.mean(summary_data[i]["fitness"]))
            fitness_mean_scores[category].append(np.mean(summary_data[i]["fitness"]))
            # fitness_max_scores.append(np.max(summary_data[i]["fitness"]))
            # fitness_max_scores.append(summary_data[i]["fitness"][np.argmax(summary_data[i]["reward"])])
            fitness_max_scores[category].append(summary_data[i]["fitness"][np.argmax(summary_data[i]["reward"])])
            # fitness_min_scores.append(np.min(summary_data[i]["fitness"]))
            # fitness_min_scores.append(summary_data[i]["fitness"][np.argmin(summary_data[i]["reward"])])
            fitness_min_scores[category].append(summary_data[i]["fitness"][np.argmin(summary_data[i]["reward"])])
            # labels.append(summary_data[i]["label"])
            labels[category].append(summary_data[i]["label"])
        # normal_scores = np.array(normal_scores)
        # summary_mean_scores = np.array(summary_mean_scores)
        # summary_max_scores = np.array(summary_max_scores)
        # summary_min_scores = np.array(summary_min_scores)
        # labels = np.array(labels)
        for category in normal_scores:
            normal_scores[category] = np.array(normal_scores[category])
            summary_mean_scores[category] = np.array(summary_mean_scores[category])
            summary_max_scores[category] = np.array(summary_max_scores[category])
            summary_min_scores[category] = np.array(summary_min_scores[category])
            fitness_mean_scores[category] = np.array(fitness_mean_scores[category])
            fitness_max_scores[category] = np.array(fitness_max_scores[category])
            fitness_min_scores[category] = np.array(fitness_min_scores[category])
            labels[category] = np.array(labels[category])
            ground_truth_max_scores[category] = np.array(ground_truth_max_scores[category])
            ground_truth_min_scores[category] = np.array(ground_truth_min_scores[category])
            all_mses = {}
            for output_filename, output_scores_array in zip(
                    [f"eval_{split}_normal.pkl", f"eval_{split}_summary_mean.pkl", f"eval_{split}_summary_max.pkl",
                     f"eval_{split}_summary_min.pkl", f"eval_{split}_fitness_mean.pkl", f"eval_{split}_fitness_max.pkl",
                     f"eval_{split}_fitness_min.pkl", f"eval_{split}_ground_truth_max.pkl",
                        f"eval_{split}_ground_truth_min.pkl"],
                    [normal_scores[category], summary_mean_scores[category], summary_max_scores[category], summary_min_scores[category], fitness_mean_scores[category],
                     fitness_max_scores[category], fitness_min_scores[category], ground_truth_max_scores[category], ground_truth_min_scores[category]]):
                with open(os.path.join(output_eval_dir, output_filename), "wb") as f:
                    pickle.dump([output_scores_array, labels], f)
                json_filename = output_filename.replace(".pkl", ".json")
                with open(os.path.join(output_eval_dir, json_filename), "w") as f:
                    res = {
                        "num_samples": len(_raw_dataset),
                        "eval_source_filename": file_path,
                    }
                    # res['mse'] = mean_squared_error(output_scores_array, labels)
                    # all_mses[output_filename.replace(".pkl", "").replace(f"eval_{split}_", "")] = res['mse']
                    res['avg'] = np.mean(output_scores_array)
                    all_mses[output_filename.replace(".pkl", "").replace(f"eval_{split}_", "")] = res['avg']
                    if "normal" not in json_filename:
                        res['num_summaries'] = len(summary_data[0]["summary"])
                        # res["eval_summary_dir"] = output_summary_dir
                        res["eval_summary_dir"] = processed_summary_dump_path
                    json.dump(res, f, indent=4)
            all_mses['num_summaries'] = len(summary_data[0]["summary"])
            # if "agent_as_rm" in script_args.exp_name:
            pickle.dump(all_mses, open(
                os.path.join(output_summary_dir, f"all_mses_{split}_{category}{script_args.exp_name}.pkl"), "wb"))
            print(all_mses)

        torch.save(summary_data, os.path.join(output_summary_dir, f"summary_data_{split}{script_args.exp_name}.pt"))
