#!/usr/bin/env python
# coding=utf-8
# Copyright BigScience, The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Reproduce the main evaluation in `Multitask Prompted Training Enables Zero-Shot Task Generalization` using PyTorch.

This script is heavily adapted from https://github.com/huggingface/transformers/blob/7533d30acd975027e83a548e4c38e06fa335291b/examples/pytorch/multiple-choice/run_swag_no_trainer.py
"""

import argparse
import logging
import os
import random
import json

import datasets
import torch
from datasets import load_dataset, load_metric, Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoTokenizer,
    default_data_collator,
)
from promptsource.templates import DatasetTemplates

from t0.data_collator import DataCollatorForMultipleChoice
from t0.model import ModelBase
from DataCollatorWithMetaInfo import DataCollatorForMultipleChoiceWithMetaInfo
import numpy as np

from subprocess import call
import signal
import threading
import time

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


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce main evaluation in T0.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
        required=True,
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_name2",
        type=str,
        default=None,
        help="The name of the second dataset to use (via the datasets library).",
        required=True,
    )
    parser.add_argument(
        "--dataset_config_name2",
        type=str,
        default=None,
        help="The configuration name of the second dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--template_name",
        type=str,
        default=None,
        help="The template/prompt name",
        required=True,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--rest_max_length",
        type=int,
        default=150,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--target_max_length",
        type=int,
        default=256,
        help="Target max length. Sequences longer than this will be truncated."
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models. The list of T0 variants can be found on `https://huggingface.co/bigscience/T0_3B`",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--parallelize",
        action="store_true",
        help=(
            "If passed, will call `model.parallelize` which splits the model on all GPUs available when applicable (model parallelism). "
            "Note that this feature is still experimental in HF Transformers."
        ),
    )
    parser.add_argument(
        "--preserve_meta_info",
        action="store_true",
        help="preserve meta information for each inferred instance.",
    )
    parser.add_argument(
        "--total_splits",
        type=int,
        default=-1,
        help="Total number of splits (set -1 to disable this feature), useful for checkpointing when inferring over large dataset.",
    )
    parser.add_argument(
        "--current_split",
        type=int,
        default=0,
        help="the current number of splits, useful for checkpointing when inferring over large dataset.",
    )
    args = parser.parse_args()
    if args.total_splits > 0:
        assert args.current_split < args.total_splits, f"impossible to run the current split with (cur) {args.current_split} / (total) {args.total_splits}"

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()


    # Handle the output directory creation
    filter_out_ids = []
    tmp_out_ds = None
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.total_splits < 0:
            metrics_filepath = os.path.join(args.output_dir, "results.json")
            auxoutput_filepath = os.path.join(args.output_dir, "validation_predictions.p")
        else:
            metrics_filepath = os.path.join(args.output_dir, f"results_{args.current_split}.json")
            auxoutput_filepath = os.path.join(args.output_dir, f"validation_predictions_{args.current_split}.p")
        tmp_path = auxoutput_filepath + ".tmp"
        if os.path.exists(tmp_path):
            try:
                tmp_out_ds = datasets.load_from_disk(tmp_path)
                filter_out_ids = set(tmp_out_ds['indices'])
                logger.info(f"Successfully loading from temporarily saved data, load {len(filter_out_ids)} data, these data will be excluded in this run")
            except:
                logger.info(f"Fail to load temporarily saved data at {tmp_path}, starting to generate from beginning")
        setup_slurm()

    accelerator.wait_for_everyone()

    # In distributed evaluation, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        if args.dataset_name == "anli":
            error_message = "For ANLI, `dataset_config_name` should be either `dev_r1`, `dev_r2` or `dev_r3`."
            assert args.dataset_config_name is not None, error_message
            assert args.dataset_config_name in ["dev_r1", "dev_r2", "dev_r3"], error_message
            raw_datasets = load_dataset(args.dataset_name, split=args.dataset_config_name)
        else:
            if args.dataset_name == "chromeNLP/quality":
                raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, split="validation")
                raw_datasets2 = load_dataset(args.dataset_name2, args.dataset_config_name2, split="validation")
            else:
                assert args.dataset_name == "custom" and args.dataset_name2 == "None", f"unrecognized dataset: {args.dataset_name}"
                # then loading local datasets, using config_name as the path
                # raw_datasets = load_dataset("json", data_files={"validation": args.dataset_config_name}, split="validation")
                # raw_datasets2 = load_dataset("json", data_files={"validation": args.dataset_config_name2}, split="validation")
                raw_ds = []
                # raw_ds2 = []
                for split in ['train', 'validation']:
                    _raw_datasets = datasets.load_from_disk(os.path.join(args.dataset_config_name, f"{split}.ds"))
                    raw_ds.append(_raw_datasets)
                raw_datasets = datasets.concatenate_datasets(raw_ds)
                # we need to build index here, otherwise later, when excluding the data, the new index will no longer be the pointer to original index
                # example: [0, 1, 2, 3, 4] - [1, 2] = [0, 3, 4] -> [0, 1, 2] (with mapping 0->0, 1->3, 2->4, but such mapping will not be stored anywhere)
                # you can imagine with the number of iteration goes up, such mapping will be accumulated
                raw_datasets = raw_datasets.add_column("original_index", list(range(len(raw_datasets))))
                if args.total_splits > 0:
                    data_size = len(raw_datasets)
                    num_per_splits = data_size // args.total_splits
                    raw_datasets = raw_datasets.select(
                        range(args.current_split * num_per_splits, (args.current_split + 1) * num_per_splits))
                if tmp_out_ds is not None:
                    assert len(filter_out_ids) > 0
                    raw_datasets = raw_datasets.filter(lambda x: x['original_index'] not in filter_out_ids)
    #TODO(Victor): enable loading pre-processed dataset from https://huggingface.co/datasets/bigscience/P3

    # Trim a number of evaluation examples
    if args.debug:
        if args.dataset_name == "custom" and args.dataset_name2 == "None":
            raw_datasets = raw_datasets.filter(lambda x: x['original_index'] < 16)
        else:
            raw_datasets = raw_datasets.select(range(min(len(raw_datasets), 16)))


    column_names = raw_datasets.column_names
    new_dataset = dict()
    if args.dataset_name2 != "None":
        for data_i in range(len(raw_datasets[column_names[0]])):
            assert raw_datasets["question"][data_i] == raw_datasets2["question"][data_i]
            assert raw_datasets["options"][data_i] == raw_datasets2["options"][data_i]
            for column_name in column_names:
                if column_name not in new_dataset:
                    new_dataset[column_name] = []
                if column_name == "context":
                    new_dataset["context"].append(raw_datasets["context"][data_i] + raw_datasets2["context"][data_i])
                else:
                    new_dataset[column_name].append(raw_datasets[column_name][data_i])
        raw_datasets = Dataset.from_dict(new_dataset)
    #print("dataset0", raw_datasets['context'][0], "\ndataset1:", raw_datasets2['context'][0], "\ndataset2:", raw_datasets_delta['context'][0])
    #exit()


    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "Either `args.config_name` or `args.model_name_or_path` should be provided."
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer, padding_side="left")
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, padding_side="left")
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if tokenizer.pad_token is None:
        for token in [tokenizer.eos_token, tokenizer.bos_token, tokenizer.sep_token]:
            if token is not None:
                tokenizer.pad_token = token
        if tokenizer.pad_token is None:
            raise ValueError("Please define a pad token id.")


    model = ModelBase.from_config(
        config=config,
        model_name_or_path=args.model_name_or_path,
        parallelize=args.parallelize
    )

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False

    # Get the prompt to apply and the possible targets.
    # TODO(Victor): If pulling from pre-processed data, remove this logic.
    #prompts = DatasetTemplates(
        #f"{args.dataset_name}"
        #if args.dataset_config_name is None
        #else f"{args.dataset_name}/{args.dataset_config_name}"
    #)
    #template = prompts[args.template_name]
    import re
    regex = "(\([A-D]\).+?(?=\n))"

    def preprocess_function(examples):
    # def preprocess_function(examples):
        bs = len(examples[column_names[0]])

        input_texts = []
        target_texts = []
        answer_choices_texts = []
        indices = []
        for i in range(bs):
            ex = {
                k: examples[k][i]
                for k in column_names
            }
            #input, target = template.apply(ex)
            question, context, options, target = ex['question'], ex['context'], ex['options'], ex['output']
            assert len(options) <= 4, f"invalid options number: {len(options)}"
            options = [x+y for x, y in zip(["(A) ", "(B) ", "(C) ", "(D) "], options)]
            input = "{}\n\n{}\n\n{}".format(question, "\n".join(options), context)

            #input, target = ex['input'], ex['output'].strip()
            #ex_answer_choices = template.get_answer_choices_list(ex)
            #ex_answer_choices = re.findall(regex, input)
            #ex_answer_choices = [x.split(") ")[-1].strip() for x in ex_answer_choices]
            ex_answer_choices = ex['options']

            assert target in ex_answer_choices, "\ntarget: {}\nanswer_choices: {}".format(target, "\n".join(ex_answer_choices))
            input_texts.append(input)
            target_texts.append(target)
            answer_choices_texts.append(ex_answer_choices)

        tokenized_inputs = tokenizer(
            input_texts,
            padding=padding,
            max_length=args.max_length,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = [
            tokenizer(
                ans_choi,
                # padding is on the right here.
                padding=False,
                max_length=args.max_length,
                truncation=True,
            )
            for ans_choi in answer_choices_texts
        ]

        features = {
            k: [
                [elem for _ in range(len(tokenized_targets[idx]["input_ids"]))]
                for idx, elem in enumerate(v)
            ]
            for k, v in tokenized_inputs.items()
        }

        features["labels"] = [
            tokenized_targets[idx]["input_ids"]
            for idx in range(bs)
        ]
        features["labels_attention_mask"] = [
            tokenized_targets[idx]["attention_mask"]
            for idx in range(bs)
        ]
        features["targets"] = [
            answer_choices_texts[idx].index(t)
            for idx, t in enumerate(target_texts)
        ]
        features['indices'] = examples["original_index"].copy()

        return features

    # meta_info = {"article_id", "question"}

    with accelerator.main_process_first():
        eval_dataset = raw_datasets.map(
            preprocess_function, batched=True, remove_columns=column_names
        # preprocess_function, batched = True, remove_columns = list(set(column_names) - meta_info)
        )
        # eval_dataset = raw_datasets.map(
        #     preprocess_function, batched=True, remove_columns=column_names
        #     # preprocess_function, batched = True, remove_columns = list(set(column_names) - meta_info)
        # )

    # Log a few random samples from the eval set:
    for index in random.sample(range(len(eval_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {eval_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        if not args.preserve_meta_info:
            data_collator = DataCollatorForMultipleChoice(
                tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
            )
        # this solution cannot work out, as accelerator.gather does not support gather non-tensor objects across cards
        # fix: we still need it anyway as the default collator behaves badly on non-target item (i.e., indices)
        else:
            data_collator = DataCollatorForMultipleChoiceWithMetaInfo(
                tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None), meta_info={'indices', }
            )

    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)


    # Use the device given by the `accelerator` object.
    if not args.parallelize:
        model.to(accelerator.device)

    # Prepare everything with our `accelerator`.
    eval_dataloader = accelerator.prepare(eval_dataloader)


    # Metrics
    metric = load_metric("accuracy")

    # Eval!
    total_batch_size = args.per_device_eval_batch_size * accelerator.num_processes

    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
    logger.info(f"  Total eval batch size (w. parallel, distributed) = {total_batch_size}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)

    model.eval()
    all_predictions = [] if tmp_out_ds is None else tmp_out_ds["predictions"]
    all_indices = [] if tmp_out_ds is None else tmp_out_ds['indices']
    all_targets = [] if tmp_out_ds is None else tmp_out_ds['targets']
    counter = 0
    for batch in eval_dataloader:
        with torch.no_grad():
            predictions = model(batch)

        _gathered_predictions = accelerator.gather(predictions)
        _gathered_indices = accelerator.gather(batch['indices'])
        _gathered_targets = accelerator.gather(batch['targets'])
        metric.add_batch(
            predictions=_gathered_predictions.argmax(dim=-1) if _gathered_predictions.dim() > 1 else _gathered_predictions,
            references=_gathered_targets,
        )

        progress_bar.update(1)
        counter += 1
        #print(_gathered_predictions)
        #exit()
        #if isinstance(predictions, list):
            #all_predictions.extend(all_predictions)
        #else:
        # all_predictions.append(_gathered_predictions.cpu().numpy())
        # all_indices.append(_gathered_indices.cpu().numpy())
        # all_targets.append(_gathered_targets.cpu().numpy())
        all_predictions.extend(_gathered_predictions.cpu().numpy().tolist())
        all_indices.extend(_gathered_indices.cpu().numpy().tolist())
        all_targets.extend(_gathered_targets.cpu().numpy().tolist())
        if accelerator.is_main_process:
            if args.debug:
                aux_output = {
                    "predictions": all_predictions,
                    "indices": all_indices,
                    "targets": all_targets
                }
                aux_output_ds = Dataset.from_dict(aux_output)
                aux_output_ds.save_to_disk(tmp_path)
            else:
                if counter % 400 == 0:
                    aux_output = {
                        "predictions": all_predictions,
                        "indices": all_indices,
                        "targets": all_targets
                    }
                    aux_output_ds = Dataset.from_dict(aux_output)
                    aux_output_ds.save_to_disk(tmp_path)

    eval_metric = metric.compute()
    accelerator.print(f"Result: {eval_metric}")
    # all_predictions = np.concatenate(all_predictions, axis=0)
    # all_indices = np.concatenate(all_indices, axis=0)
    # all_targets = np.concatenate(all_targets, axis=0)

    results = {
        "dataset_name": args.dataset_name,
        "dataset_config_name": args.dataset_config_name,
        "template_name": args.template_name,
        "evaluation": eval_metric
    }
    if accelerator.is_main_process:
        # aux_output = [all_predictions, all_indices, all_targets]
        aux_output = {
            "predictions": all_predictions,
            "indices": all_indices,
            "targets": all_targets
        }
        aux_output_ds = Dataset.from_dict(aux_output)
        if args.output_dir is not None:
            with open(metrics_filepath, "w") as f:
                json.dump(results, f, indent=4)
            aux_output_ds.save_to_disk(auxoutput_filepath)


            # if args.total_splits < 0:
            #     with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            #         json.dump(results, f, indent=4)
            #     torch.save(aux_output, os.path.join(args.output_dir, "validation_predictions.p"))
            # else:
            #     with open(os.path.join(args.output_dir, f"results_{args.current_split}.json"), "w") as f:
            #         json.dump(results, f, indent=4)
            #     torch.save(aux_output, os.path.join(args.output_dir, f"validation_predictions_{args.current_split}.p"))


if __name__ == "__main__":
    main()
