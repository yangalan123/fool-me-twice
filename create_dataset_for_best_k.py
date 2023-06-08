import argparse
import copy
import traceback
import json
import loguru

import torch
import traceback
from datasets import load_dataset
from tqdm import tqdm
import datasets, transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, HfArgumentParser, pipeline, TrainingArguments, AutoModelForCausalLM, \
    AutoModelForSeq2SeqLM
from transformers.trainer_utils import get_last_checkpoint

from trl import AutoModelForSeq2SeqLMWithValueHead, AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from tqdm import trange, tqdm

# from utils import loadDecSumdataset

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
from transformers.pipelines.pt_utils import KeyDataset, KeyPairDataset

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
    time_to_live = 39600  # just a bit less than 4 hrs
    schedule_death(time_to_live)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task"},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    text_column: Optional[str] = field(default='text', metadata={
        "help": "The name of the column in the datasets containing the full texts."})
    label_column: Optional[str] = field(default='label', metadata={
        "help": "The name of the column in the datasets containing the full labels."})
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            # if self.task_name not in task_to_keys.keys():
            #     raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
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

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    generator_model_name: Optional[str] = field(default="lvwerra/t5-imdb",
                                                metadata={"help": "the model name for generator"})
    agent_model: Optional[str] = field(default="lvwerra/distilbert-imdb", metadata={"help": "the agent model name"})
    # reward_model: Optional[str] = field(default="lvwerra/distilbert-imdb", metadata={"help": "the reward model name"})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    # use_validation_dataset: Optional[bool] = field(default=False, metadata={"help": "use validation dataset"})
    # num_samples_per_instance: Optional[int] = field(default=1,
    #                                                 metadata={"help": "number of samples to be generated per instance"})
    # def __post_init__(self):
    #     assert self.batch_size >= self.mini_batch_size, "batch_size must be >= mini_batch_size"
    #     if self.use_lora:
    #         if self.num_shared_layers != 0:
    #             print("WARNING: num_shared_layers is ignored when using lora")

def add_gt_evidence(example):
    if "sentence1" not in example:
        example['sentence1'] = example['text']
    example['sentence2'] = example['sentence2']  + " ".join([x['text'] for x in example['gold_evidence']])
    return example

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

    setup_slurm()
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(script_args.generator_model_name)
    if "gpt2" in script_args.generator_model_name:
        tokenizer.pad_token = tokenizer.eos_token

    decoder_only_flag = False
    try:
        model = AutoModelForCausalLM.from_pretrained(script_args.generator_model_name)
        decoder_only_flag = True
    except:
        model = AutoModelForSeq2SeqLM.from_pretrained(script_args.generator_model_name)

    # We retrieve the dataloader by calling the `build_dataset` function.
    # dataset = build_imdb_dataset(tokenizer)
    train_dataset, eval_dataset, test_dataset = loadDecSumdataset(training_args, data_args, script_args, model,
                                                                  tokenizer, logger, raw=True)
    generation_kwargs = {"num_beams": 1, "do_sample": True, "eos_token_id": -1}
    if "gpt2" in script_args.generator_model_name:
        generation_kwargs['pad_token_id'] = tokenizer.eos_token_id

    # if decoder_only_flag:
    #     # for general decoder-only model, we need to set the max_length to the model's max length
    #     generation_kwargs['max_length'] = tokenizer.model_max_length
    # else:
    generation_kwargs["max_new_tokens"] = data_args.max_target_length
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "softmax", "batch_size": training_args.per_device_train_batch_size}

    # if script_args.use_validation_dataset:
    #     dataset_used_for_rej = eval_dataset
    # else:
    #     dataset_used_for_rej = train_dataset

    clf_pipe = pipeline(task="text-classification", model=script_args.agent_model,
                        device=0 if torch.cuda.is_available() else "cpu")
    save_root_dir = training_args.output_dir
    os.makedirs(save_root_dir, exist_ok=True)


    def collater(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    for dataset_used_for_rej, split in zip([train_dataset, eval_dataset, test_dataset], ["train", "dev", "test"]):
        # train_dataloader = DataLoader(dataset_used_for_rej, batch_size=training_args.per_device_train_batch_size, collate_fn=collater)
        all_outputs = []
        # for batch in tqdm(train_dataloader):
        # processed_dataset_for_pipe = dataset_used_for_rej
        # if split == "dev":
        _dataset = dataset_used_for_rej.map(add_gt_evidence, num_proc=4)
        processed_dataset_for_pipe = KeyPairDataset(_dataset, "sentence1", "sentence2")
        with torch.no_grad():
            # outputs = list(clf_pipe(processed_dataset_for_pipe, **sent_kwargs))
            outputs = []
            for out in tqdm(clf_pipe(processed_dataset_for_pipe, **sent_kwargs), desc="Producing outputs", total=len(processed_dataset_for_pipe), leave=False, position=0):
                outputs.append(out)
            print("producing outputs done")
            labels = dataset_used_for_rej[data_args.label_column]
            _outputs = [get_score_from_output(x, label) for x, label in zip(outputs, labels)]
            # for batch_i, _item in enumerate(batch):
            all_keys = _dataset[0].keys()
            new_dataset = _dataset.add_column("difference", _outputs)
            # below code is too slow, let's use huggingface
            # for output_i in trange(len(_outputs), desc="Processing outputs", position=1, leave=False):
            #     new_dict = {}
            #     for key in all_keys:
            #         # new_dict[key] = copy.deepcopy(batch[key][batch_i])
            #         try:
            #             if torch.is_tensor(_dataset[key][output_i]):
            #                 if len(_dataset[key][output_i]) == 1:
            #                     new_dict[key] = _dataset[key][output_i].item()
            #                 else:
            #                     print(f"Warning: tensor with length > 1: {key}" + str(_dataset[key][output_i]))
            #                     exit()
            #                     new_dict[key] = _dataset[key][output_i].tolist()
            #             elif isinstance(_dataset[key][output_i], list):
            #                 new_dict[key] = [x.item() if torch.is_tensor(x) else x for x in _dataset[key][output_i] ]
            #             else:
            #                 new_dict[key] = copy.deepcopy(_dataset[key][output_i])
            #         except Exception as e:
            #             print("key:", key)
            #             print("batch_i:", output_i)
            #             print(type(_dataset[key]))
            #             print(_dataset[key][output_i])
            #             print("Exception:", e)
            #             traceback.print_exc()
            #             exit()
            #     # for _key in _item:
            #     #     try:
            #     #         new_dict[_key] = _item[_key]
            #     #     except Exception as e:
            #     #         print(e, "\nKey:" + _key + "\nItem:" + str(_item))
            #     #         traceback.print_exc()
            #     #         exit()
            #     # new_dict["difference"] = abs(new_dict["label"] - float(_outputs[batch_i]))
            #     # new_dict['difference_sign'] = (float(_outputs[batch_i]) - new_dict['label']) >= 0
            #     new_dict['difference'] = float(_outputs[output_i])
            #     # for key in batch:
            #     #     print(key, batch[key])
            #     #     print(type(batch[key]))
            #     # exit()
            #     all_outputs.append(new_dict)
        # with open(os.path.join(save_root_dir, f"{split}_difference.json"), "w") as f:
        #     for line in all_outputs:
        #         f.write(json.dumps(line) + "\n")
        new_dataset.to_json(os.path.join(save_root_dir, f"{split}_difference.json"))
            # batch = {k: v.to(training_args.device) for k, v in batch.items()}
            # with torch.no_grad():
            #     outputs = sentiment_pipe(batch["source_text"], **sent_kwargs)
            #     batch["sentiment"] = torch.tensor([x['label'] for x in outputs]).to(training_args.device)
