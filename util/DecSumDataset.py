import json
import random
from datasets import load_dataset
from transformers import PretrainedConfig
import datasets
from datasets import Dataset
import torch
import numpy as np
import os
import copy

process_func = lambda x: float(-np.log(x + 1e-7))
DEFAULT_DATASET_ROOT_DIR = "/net/scratch/chenghao/fm2/pragsum_dataset"
DEFAULT_TARGET_SUMMARY_DIR = "/net/scratch/chenghao/fm2/general_summary_longt5/GeneralContext/train_sample_16"
def get_score_from_output(score_list, label, compute_acc=False):
    # print(score_list, label)
    for item in score_list:
        if item['label'].lower() == label.lower():
            if not compute_acc:
                return process_func(item['score'])
            else:
                prediction_id = np.argmax([x['score'] for x in score_list])
                prediction = score_list[prediction_id]['label']
                return 0.0 if prediction.lower() != label.lower() else 1.0


# get_score_from_output = lambda score_list, label: \
#     [process_func(x['score']) for x in score_list if x['label'].lower() == label.lower()][0]

# reuse the code from post_processing_summary.py
def load_summary_and_create_context_dict(root_dir=DEFAULT_DATASET_ROOT_DIR, target_summary_dir=DEFAULT_TARGET_SUMMARY_DIR):
    general_context = []
    with open(os.path.join(root_dir, "GeneralContext.json"), "r") as f:
        for line in f:
            data = json.loads(line)
            general_context.append(data)
    # target_dir = "/net/scratch/chenghao/fm2/general_summary_longt5/GeneralContext/train_sample_1"
    # target_summary_dir = "/net/scratch/chenghao/fm2/general_summary_longt5/GeneralContext/train_sample_16"
    generated_predictions_files = [os.path.join(target_summary_dir, x) for x in os.listdir(target_summary_dir) if x.startswith("generated_predictions")]
    for file in generated_predictions_files:
        with open(os.path.join(target_summary_dir, file), "r") as f:
            buf = f.read()
            summaries = buf.split("\n\n\n")
        assert len(summaries) % len(general_context) == 0, f"summarization number {len(summaries)} must be divisible by context number {len(general_context)}"
        num_summary_per_context = len(summaries) // len(general_context)
        for i in range(len(general_context)):
            if "summary_longt5" not in general_context[i]:
                general_context[i]["summary_longt5"] = []
            general_context[i]["summary_longt5"].extend(summaries[i * num_summary_per_context: (i + 1) * num_summary_per_context])

    context_dict = dict()
    for i in range(len(general_context)):
        context_dict[general_context[i]['title']] = general_context[i]

    return context_dict

def loadDecSumdataset(training_args, data_args, model_args, model, tokenizer, logger, raw=False):
    # Loading a dataset from your local files.
    # CSV/JSON training and evaluation files are needed.
    if data_args.test_file is not None:
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file, "test": data_args.test_file}
    else:
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file,}

    # Get the test dataset: you can provide your own CSV/JSON test file (see below)
    # when you use `do_predict` without specifying a GLUE benchmark task.
    if training_args.do_predict:
        if data_args.test_file is not None:
            train_extension = data_args.train_file.split(".")[-1]
            test_extension = data_args.test_file.split(".")[-1]
            assert (
                    test_extension == train_extension
            ), "`test_file` should have the same extension (csv or json) as `train_file`."
            data_files["test"] = data_args.test_file
            print(f"set up test files : {data_args.test_file}")
        else:
            raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

    if data_args.train_file.endswith(".csv"):
        # Loading a dataset from local csv files
        raw_datasets = load_dataset(
            "csv",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        # Loading a dataset from local json files
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features[data_args.label_column].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features[data_args.label_column].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique(data_args.label_column)
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
    non_label_column_names = [name for name in raw_datasets["train"].column_names if name != data_args.label_column]
    if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    else:
        sentence1_key, sentence2_key = data_args.text_column, None
        # if len(non_label_column_names) >= 2:
        #     sentence1_key, sentence2_key = non_label_column_names[:2]
        # else:
        #     # sentence1_key, sentence2_key = non_label_column_names[0], None
        #     sentence1_key, sentence2_key = "text", None
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and data_args.task_name is not None
            and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    # if label_to_id is not None:
    #     model.config.label2id = label_to_id
    #     model.config.id2label = {id: label for label, id in config.label2id.items()}
    # elif data_args.task_name is not None and not is_regression:
    #     model.config.label2id = {l: i for i, l in enumerate(label_list)}
    #     model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and data_args.label_column in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples[data_args.label_column]]
        if data_args.label_column != "label" and "label" in examples:
            result["label"] = examples[data_args.label_column]
            result['original_label'] = examples['label']
        return result

    if not raw:
        with training_args.main_process_first(desc="dataset map pre-processing"):
            raw_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    eval_dataset = None
    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    predict_dataset = None
    if training_args.do_predict:
        # if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))


    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    if training_args.do_eval:
        for index in random.sample(range(len(eval_dataset)), 3):
            logger.info(f"Sample {index} of the evaling set: {eval_dataset[index]}.")
    if training_args.do_predict:
        for index in random.sample(range(len(predict_dataset)), 3):
            logger.info(f"Sample {index} of the predicting set: {predict_dataset[index]}.")

    return train_dataset, eval_dataset, predict_dataset


def prepare_dataset_partition_for_best_of_k(file_path, split, output_eval_dir, processed_summary_dump_path, raw_datasets, logger):
    # for split, file_path in zip(['train', 'dev'], [data_args.train_file, data_args.validation_file]):
    _raw_dataset = raw_datasets[split]
    # fixed using train_sample_xxx in post_processing_summary.py
    processed_summary_dump = datasets.Dataset.load_from_disk(processed_summary_dump_path)
    # output_prediction_files = glob.glob(os.path.join(output_summary_dir, "generated_predictions_*"))
    # create evaluation output dir in the parent dir of the file_path
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
    logger.warning("Loaded {} examples from {}".format(len(summary_data), file_path))
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
    return summary_data, summary_pointers, _raw_dataset, dataset, original_test_ds
