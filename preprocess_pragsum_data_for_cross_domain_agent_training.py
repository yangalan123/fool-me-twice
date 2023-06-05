import os
from collections import Counter
import json
from util.get_wiki_dict import get_wiki_dict
from datasets import load_dataset
def add_claims(example):
    example['sentence1'] = example['text']
    return example

def add_evidence(example, category):
    # example['sentence2'] = " ".join([x['text'] for x in example['gold_evidence']])
    # return example
    if example['category'] == category:
        example['sentence2'] = example["wiki-summary"]
    else:
        example['sentence2'] = ""
    return example

if __name__ == '__main__':
    root_dir = "/net/scratch/chenghao/fm2/pragsum_dataset"
    #output_dir = os.path.join(root_dir, "AgentTraining_w_gold_evidence_longt5_summary")
    output_dir = os.path.join(root_dir, "AgentTraining_domain_wise")
    # output_dir = os.path.join(root_dir, "AgentTraining_w_longt5_summary")
    #output_dir = os.path.join(root_dir, "AgentTraining_w_summary")
    #output_dir = os.path.join(root_dir, "AgentTraining_no_evidence")
    os.makedirs(output_dir, exist_ok=True)
    # only for longt5
    # root_dir = "/net/scratch/chenghao/fm2/pragsum_dataset/GeneralSummaryLongT5"
    wiki_page_dict = get_wiki_dict("/net/scratch/chenghao/fm2/pages/")
    all_categories = set([x['category'] for x in wiki_page_dict.values()])
    data_files = {
        "train": os.path.join(root_dir, "train.json"),
        "dev": os.path.join(root_dir, "dev.json"),
    }
    dataset = load_dataset('json', data_files=data_files)
    dataset = dataset.map(add_claims, num_proc=4)

    for _category in all_categories:
        # for split in ['train', 'dev']:
        label_counter = Counter()
        agent_output_dir = os.path.join(output_dir, _category)
        os.makedirs(agent_output_dir, exist_ok=True)
        remained_dataset = None
        # domain_train_dataset = dataset["train"].filter(lambda x: x['category'] == _category)
        domain_train_dataset = dataset['train'].map(lambda x: add_evidence(x, _category), num_proc=4)
        # domain_eval_test_dataset = dataset["dev"].filter(lambda x: x['category'] == _category).train_test_split(test_size=0.5, seed=42)
        # [TODO]: we may also want to consider split training set later, as we only have limited number of validation set,
        #  [TODO]: but for now, in order to avoid retraining the agent, we just use the same training set (with different setting of "sentence2")
        domain_eval_test_dataset = dataset["dev"].map(lambda x: add_evidence(x, _category)).train_test_split(test_size=0.5, seed=42)
        domain_eval_dataset = domain_eval_test_dataset['train']
        domain_test_dataset = domain_eval_test_dataset['test']
        domain_train_dataset.to_json(os.path.join(agent_output_dir, f"train.json"))
        domain_eval_dataset.to_json(os.path.join(agent_output_dir, f"dev.json"))
        domain_test_dataset.to_json(os.path.join(agent_output_dir, f"test.json"))

            # with open(os.path.join(agent_output_dir, f"{split}.json"), "w") as f_out:
            #     for line in f_in:
            #         data = json.loads(line)
            #         data['sentence1'] = data['text']
            #         label_counter[data['label']] += 1
            #         #data['sentence1'] = data['wiki-summary']
            #         # for gold evidence
            #         # data['sentence2'] = " ".join([x['text'] for x in data['gold_evidence']])
            #         # for w_summary
            #         #data['sentence2'] = data['wiki-summary']
            #         # for no_evidence
            #         #data['sentence2'] = ""
            #         # for longt5 summary
            #         # data['sentence2'] = data['summary_longt5']
            #         # for longt5 summary + gold evidence
            #         data['sentence2'] = " ".join([x['text'] for x in data['gold_evidence']]) + " " + data['summary_longt5']
            #         data.pop("wiki-non-summary")
            #         f_out.write(json.dumps(data) + "\n")
            # print(label_counter)

