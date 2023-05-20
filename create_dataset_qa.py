import os
from datasets import Dataset
from collections import Counter
import json
if __name__ == '__main__':
    #root_dir = "/data/chenghao/fool-me-twice/pragsum_dataset"
    root_dir = "/net/scratch/chenghao/fm2/pragsum_dataset"
    output_dir = os.path.join(root_dir, "AgentTraining_zeroshot_qa_t0")
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'dev', 'test']:
        label_counter = Counter()
        all_data = {
            "question":[],
            "output": [],
            "options": [],
            "context": [],
        }
        #with open(os.path.join(output_dir, f"{split}.json"), "w") as f_out:
        with open(os.path.join(root_dir, f"{split}.json"), "r", encoding='utf-8') as f_in:
            for line in f_in:
                data = json.loads(line)
                all_data['question'].append("Is the following claim supported by the given context? {}".format(data['text']))
                label_counter[data['label']] += 1
                all_data['output'].append(data['label'])
                all_data['options'].append(["SUPPORTS", "REFUTES"])
                #if data['label'] == "SUPPORTS":
                    #data['label'] = "entailment"
                #else:
                    #assert data['label'] == "REFUTES"
                    #data['label'] = "contradiction"
                #data['sentence1'] = data['wiki-summary']
                all_data['context'].append(" ".join([x['text'] for x in data['gold_evidence']]))
                #data['sentence2'] = data['wiki-summary']
                #f_out.write(json.dumps(data) + "\n")
        ds = Dataset.from_dict(all_data)
        _split = split if split != "dev" else "validation"
        ds.save_to_disk(os.path.join(output_dir, f"{_split}.ds"))
        print(label_counter)

