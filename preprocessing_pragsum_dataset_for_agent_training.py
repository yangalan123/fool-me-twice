import os
from collections import Counter
import json
if __name__ == '__main__':
    #root_dir = "/data/chenghao/fool-me-twice/pragsum_dataset"
    root_dir = "/net/scratch/chenghao/fm2/pragsum_dataset"
    output_dir = os.path.join(root_dir, "AgentTraining_zeroshot")
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'dev', 'test']:
        label_counter = Counter()
        with open(os.path.join(output_dir, f"{split}.json"), "w") as f_out:
            with open(os.path.join(root_dir, f"{split}.json"), "r", encoding='utf-8') as f_in:
                for line in f_in:
                    data = json.loads(line)
                    data['sentence1'] = data['text']
                    label_counter[data['label']] += 1
                    if data['label'] == "SUPPORTS":
                        data['label'] = "entailment"
                    else:
                        assert data['label'] == "REFUTES"
                        data['label'] = "contradiction"
                    #data['sentence1'] = data['wiki-summary']
                    data['sentence2'] = " ".join([x['text'] for x in data['gold_evidence']])
                    #data['sentence2'] = data['wiki-summary']
                    f_out.write(json.dumps(data) + "\n")
        print(label_counter)

