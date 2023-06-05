import os
from collections import Counter
import json
if __name__ == '__main__':
    root_dir = "/data/chenghao/fool-me-twice/pragsum_dataset"
    #output_dir = os.path.join(root_dir, "AgentTraining_w_gold_evidence_longt5_summary")
    #output_dir = os.path.join(root_dir, "AgentTraining_w_wiki_summary_longt5_summary")
    output_dir = os.path.join(root_dir, "AgentTraining_w_gold_evidence_wiki_summary")
    # output_dir = os.path.join(root_dir, "AgentTraining_w_longt5_summary")
    #output_dir = os.path.join(root_dir, "AgentTraining_w_summary")
    #output_dir = os.path.join(root_dir, "AgentTraining_no_evidence")
    os.makedirs(output_dir, exist_ok=True)
    # only for longt5
    root_dir = "/data/chenghao/fool-me-twice/pragsum_dataset/GeneralSummaryLongT5"
    for split in ['train', 'dev', 'test']:
        label_counter = Counter()
        with open(os.path.join(output_dir, f"{split}.json"), "w") as f_out:
            with open(os.path.join(root_dir, f"{split}.json"), "r", encoding='utf-8') as f_in:
                for line in f_in:
                    data = json.loads(line)
                    data['sentence1'] = data['text']
                    label_counter[data['label']] += 1
                    #data['sentence1'] = data['wiki-summary']
                    # for gold evidence
                    # data['sentence2'] = " ".join([x['text'] for x in data['gold_evidence']])
                    # for w_summary
                    #data['sentence2'] = data['wiki-summary']
                    # for no_evidence
                    #data['sentence2'] = ""
                    # for longt5 summary
                    # data['sentence2'] = data['summary_longt5']
                    # for longt5 summary + gold evidence
                    #data['sentence2'] = " ".join([x['text'] for x in data['gold_evidence']]) + " " + data['summary_longt5']
                    # for longt5 summary + wiki summary
                    #data['sentence2'] = data['wiki-summary'] + " " + data['summary_longt5']
                    # for gold_evidence + wiki summary
                    data['sentence2'] = " ".join([x['text'] for x in data['gold_evidence']]) + " " + data['wiki-summary']
                    data.pop("wiki-non-summary")
                    f_out.write(json.dumps(data) + "\n")
        print(label_counter)

