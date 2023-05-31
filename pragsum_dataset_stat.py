import os
import matplotlib.pyplot as plt
from collections import Counter
import json
from util.get_wiki_dict import get_wiki_dict
# mainly used for summary-gold-evidence overlapping computation
if __name__ == '__main__':
    root_dir = "/data/chenghao/fool-me-twice/pragsum_dataset"
    visualization_dir = "visualization"
    os.makedirs(visualization_dir, exist_ok=True)
    output_dir = os.path.join(root_dir, "AgentTraining")
    # output_dir = os.path.join(root_dir, "AgentTraining_w_longt5_summary")
    #output_dir = os.path.join(root_dir, "AgentTraining_w_summary")
    #output_dir = os.path.join(root_dir, "AgentTraining_no_evidence")
    os.makedirs(output_dir, exist_ok=True)
    wiki_page_dict = get_wiki_dict("/data/chenghao/fool-me-twice/pages/")
    # only for longt5
    # root_dir = "/data/chenghao/fool-me-twice/pragsum_dataset/GeneralSummaryLongT5"
    for split in ['train', 'dev', 'test']:
        label_counter = Counter()
        # with open(os.path.join(output_dir, f"{split}.json"), "w") as f_out:
        contain_gold_evidence = Counter()
        contain_proportion = list()
        with open(os.path.join(root_dir, f"{split}.json"), "r", encoding='utf-8') as f_in:
            for line in f_in:
                data = json.loads(line)
                data['sentence1'] = data['text']
                label_counter[data['label']] += 1
                #data['sentence1'] = data['wiki-summary']
                # for gold evidence
                data['sentence2'] = " ".join([x['text'] for x in data['gold_evidence']])
                gold_evidence_section_header = [x['section_header'] for x in data['gold_evidence']]
                summary_in_gold_evidence = [x['text'] for x in data['gold_evidence'] if x['section_header'] == 'Summary']
                wiki_page = wiki_page_dict[data['wiki-page']]
                summary_all_num = len(wiki_page['summary_section'])
                proportion = len(summary_in_gold_evidence) / summary_all_num
                contain_proportion.append(proportion)
                if len(summary_in_gold_evidence) > 0:
                    contain_gold_evidence['contain'] += 1
                else:
                    contain_gold_evidence['not_contain'] += 1
                # for w_summary
                #data['sentence2'] = data['wiki-summary']
                # for no_evidence
                #data['sentence2'] = ""
                # for longt5 summary
                # data['sentence2'] = data['summary_longt5']
                # data.pop("wiki-non-summary")
                # f_out.write(json.dumps(data) + "\n")
        # plot the distribution of summary-gold-evidence overlapping
        # set font size to be 20
        plt.rcParams.update({'font.size': 20})
        plt.hist(contain_proportion, bins=20)
        plt.xlabel('proportion of summary in gold evidence')
        plt.ylabel('number of samples')
        plt.title(f'{split} set')
        plt.savefig(os.path.join(visualization_dir, f'{split}_summary_gold_evidence_overlapping.png'))
        plt.clf()
        plt.close()
        print(contain_gold_evidence)
        print(label_counter)

