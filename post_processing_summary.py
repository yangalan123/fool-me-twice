import glob
import datasets
import os
import json
# redistributed generated general summary under each entry
if __name__ == '__main__':
    general_context = []
    root_dir = "/net/scratch/chenghao/fm2/pragsum_dataset"
    output_root_dir = os.path.join(root_dir, "AgentTraining_domain_wise")
    with open(os.path.join(root_dir, "GeneralContext.json"), "r") as f:
        for line in f:
            data = json.loads(line)
            general_context.append(data)
    # target_dir = "/net/scratch/chenghao/fm2/general_summary_longt5/GeneralContext/train_sample_1"
    target_summary_dir = "/net/scratch/chenghao/fm2/general_summary_longt5/GeneralContext/train_sample_16"
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

    domain_dirs = glob.glob(os.path.join(output_root_dir, "*"))
    for domain_dir in domain_dirs:
        for split in ['train', 'dev', 'test']:
            # os.makedirs(output_dir, exist_ok=True)
            print("processing", split)
            split_dict = dict()
            # with open(os.path.join(output_root_dir, f"{split}.json"), "w") as f_out:
            # with open(os.path.join(root_dir, f"{split}.json"), "r", encoding='utf-8') as f_in:
            with open(os.path.join(domain_dir, f"{split}.json"), "r", encoding='utf-8') as f_in:
                for line in f_in:
                    data = json.loads(line)
                    # if data['wiki-page'] in context_dict:
                    assert data['wiki-page'] in context_dict, f"{data['wiki-page']} not in context_dict for {split} at {domain_dir}"
                    data['summary_longt5'] = list(set(context_dict[data['wiki-page']]['summary_longt5']))
                    for key in data:
                        if key not in split_dict:
                            split_dict[key] = []
                        split_dict[key].append(data[key])
                dataset = datasets.Dataset.from_dict(split_dict)
                dataset.save_to_disk(os.path.join(domain_dir, f"{split}_with_summary_longt5.ds"))
                    # f_out.write(json.dumps(data) + "\n")

