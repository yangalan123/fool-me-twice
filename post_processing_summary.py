import os
import json
if __name__ == '__main__':
    target_dir = "/data/chenghao/fool-me-twice/general_summary_longt5/GeneralContext/train_sample_1"
    with open(os.path.join(target_dir, "generated_predictions_0_1sample.txt"), "r") as f:
        buf = f.read()
        summaries = buf.split("\n\n\n")
    general_context = []
    root_dir = "/data/chenghao/fool-me-twice/pragsum_dataset"
    with open(os.path.join(root_dir, "GeneralContext.json"), "r") as f:
        for line in f:
            data = json.loads(line)
            general_context.append(data)
    assert len(summaries) == len(general_context), f"summarization length mismatch: {len(summaries)} != {len(general_context)}"
    context_dict = dict()
    for i in range(len(summaries)):
        general_context[i]["summary_longt5"] = summaries[i]
        context_dict[general_context[i]['title']] = general_context[i]
    for split in ['train', 'dev', 'test']:
        output_dir = os.path.join(root_dir, "GeneralSummaryLongT5")
        os.makedirs(output_dir, exist_ok=True)
        print("processing", split)
        with open(os.path.join(output_dir, f"{split}.json"), "w") as f_out:
            with open(os.path.join(root_dir, f"{split}.json"), "r", encoding='utf-8') as f_in:
                for line in f_in:
                    data = json.loads(line)
                    if data['wiki-page'] in context_dict:
                        data['summary_longt5'] = context_dict[data['wiki-page']]['summary_longt5']
                    f_out.write(json.dumps(data) + "\n")

