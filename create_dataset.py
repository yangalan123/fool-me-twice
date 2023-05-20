import glob
import json
import os

if __name__ == '__main__':
    source_dir = "pages/"
    matched_files = glob.glob(source_dir + "*.sentences.json")
    output_dir = "pragsum_dataset"
    os.makedirs(output_dir, exist_ok=True)
    page_dict = {}
    exist_summary_dict = dict()
    category_set = dict()
    summary_dict = dict()
    data_dict = dict()
    for file in matched_files:
        with open(file, "r") as f:
            data = json.load(f)
            sentences = data["sentences"]
            category = data['category']
            if category not in category_set:
                category_set[category] = []
            title = data["title"]
            data_dict[title] = data
            buf = []
            for sentence_item in sentences:
                text = sentence_item["line"]
                sentence_id = sentence_item["sentence"]
                section = sentence_item["name"]
                if section == "Summary" and sentence_id == 0:
                    # if title not in exist_summary_dict:
                    # assert title not in exist_summary_dict, f"{title} already exists, {exist_summary_dict[title]}, {section}, {sentence_id}, {file}"
                    # if title in exist_summary_dict:
                    #     assert exist_summary_dict[title]["summary"] == summary, f"{title} already exists, {exist_summary_dict[title]}\n, section: {section},\n {sentence_id}, \nfile: {file}"
                    if title in exist_summary_dict:
                        assert exist_summary_dict[title]['file'] == file, f"{title} already exists, {exist_summary_dict[title]}\n, section: {section},\n {sentence_id}, \nfile: {file}"
                    exist_summary_dict[title] = {
                        "section": section,
                        "sentence_id": sentence_id,
                        "file": file,
                        "category": category,
                        # "summary": summary,
                        # "non-summary": non_summary
                    }
                whole_id = sentence_item["id"]
                if sentence_id == 0:
                    buf.append(f"Section: {section}")
                buf.append(text)
            assert title not in page_dict, f"{title} already exists, {title}, {section}, {sentence_id}, {file}"
            page_dict[title] = "\n".join(buf)
    print(f"number of pages: {len(page_dict)}")
    print(f"number of pages with summary: {len(exist_summary_dict)} / {len(page_dict)} ({len(exist_summary_dict) / len(page_dict) * 100:.2f}%)")
    for title in exist_summary_dict:
        data = data_dict[title]
        sentences = data["sentences"]
        section_names = set([sentence_item["name"] for sentence_item in sentences])
        section_partition = dict()
        for section_name in section_names:
            section_partition[section_name] = [sentence_item for sentence_item in sentences if
                                               sentence_item["name"] == section_name]
        summary = "".join([x['line'] for x in section_partition['Summary']])
        non_summary = "".join(
            ["".join([x['line'] for x in y]) for y in section_partition.values() if y[0]['name'] != 'Summary'])
        exist_summary_dict[title]["summary"] = summary
        exist_summary_dict[title]["non-summary"] = non_summary
    coverage_count = 0
    all_instances = set()
    not_covered = set()
    covered = set()
    for split in ['train', 'dev', 'test']:
        source_dataset = os.path.join("dataset", f"{split}.jsonl")
        # target_data_domain = dict()
        target_data = []
        with open(source_dataset, "r", encoding='utf-8') as f:
            for line_i, line in enumerate(f):
                data = json.loads(line)
                text = data["text"]
                label = data["label"]
                page = data["wikipedia_page"]
                # num_count += 1
                all_instances.add((text, label))
                if page in page_dict:
                    coverage_count += 1
                    covered.add(page)
                else:
                    # try to resolve mismatch problem
                    # one possibility could be the page name replace space with underscore
                    page = page.replace(" ", "_")
                    if page in page_dict:
                        coverage_count += 1
                        covered.add(page)
                    else:
                        not_covered.add(page)
                if page in exist_summary_dict:
                    category = exist_summary_dict[page]["category"]
                    # if category not in target_data_domain:
                    #     target_data_domain[category] = dict()
                    # if split not in target_data_domain[category]:
                    #     target_data_domain[category][split] = []
                    data["wiki-summary"] = exist_summary_dict[page]["summary"]
                    data["wiki-non-summary"] = exist_summary_dict[page]["non-summary"]
                    data['wiki-page'] = page
                    target_data.append(data)
        with open(os.path.join(output_dir, f"{split}.json"), "w", encoding='utf-8') as f:
            for data in target_data:
                f.write(json.dumps(data) + "\n")

    with open(os.path.join(output_dir, "GeneralContext.json"), "w", encoding='utf-8') as f:
        for title in exist_summary_dict:
            f.write(json.dumps({
                "title": title,
                "text": exist_summary_dict[title]["non-summary"],
            }) + "\n")
    print(f"number of instances: {len(all_instances)}")
    print(f"number of instances with coverage: {coverage_count} / {len(all_instances)} ({coverage_count / len(all_instances) * 100:.2f}%)")
    print(f"number of pages not covered: {len(not_covered)}: {not_covered}")
    print(f"delta set: {set(page_dict.keys()) - covered}")





