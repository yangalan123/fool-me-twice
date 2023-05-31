import glob
import json
def get_wiki_dict(source_dir_for_pages):
    source_dir = source_dir_for_pages
    matched_files = glob.glob(source_dir + "*.sentences.json")
    data_dict = dict()
    # category_set = dict()
    for file in matched_files:
        with open(file, "r") as f:
            data = json.load(f)
            sentences = data["sentences"]
            category = data['category']
            # if category not in category_set:
            #     category_set[category] = []
            title = data["title"]
            data['summary_section'] = [x for x in sentences if x['name'] == 'Summary']
            data_dict[title] = data
    return data_dict
