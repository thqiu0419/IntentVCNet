import json
import re

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')



source_data = read_jsonl("/data3/InternVL/internvl_chat/shell/data/eval_public_data_f32.jsonl")
target_data = []
for data in source_data:
    data['conversations'][0]['value'] = data['conversations'][0]['value'].split(' The coordinates in each frame are')[0]
    target_data.append(data)
write_jsonl(target_data, "/data3/InternVL/internvl_chat/shell/data/eval_public_data-text.jsonl")