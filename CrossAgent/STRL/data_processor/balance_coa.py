import json
from tqdm import tqdm
import random

with open("OpenHA/rl/datasets/sft/ssrl_coldstart/20250906/mc-mix-coa-ssrl_base-qwen2-vl-7b-250830.jsonl"):
    datas = [json.loads(line) for line in open("OpenHA/rl/datasets/sft/ssrl_coldstart/20250906/mc-mix-coa-ssrl_base-qwen2-vl-7b-250830.jsonl")]

new_datas = []

for coa in ["Motion:", "Grounding:", "Action:"]:

    single_num = 0
    while True:
        for data in tqdm(datas):
            solution = data["conversations"][-1]["content"][0]["text"]
            if coa in solution:
                new_datas.append(data)
                single_num += 1
            if single_num >= 10000:
                break
        
        if single_num >= 10000:
            break

random.shuffle(new_datas)
with open("OpenHA/rl/datasets/sft/ssrl_coldstart/20250906/mc-mix-coa-ssrl_base-qwen2-vl-7b-250830-balance_coa.jsonl", "w") as f:
    for data in new_datas:
        f.write(json.dumps(data) + "\n")