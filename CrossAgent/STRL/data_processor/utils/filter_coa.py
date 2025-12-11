import re
from tqdm import tqdm
from openagents.agents.utils.system_prompt import coa_system_prompt
from openagents.agents.utils.system_prompt import mix_action_system_prompt
import math
import torch
from minestudio.simulator.entry import MinecraftSim
import numpy as np
from copy import deepcopy
import random
from openagents.agents.utils.action_mapping import TextActionTokenizer
import json

random.seed(42)
tokenizer = TextActionTokenizer()


with open("OpenHA/rl/datasets/sft/msrl_coldstart/20250909/mc-mix-coa-qwen2-vl-7b-250906_120-withsystem.jsonl") as f:
    datas = [json.loads(line) for line in f.readlines()]

new_datas = []

for data in tqdm(datas):
    if len(data["conversations"][-1]["content"]) > 1:
        import pdb
        pdb.set_trace() 
        continue
    solution = data["conversations"][-1]["content"][0]["text"]
    data["conversations"][-1]["content"][0]["text"] = data["conversations"][-1]["content"][0]["text"].replace("<|im_start|>assistant\n", "")

    for i in range(len(data["conversations"]) - 1):
        if "Action:" in data["conversations"][i]["content"][0]["text"] and i != 0:
            assert data["conversations"][i]["role"] == "assistant"
            data["conversations"][i]["content"][0]["text"] = "Action:" + data["conversations"][i]["content"][0]["text"].split("Action:")[1]

    

    sa = deepcopy(tokenizer.decode(solution)[0])
    # import pdb
    # pdb.set_trace() 
    #sa["camera"] = np.array(sa["camera"])
    sa["camera"] = np.round(sa["camera"], 0)

    valid = False
    for key in sa:
        if key != "camera":
            if 0.0 != sa.get(key, None):
                valid = True
                break
        if key == "camera":
            if not np.array_equal(np.array([0.0, 0.0]), sa.get("camera", None)):
                valid = True
                break


    if valid:
        new_datas.append(data)

    else:
        print(f"filter one no-op data: {solution}")

with open("OpenHA/rl/datasets/sft/msrl_coldstart/20250909/mc-mix-coa-qwen2-vl-7b-250906_120-withsystem-onlyraw.jsonl", "w") as f:
    for data in new_datas:
        f.write(json.dumps(data) + "\n")