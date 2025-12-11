
from openagents.agents.utils.system_prompt import mix_action_system_prompt
import json
import cv2
from tqdm import tqdm

system_prompt = mix_action_system_prompt

def apply_system_message(example):
    assert example["conversations"][0]["role"] == "user"
    if system_prompt not in example["conversations"][0]["content"][0]["text"]:
        example["conversations"][0]["content"][0]["text"] = system_prompt + example["conversations"][0]["content"][0]["text"]

    #only raw history
    assert example["conversations"][-1]["role"] == "assistant"
    for i in range(1, len(example["conversations"])-1):
        if example["conversations"][i]["role"] == "assistant":
            for j in range(len(example["conversations"][i]["content"])):
                example["conversations"][i]["content"][j]["text"] = "Action:" + example["conversations"][i]["content"][j]["text"].split("Action:")[-1]


for coa_type in ["text", "grounding", "motion"]:

    with open(f"/share/hkc/datasets/mc-msrl_coldstart-{coa_type}_250923-train.jsonl", "r") as f:
        dataset = [json.loads(line) for line in f]
        for data in tqdm(dataset, desc="Applying system prompt"):
            data = apply_system_message(data)


    with open(f"OpenHA/rl/datasets/sft/msrl_coldstart/20250923/mc-msrl_coldstart-{coa_type}_250923-train-withsystem.jsonl", "w") as f:
        for data in dataset:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")