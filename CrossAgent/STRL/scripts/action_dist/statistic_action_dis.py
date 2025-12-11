import json
import os

dir_path = "OpenHA/examples/output/mc-mix-01reward-coa-qwen2-vl-7b-250825_20-greedy"

def statistic_action(dir_path):
    action_num = {
        "raw": 0,
        "motion": 0,
        "grounding": 0
    }
    action_prop = {
        "raw": 0,
        "motion": 0,
        "grounding": 0
    }
    for datename in os.listdir(dir_path):
        task_dir = os.path.join(dir_path, datename)
        raw_actions_path = os.path.join(task_dir, "raw_action.jsonl")
        if not os.path.exists(raw_actions_path):
            continue
        with open(raw_actions_path, "r") as f:
            raw_actions = f.readlines()
        for line in raw_actions:
            if "Motion:" in line:
                action_num["motion"] += 1
            elif "Grounding:" in line:
                action_num["grounding"] += 1
            elif "Action:" in line:
                action_num["raw"] += 1
    
    action_num_sum = action_num["motion"]+action_num["grounding"] + action_num["raw"]
    return action_num

model_paths = [
    "OpenHA/examples/output/mc-mix-coa-qwen2-vl-7b-250816_400-greedy",
    "OpenHA/examples/output/mc-mix-01reward-coa-qwen2-vl-7b-250825_20-greedy",
    
]

statistic_info = {}
for model_path in model_paths:
    if model_path.split("/")[-1] not in statistic_info:
        statistic_info[model_path.split("/")[-1]] = {}
    for task_name in os.listdir(model_path):
        task_group = task_name.split("_")[0]
        if task_group not in statistic_info[model_path.split("/")[-1]]:
            statistic_info[model_path.split("/")[-1]][task_group] = {}
        task_path = os.path.join(model_path, task_name)
        if not os.path.isdir(task_path):
            continue
        action_num = statistic_action(task_path)
        print(f"Model: {model_path}, Task: {task_name}")
        print(f"Action Num: {action_num}")
        print("=====================================")

        if "action_num" not in statistic_info[model_path.split("/")[-1]][task_group]:
            statistic_info[model_path.split("/")[-1]][task_group]["action_num"] = {
                "motion": 0,
                "grounding": 0,
                "raw": 0
            }
        for action_type in action_num:
            statistic_info[model_path.split("/")[-1]][task_group]["action_num"][action_type] += action_num[action_type]
    
    for task_name in os.listdir(model_path):
        task_group = task_name.split("_")[0]
        if "action_num" not in statistic_info[model_path.split("/")[-1]][task_group]:
            continue
        action_num = statistic_info[model_path.split("/")[-1]][task_group]["action_num"]
        action_num_sum = action_num["motion"]+action_num["grounding"] + action_num["raw"]
        action_prop = {
            "motion": action_num["motion"] / action_num_sum,
            "raw": action_num["raw"] / action_num_sum,
            "grounding": action_num["grounding"] / action_num_sum
        }
        statistic_info[model_path.split("/")[-1]][task_group]["action_prop"] = action_prop

with open("action_dist.json", "w") as f:
    json.dump(statistic_info, f, indent=4)
