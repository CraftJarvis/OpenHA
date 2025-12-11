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
from rl.datasets.utils.instructions import get_task_type


tokenizer = TextActionTokenizer()
def extract_solution(solution_str):
    assert "<raw>" in solution_str and "</raw>" in solution_str, "Solution string must contain <raw> tags."
    return solution_str.split("<raw>")[1].split("</raw>")[0].strip() if "<raw>" in solution_str and "</raw>" in solution_str else None

dummy_env = MinecraftSim()
CAMERA_SCALER = 360.0 / 2400.0
def motion2action(motion_x, motion_y, speed: float = 10):
    camera_x, camera_y = motion_x, motion_y
    distance =max(abs(camera_x), abs(camera_y))
    num_steps= int(random.uniform(5, 10) * math.sqrt(distance) / speed)
    if num_steps < 1:
        num_steps = 1
    
    d1 = (camera_x / num_steps) 
    d2 = (camera_y / num_steps) 
    action = dummy_env.noop_action() 
    action['camera'] = np.array([d2 * CAMERA_SCALER, d1 * CAMERA_SCALER])
    return (d2 * CAMERA_SCALER, d1 * CAMERA_SCALER)



def extract_solution_vla(solution_str):
    print("[DEBUG] Groundtruth:", tokenizer.decode(solution_str))
    return solution_str

    
    # import pdb
    # pdb.set_trace()
    # try:
    #     if "move(" not in solution_str:
    #         return "click"
    #     else:
    #         ogn_point_pos = solution_str.split("The cursor is located at")[1].split("[[")[1].split("]]")[0].strip()
    #         ogn_point_pos = [float(x) for x in ogn_point_pos.split(",")]
    #         obj_point_pos = solution_str.split("move_to")[1].split("[[")[1].split("]]")[0].strip()
    #         obj_point_pos = [float(x) for x in obj_point_pos.split(",")]
    #         motion_x = 640*((obj_point_pos[0] - ogn_point_pos[0])/100)
    #         motion_y = 360*((obj_point_pos[1] - ogn_point_pos[1])/100)
    #         return "move_to"+ str(motion2action(motion_x, motion_y))
    # except Exception as e:
    #     import pdb
    #     pdb.set_trace()

def count_suffix(lst):
    if not lst:
        return 0
    last = lst[-1]
    count = 0

    try:
        for x in reversed(lst):
            if (str(x) == str(last)):
                count += 1
            else:
                break
    except Exception as e:
        import pdb
        pdb.set_trace()
    return count


def convert_content_to_str(conversations):
    new_conversations = []
    for message in conversations:
        # 拼接 content list 里的各项
        parts = []
        try:
            for c in message["content"]:
                if c["type"] == "text":
                    parts.append(c["text"])
                else:
                    # image / video / 其他都直接保留占位符
                    parts.append(c.get("text", f"<{c['type']}>"))
        except:
            import pdb
            pdb.set_trace()
        # 新的 message，content 变成字符串
        new_conversations.append({
            "role": message["role"],
            "content": "".join(parts)
        })
    return new_conversations



def parse_example_mixcoa(example, history_length=-1):

    assert 2*(history_length+1) <= len(example["conversations"]), "Conversations length mismatch."

    example_new = deepcopy(example)
    if history_length == -1:
        history_length = random.randint(0, min(len(example["image"])-1, 9))

    example_new["conversations"] = example["conversations"][:2*(history_length+1)]
    example_new["image"] = example["image"][:(history_length+1)]

    #除了最后一位的solution，其他的history中只有raw action
    for i in range(len(example_new["conversations"]) - 1):
        if "Action:" in example_new["conversations"][i]["content"][0]["text"] and i != 0:
            assert example_new["conversations"][i]["role"] == "assistant"
            example_new["conversations"][i]["content"][0]["text"] = "Action:" + example_new["conversations"][i]["content"][0]["text"].split("Action:")[1]

    
    images = []
    for image_dict in example_new["image"]:
        images.append({
            "image": image_dict["image_path"].replace("/DATA", "/public/hgx14")
        })
    
    return example_new, images




def make_map_fn(split, data_source="coa_craft", system_prompt = mix_action_system_prompt, history_length=-1):
    def map_fn(example, idx):

        if len(example['image'])==0: 
            return None

        ogn_example = deepcopy(example)
        example, images = parse_example_mixcoa(example, history_length=history_length)

        try:
            task_type = get_task_type(example["conversations"][0]["content"][0]["text"].replace(system_prompt, "").strip())
            if task_type == "unknown":
                task_type = get_task_type(example["conversations"][0]["content"][1]["text"].replace(system_prompt, "").strip())
        except Exception as e:
            return None

        if system_prompt not in example["conversations"][0]["content"][0]["text"]:
            example["conversations"][0]["content"][0]["text"] = system_prompt + example["conversations"][0]["content"][0]["text"]

        #filter no_op


        prompt_conversations = deepcopy(example["conversations"][:-1])
        for conv in prompt_conversations:
            new_content = ""
            for i in range(len(conv["content"])):
                new_content = new_content+conv["content"][i]["text"]
            conv["content"] = new_content
            
        solution  = example["conversations"][-1]["content"][0]["text"]
        if "Action: move(0, 0) and press()" in solution and "Action: move(0, 0) and press() and" not in solution:
            #print("Filtered no-op action:", solution)
            return None


        return {
            "data_source": data_source,
            "prompt": prompt_conversations,
            "images": images,
            "ability": "coa_craft",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution if solution else ""
            },
            "extra_info": {
                "split": split,
                "index": idx, 
            },
            "task_type": task_type
        }
    return map_fn


def load_json_or_jsonl(file_path):
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r') as f:
            return [json.loads(line) for line in f]
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported file format. Use .json or .jsonl")


if __name__ == "__main__":
    import argparse
    import json
    import os
    import pandas as pd
    import random
    parser = argparse.ArgumentParser(description="Process dataset for COA-Craft.")
    parser.add_argument("--train_path", type=str, default="/public/hgx14/datasets/minecraft-trajectory.cache/mc-mix-coa_coldstart-0906.jsonl")
    parser.add_argument("--valid_path", type=str, default=None) #"/share/hkc/craft_cold_start/utils/balanced-valid.json")
    parser.add_argument("--output_path", type=str, default="OpenHA/rl/datasets/ssrl/20250906/mc-mix-coa")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    task_types = ["craft_item", "kill_entity", "mine_block", "unknown"]
    task_type_counts = {task_type: 0 for task_type in task_types}
    task_type_limits = {"craft_item": 20000, "kill_entity": 20000, "mine_block": 20000, "unknown": 4000}


    dataset = load_json_or_jsonl(args.train_path)
    processed_dataset = []
    for j in range(3):
        random.shuffle(dataset)
        idx = 0
        for idx, example in enumerate(dataset):
            processed_data = make_map_fn("train")(example, idx)

            if processed_data is not None and task_type_counts[processed_data["task_type"]] < task_type_limits[processed_data["task_type"]]:
                processed_dataset.append(processed_data)
                task_type_counts[processed_data["task_type"]] += 1

            if idx % 1000 == 0:
                print(f"Current task index: {idx}, Current task type counts: {task_type_counts}")

    df = pd.DataFrame(processed_dataset)
    df.to_parquet(os.path.join(args.output_path, f"train.parquet"))



    task_types = ["craft_item", "kill_entity", "mine_block", "unknown"]
    task_type_counts = {task_type: 0 for task_type in task_types}
    task_type_limits = {"craft_item": 2000, "kill_entity": 2000, "mine_block": 2000, "unknown": 400}
    processed_dataset = []
    for i in range(1):
        random.shuffle(dataset)
        for idx, example in enumerate(dataset):
            processed_data = make_map_fn("valid")(example, idx)

            if processed_data is not None and task_type_counts[processed_data["task_type"]] < task_type_limits[processed_data["task_type"]]:
                processed_dataset.append(processed_data)
                task_type_counts[processed_data["task_type"]] += 1

            if idx % 1000 == 0:
                print(f"Current task type counts: {task_type_counts}")

    df_valid = pd.DataFrame(processed_dataset)
    df_valid.to_parquet(os.path.join(args.output_path, f"valid.parquet"))