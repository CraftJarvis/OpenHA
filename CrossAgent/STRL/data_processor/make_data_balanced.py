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


with open("OpenHA/openagents/assets/instructions.json", "r") as f:
    instructions = json.load(f)

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
    return solution_str

    
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


def parse_history2action(example, history_length=0):
    assert 2*(history_length+1) <= len(example["conversations"]), "Conversations length mismatch."
    example_new = deepcopy(example)
    example_new["conversations"] = [example["conversations"][0]]
    example_new["conversations"].extend(example["conversations"][len(example["conversations"])-2*history_length-1:])  # 保留最后两轮对话
    example_new["image"] = example["image"][-history_length-1:]  # 保留最后一张图片

    raw_actions = []
    assert len(example["conversations"])%2 ==0, "Conversations length should be even."
    for i in range(len(example["conversations"])//2):
        raw_actions.append(tokenizer.decode(example["conversations"][2*i+1]["content"][0]["text"])[0])

    return example_new
    # chosen_prob = 1/ (2**count_suffix(raw_actions))
    # if random.random() < chosen_prob:
    #     return example_new
    # else:
    #     return None


def compute_camera_toward(action):
    camera = action["camera"]
    if camera[0]==0 and camera[1]==0:
        return "camera_stop"
    elif camera[0] > 0 and abs(camera[0]) > abs(camera[1]):
        return "camera_right"
    elif camera[0] < 0 and abs(camera[0]) > abs(camera[1]):
        return "camera_left"
    elif camera[1] > 0 and abs(camera[1]) > abs(camera[0]):
        return "camera_down"
    elif camera[1] < 0 and abs(camera[1]) > abs(camera[0]):
        return "camera_up"
    elif camera[1] > 0:
        return "camera_down"
    elif camera[1] < 0:
        return "camera_up"
    else:
        import pdb
        pdb.set_trace()

def make_map_fn(split, data_source="coa_craft", system_prompt = mix_action_system_prompt, history_length = 0):
    def map_fn(example, idx, action_nums_ogn = {}, action_limit_nums = {}, action_requirement = None, craftonly = False):

        action_nums = deepcopy(action_nums_ogn)
        assert history_length == 0
        if len(example['image'])==0: 
            return None

        ogn_example = deepcopy(example)
        example = parse_history2action(example, history_length=history_length)
        if example is None:
            return None

        if idx%100 ==0:
            print(f"Processing example {idx} in split {split}")
            print(action_nums)
            print(f"Action requirement: {action_requirement}")
        try:
            user_query = example["conversations"][0]["content"][0]["text"]
        except:
            import pdb
            pdb.set_trace()
    
        prompt = f"{system_prompt}\n<image>\n{user_query}\n"

        if "<image>" in system_prompt or "<image>" in user_query:
            return None

        # 拼接 assistant 响应



        assistant_parts = []
        for item in example["conversations"][1]["content"]:
            if item["type"] == "text":
                assistant_parts.append(item["text"])
            elif item["type"] == "point":
                if "object" in item:
                    assistant_parts.append(f"<|object_ref_start|>{item['object']}<|object_ref_end|>")
                assistant_parts.append(f"<|point_start|>{item['point']}<|point_end|>")
        response = " ".join(assistant_parts)

        solution_action = tokenizer.decode(response)[0]
        solution = extract_solution_vla(response)

        if craftonly:
            for key in ["forward", "back", "left", "right", "use", "jump", "attack", "sneak", "inventory"]:
                if key in solution_action and solution_action[key] == 1:
                    return None
                    

        camera_toward = compute_camera_toward(solution_action)
        if action_requirement is not None:
            if "camera" in action_requirement:
                if camera_toward not in action_requirement:
                    return None
            elif solution_action[action_requirement] == 0:
                    return None
            else:
                pass

        for key in action_nums.keys():
            if "camera" not in key and solution_action[key] == 1:
                if key in action_limit_nums and action_nums[key] == action_limit_nums[key]:
                    return None
                else:
                    action_nums[key] += 1

            if "camera" in key and camera_toward in key:
                if key in action_limit_nums and action_nums[key] == action_limit_nums[key]:
                    return None
                else:
                    action_nums[key] += 1

            


        assert len(example["image"]) == 1, "Each example should contain exactly one image."
        instruction = example["conversations"][0]["content"][0]["text"]
        
        data_source = "notknown"
        for key in instructions.keys():
            if instruction in instructions[key]:
                data_source = key.split(":")[0]
                print("data_source:", data_source)
        if data_source == "notknown":
            words = instruction.split(" ")
            try:
                data_source = words[0]+"_"+words[1]
            except: 
                return None
        if data_source == "notknown":
            return None

        if "kill" not in data_source:
            return None
        return {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": prompt
            }],
            "images": [{"image": example["image"][0]["image_path"]}],
            "ability": "coa_craft",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution if solution else ""
            },
            "extra_info": {
                "split": split,
                "index": idx
            }
        }, action_nums
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
    parser.add_argument("--train_path", type=str, default="/share/hkc/datasets/mc-mix-action-0816-train.jsonl")
    parser.add_argument("--valid_path", type=str, default=None) #"/share/hkc/craft_cold_start/utils/balanced-valid.json")
    parser.add_argument("--output_path", type=str, default="OpenHA/rl/datasets/mc-mix-action_balanced_killentity")
    args = parser.parse_args()

    action_nums = {
        "forward": 0,
        "back": 0,
        "left": 0,
        "right": 0,
        "use": 0,
        "jump": 0,
        "attack": 0,
        "sneak": 0,
        "inventory": 0,
        "camera_stop": 0,
        "camera_left": 0,
        "camera_right": 0,
        "camera_up": 0,
        "camera_down": 0,
    }
    action_limit_nums = {
        "forward": 2000,
        "back": 2000,
        "left": 2000,
        "right": 2000,
        "use": 2000,
        "jump": 2000,
        "attack": 2000,
        "camera_stop": 2000
        # "sneak": 5000,
        # "inventory": 5000, don't need?
        # "camera_stop": 2000,
        # "camera_left": 2000,
        # "camera_right": 2000,
        # "camera_up": 2000,
        # "camera_down": 2000,
    }

    action_list = list(action_limit_nums.keys())
    action_requirement_idx = 0
    
    os.makedirs(args.output_path, exist_ok=True)
    for split, path in tqdm([("train", args.train_path), ("valid", args.valid_path)]):
        if split == "valid":
            processed_dataset = random.sample(processed_dataset, 2048) if len(dataset) > 2048 else dataset
        else:
            dataset = load_json_or_jsonl(path)
            random.shuffle(dataset)
            processed_dataset = []
            
            for j in range(2):
                for idx, example in enumerate(dataset):
                    action_requirement_idx = random.randint(0, len(action_list)-1)
                    processed_data_action_nums = make_map_fn(split)(example, idx, action_nums_ogn=action_nums, action_limit_nums=action_limit_nums, action_requirement=action_list[action_requirement_idx])
                    if processed_data_action_nums is not None:
                        processed_data, action_nums = processed_data_action_nums
                        processed_dataset.append(processed_data)
                    
                    end = True
                    for key, val in action_limit_nums.items():
                        if action_nums[key] < val:
                            end = False
                            break
                    if end:
                        print(f"Reached action limit for split {split}. Ending processing.")
                        break

            total_num = len(processed_dataset)
            additional_num = 0
            for idx, example in enumerate(dataset):
                processed_data_action_nums = make_map_fn(split)(example, idx, action_nums_ogn=action_nums, action_limit_nums=action_limit_nums, action_requirement=random.choice(["camera_up", "camera_down", "camera_left", "camera_right"]), craftonly=True)
                if processed_data_action_nums is not None:
                    processed_data, action_nums = processed_data_action_nums
                    processed_dataset.append(processed_data)
                    additional_num += 1
                if additional_num*4 > total_num:
                    break

            random.shuffle(processed_dataset)



        print(f"After processing split {split}, action counts: {action_nums}")
        df = pd.DataFrame(processed_dataset)
        df.to_parquet(os.path.join(args.output_path, f"{split}.parquet"))
