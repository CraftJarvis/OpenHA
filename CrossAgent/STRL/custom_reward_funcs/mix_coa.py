import ray
import re
import math
from openagents.agents.utils.action_mapping import TextActionTokenizer
import torch
from minestudio.simulator.entry import MinecraftSim
import random
import numpy as np

def parse_action(action_str):
    if "move_to" in action_str:
        # 提取 move_to 的坐标
        coords = re.findall(r'move_to\(([^)]+)\)', action_str)
        if coords:
            x, y = map(float, coords[0].split(','))
            return {'hotbar.1': 0, 'hotbar.2': 0, 'hotbar.3': 0, 'hotbar.4': 0, 'hotbar.5': 0, 'hotbar.6': 0, 'hotbar.7': 0, 'hotbar.8': 0, 'hotbar.9': 0, 'forward': 0, 'back': 0, 'left': 0, 'right': 0, 'sprint': 0, 'sneak': 0, 'use': 0, 'drop': 0, 'attack': 0, 'jump': 0, 'inventory': 0, 'camera': np.array([x, y])}
    elif "click" in action_str:
        return {'hotbar.1': 0, 'hotbar.2': 0, 'hotbar.3': 0, 'hotbar.4': 0, 'hotbar.5': 0, 'hotbar.6': 0, 'hotbar.7': 0, 'hotbar.8': 0, 'hotbar.9': 0, 'forward': 0, 'back': 0, 'left': 0, 'right': 0, 'sprint': 0, 'sneak': 0, 'use': 0, 'drop': 0, 'attack': 1, 'jump': 0, 'inventory': 0, 'camera': np.array([0.0, 0.0])}


tokenizer = None

def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer = TextActionTokenizer()
    return tokenizer

import torch
import torch.nn.functional as F

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    给定ground_truth中包含的一组reserved_special_token，
    返回solution_str中正确匹配这些token的比例得分。
    综合考虑按钮匹配与相机方向相似度。
    """
    
    tokenizer = get_tokenizer()
    try:
        gt = tokenizer.decode(ground_truth)[0]
        sa = tokenizer.decode(solution_str)[0]

        if random.random() < 0.001:
            print(f"[DEBUG] ground_action: {gt}\n[DEBUG] solution_action: {sa}")

    except Exception as e:
        print(f"[Error] in compute_score: {e}")
        return 0
    reward = 1
    gt["camera"] = np.round(gt["camera"], 0)
    sa["camera"] = np.round(sa["camera"], 0)

    # no_op = True
    # for key in gt:
    #     if key != "camera" and gt[key] != 0:
    #         no_op = False
    #         break
    #     if key == "camera" and not np.array_equal(gt["camera"], np.array([0.0, 0.0])):
    #         no_op = False
    #         break
    
    # if no_op:
    #     breakpoint()




    for key in gt:
        if key != "camera":
            if gt[key] != sa.get(key, None):
                reward = 0
                break
        if key == "camera":
            if not np.array_equal(gt["camera"], sa.get("camera", None)):
                reward = 0
                break


    extra_info = {
        "Motion_num": 0,
        "Motion_correct_num": 0,
        "Grounding_num": 0,
        "Grounding_correct_num": 0,
        "Action_num": 0,
        "Action_correct_num": 0
    }

    if "Motion:" in solution_str:
        extra_info["Motion_num"] += 1
        if reward == 1:
            extra_info["Motion_correct_num"] += 1
    if "Grounding:" in solution_str:
        extra_info["Grounding_num"] += 1
        if reward == 1:
            extra_info["Grounding_correct_num"] += 1
    if "Action:" in solution_str:
        extra_info["Action_num"] += 1
        if reward == 1:
            extra_info["Action_correct_num"] += 1

    return {"score": reward, **extra_info}

if __name__ == "__main__":
    # 测试用例
    tokenizer = get_tokenizer()
    with open("/DATA/hkc/workspace/toyspace/1.jsonl", "r") as f:
        lines = f.readlines()
    for line in lines:
        import json
        data = json.loads(line)
        gt = tokenizer.decode(data["reward_model"]['ground_truth'])[0]
        gt["camera"] = np.round(gt["camera"], 0)
        print(gt["camera"])

    #print(f"Computed Score: {score}")