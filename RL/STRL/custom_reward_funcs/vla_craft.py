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
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    给定ground_truth中包含的一组reserved_special_token，
    返回solution_str中正确匹配这些token的比例得分。
    综合考虑按钮匹配与相机方向相似度。
    """

    #print(f"solution_str: {solution_str}")

    tokenizer = get_tokenizer()
    try:
        ground_action = parse_action(ground_truth)
        solution_action = tokenizer.decode(solution_str)[0]
    except Exception as e:
        print(f"[Error] in compute_score: {e}")
        return final_score #{"score": final_score, "button_score": button_score, "camera_score": camera_score}

    # 1. 按钮键（排除 camera）
    button_keys = [k for k in ground_action.keys() if k != 'camera']
    num_buttons = len(button_keys)
    
    # 2. 计算按钮匹配数量
    correct_buttons = sum(
        ground_action[k] == solution_action.get(k, not ground_action[k]) for k in button_keys
    )
    button_score = 2*correct_buttons / (num_buttons + len(solution_action) - 1) if num_buttons > 0 else 0.0

    # 3. 相机 loss → 相似度得分
    try:
        camera_gt = torch.tensor(ground_action["camera"], dtype=torch.float32)
        camera_pred = torch.tensor(solution_action["camera"], dtype=torch.float32)
        
        max_mse = 3.0  # 假设最大均方误差为0.1
        mse = torch.nn.functional.mse_loss(camera_gt, camera_pred).item()
        camera_score = max(0, 1 - mse / max_mse)  # 假设 max_mse 是一个合理的上限

        # camera_score = 1/math.exp(5*torch.nn.functional.mse_loss(camera_gt, camera_pred).item())
        # camera_score = max(0.0, min(1.0, camera_score))
    except Exception as e:
        print(f"[Warning] camera compare failed: {e}")
        return final_score #{"score": final_score, "button_score": button_score, "camera_score": camera_score}

    # 4. 综合得分（按钮为主，camera 为次）
    final_score = button_score  + camera_score 

    return final_score #{"score": final_score, "button_score": button_score, "camera_score": camera_score}

if __name__ == "__main__":
    # 测试用例
    ground_truth = "<|reserved_special_token_178|> <|reserved_special_token_179|> <|reserved_special_token_180|>"
    solution_str = "<|reserved_special_token_178|><|reserved_special_token_221|><|reserved_special_token_235|><|reserved_special_token_179|>"
    
    score = compute_score("test_source", solution_str, ground_truth)
    #print(f"Computed Score: {score}")