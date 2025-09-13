import json, os, random 
from openagents.assets import MINE_BLOCK_SPAWN_FILE, EVENT_DESCRIPTION

with open(MINE_BLOCK_SPAWN_FILE, "r") as f:
    mine_block_spawn_dict = json.load(f)

def get_available_mine_block_tasks():
    return list(mine_block_spawn_dict.keys())

def gen_mine_block_task_config(task_name, difficulty:str="zero"):
    if task_name not in EVENT_DESCRIPTION.keys():
        task_description = task_name.replace(":", " ").replace("_", " ")
    else:
        task_description = random.choice(EVENT_DESCRIPTION[task_name])
    
    seed_config = random.choice(mine_block_spawn_dict[task_name]['seeds'])
    seed_number = seed_config['seed']
    seed_position = seed_config['position']
    
    maximum_steps = 600
    commands = [f"/tp @s {' '.join([str(pos) for pos in seed_position])}"]
    
    init_inventory = []
    tool_list = mine_block_spawn_dict[task_name]['tool']
    if tool_list:
        if difficulty == "zero":
            tool_list = list(set(tool_list) - {""})
            assert tool_list
        tool = random.choice(tool_list)
        if tool and tool != "air":
            slot = 0 if difficulty != "hard" else random.choice(list(range(0,9)))
            init_inventory = [{'slot': slot, 'type': tool, 'quantity': 1}]
            
    reward_cfg = [
            {
                "event": "mine_block", 
                "identity": f"mine {task_name.split(':')[-1]} blocks", 
                "objects": [task_name.split(':')[-1]], 
                "reward": 1.0, 
                "max_reward_times": 1
            }
        ]
    
    task_config = {
        "task_name": task_name,
        "task_description": task_description,
        "seed": seed_number,
        "init_actions": [{"attack": 0, "back": 0, "forward": 0, "jump": 0, "left": 0, "right": 0, "sneak": 0, "sprint": 0, "use": 0, "drop": 0, "inventory": 0, "hotbar.1": 0, "hotbar.2": 0, "hotbar.3": 0, "hotbar.4": 0, "hotbar.5": 0, "hotbar.6": 0, "hotbar.7": 0, "hotbar.8": 0, "hotbar.9": 0, "camera": [0.0, 0.0]}]*2,
        "maximum_steps": maximum_steps, 
        "callback": {
            "init_inventory": {
                "init_inventory": init_inventory, 
                "inventory_distraction_level": difficulty,
                "equip_distraction_level": "normal",
            },
            "commands": commands,
        },
        "rewards": reward_cfg
    }
    return task_config

""" 
def gen_mine_block_task_config(task_name):
    if task_name == "random":
        task_name = random.choice(list(mine_block_spawn_dict.keys()))
        
    if task_name not in mine_block_spawn_dict.keys():
        raise KeyError(f"not supported task name: {task_name}")

    if task_name not in EVENT_DESCRIPTION.keys():
        task_description = task_name
    else:
        task_description = random.choice(EVENT_DESCRIPTION[task_name])
    
    seed_config = random.choice(mine_block_spawn_dict[task_name]['seeds'])
    seed_number = seed_config['seed']
    seed_position = seed_config['position']
    maximum_steps = 600
    commands = [f"/tp @s {' '.join([str(pos) for pos in seed_position])}"]
    init_inventory = [{'slot': 0, 'type': random.choice(mine_block_spawn_dict[task_name]['tool']), 'quantity': 1}] if len(mine_block_spawn_dict[task_name]['tool']) > 0 else []
    
    reward_cfg = [
            {
                "event": "mine_block", 
                "identity": f"mine {task_name.split(':')[-1]} blocks", 
                "objects": [task_name.split(':')[-1]], 
                "reward": 1.0, 
                "max_reward_times": 1
            }
        ]
    
    task_config = {
        "task_name": task_name,
        "task_description": task_description,
        "seed": seed_number,
        "maximum_steps": maximum_steps, 
        "callback": {
            "init_inventory": init_inventory,
            "commands": commands,
        },
        "rewards": reward_cfg
    }
    return task_config
"""


