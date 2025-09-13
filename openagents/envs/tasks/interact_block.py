import json, os, random 
from openagents.assets import INTERACT_BLOCK_SPAWN_FILE, EVENT_DESCRIPTION

with open(INTERACT_BLOCK_SPAWN_FILE, "r") as f:
    interact_block_spawn_dict = json.load(f)

def get_available_interact_block_tasks():
    return list(interact_block_spawn_dict.keys())

def gen_interact_block_task_config(task_name, difficulty:str="zero"):

    interact_block_spawn_file = interact_block_spawn_dict[task_name]
    

    if task_name not in EVENT_DESCRIPTION.keys():
        task_description = task_name.replace(":", " ").replace("_", " ")
    else:
        task_description = random.choice(EVENT_DESCRIPTION[task_name])
    
    seed_config = random.choice(interact_block_spawn_file['seeds'])
    seed_number = seed_config['seed']
    seed_position = seed_config['position']
    maximum_steps = 600
    blocks = interact_block_spawn_file["block"]
    commands = [f"/tp @s {' '.join([str(pos) for pos in seed_position])}"]
    for block in blocks:
        block_name = list(block.keys())[0]
        block_relative_position = list(block.values())[0]
        commands.append(f"/setblock ^{block_relative_position[0]} ^{block_relative_position[1]} ^{block_relative_position[2]} minecraft:{block_name}")
    #commands.append(interact_block_spawn_file["commands"])
    
    
    reward_cfg = [{
        "event": "custom",
        "max_reward_times": 1,
        "reward": 1.0,
        "objects": [task_name.split(":")[-1]],
        "identity": f"{task_name}",
    }]
    
    task_config = {
        "task_name": task_name,
        "task_description": task_description,
        "seed": seed_number,
        "init_actions": [],
        "maximum_steps": maximum_steps, 
        "callback": {
            "init_inventory": {
                "init_inventory": [], 
                "inventory_distraction_level": difficulty,
                "equip_distraction_level": difficulty,
                "forbidden_slots": [0],
            },
            "commands": commands,
        },
        "rewards": reward_cfg
    }
    return task_config