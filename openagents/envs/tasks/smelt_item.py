import json, os, random 
from openagents.assets import SMELT_ITEM_SPAWN_FILE, EVENT_DESCRIPTION, OPEN_GUI_ACTIONS_FILE

with open(SMELT_ITEM_SPAWN_FILE, "r") as f:
    smelt_item_spawn_dict = json.load(f)

with open(OPEN_GUI_ACTIONS_FILE, "r") as f:
    open_gui_actions = json.load(f)

def get_available_smelt_item_tasks():
    return list(smelt_item_spawn_dict.keys())

def gen_smelt_item_task_config(task_name:str,difficulty:str="zero"):
    if task_name not in EVENT_DESCRIPTION.keys():
        task_description = task_name.replace(":", " ").replace("_", " ")
    else:
        task_description = random.choice(EVENT_DESCRIPTION[task_name])
    seed_config = random.choice(smelt_item_spawn_dict[task_name]['seeds'])
    seed_number = seed_config['seed']
    seed_position = seed_config['position']
    goal_object = smelt_item_spawn_dict[task_name]["goal"]
    maximum_steps = 600
    commands = [f"/tp @s {' '.join([str(pos) for pos in seed_position])}"]
    init_actions = []
    if smelt_item_spawn_dict[task_name]["need_furnace"]:
        init_actions = open_gui_actions["furnace"][f"{seed_number}_{seed_position[0]}_{seed_position[1]}_{seed_position[2]}"]
    forbidden_slots = [0]
    init_inventory = smelt_item_spawn_dict[task_name].get('init_inventory', [])
    
    reward_cfg = [
            {
                "event": "craft_item", 
                "identity": f"craft {goal_object}", 
                "objects": [goal_object], 
                "reward": 1.0, 
                "max_reward_times": 1
            }
        ]
    
    task_config = {
        "task_name": task_name,
        "task_description": task_description,
        "seed": seed_number,
        "init_actions": init_actions,
        "maximum_steps": maximum_steps, 
        "callback": {
            "init_inventory": {
                "init_inventory": init_inventory, 
                "inventory_distraction_level": difficulty,
                "equip_distraction_level": difficulty,
                "forbidden_slots": forbidden_slots,
            },
            
            "commands": commands,
        },
        "rewards": reward_cfg
    }
    return task_config