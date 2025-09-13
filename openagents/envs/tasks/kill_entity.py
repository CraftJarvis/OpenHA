import json, os, random 
from openagents.assets import KILL_ENTITY_SPAWN_FILE, EVENT_DESCRIPTION

with open(KILL_ENTITY_SPAWN_FILE, "r") as f:
    kill_entity_spawn_dict = json.load(f)

def get_available_kill_entity_tasks():
    available_kill_entity_task_names = []
    for kill_entity_task_name,kill_entity_spawn_file in kill_entity_spawn_dict.items():
        spawn_label = kill_entity_spawn_file.get("label")
        if "normal" in spawn_label:
            available_kill_entity_task_names.append(kill_entity_task_name)
    return available_kill_entity_task_names

def gen_kill_entity_task_config(task_name, difficulty:str="zero"):

    kill_entity_spawn_file = kill_entity_spawn_dict[task_name]
    

    if task_name not in EVENT_DESCRIPTION.keys():
        task_description = task_name.replace(":", " ").replace("_", " ")
    else:
        task_description = random.choice(EVENT_DESCRIPTION[task_name])
    
    seed_config = random.choice(kill_entity_spawn_file['seeds'])
    seed_number = seed_config['seed']
    seed_position = seed_config['position']
    maximum_steps = 600
    commands = [f"/tp @s {' '.join([str(pos) for pos in seed_position])}"]
    init_inventory = []
    tool_list = kill_entity_spawn_file['tool']
    if tool_list:
        if difficulty == "zero":
            tool_list = list(set(tool_list) - {""})
            assert tool_list
        tool = random.choice(tool_list)
        if tool and tool != "air":
            slot = 0 if difficulty != "hard" else random.choice(list(range(0,9)))
            init_inventory = [{'slot': slot, 'type': tool, 'quantity': 1}]
    
    mobs=[{
        "name": task_name.split(":")[-1],
        "number": 1,
        "range_x":  [-1, 1],
        "range_z":  [2, 7],
    }]
    commands.append(kill_entity_spawn_file["commands"])
    
    reward_cfg = [{
        "event": "kill_entity",
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
                "init_inventory": init_inventory, 
                "inventory_distraction_level": difficulty,
                "equip_distraction_level": "normal",
            },
            "commands": commands,
            "mobs": mobs,
        },
        "rewards": reward_cfg
    }
    return task_config