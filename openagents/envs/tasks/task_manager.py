import random
import re
from typing import Union, Dict, Any 

from openagents.envs.tasks.mine_block import get_available_mine_block_tasks, gen_mine_block_task_config
from openagents.envs.tasks.craft_item import get_available_craft_item_tasks, gen_craft_item_task_config
from openagents.envs.tasks.kill_entity import get_available_kill_entity_tasks, gen_kill_entity_task_config
from openagents.envs.tasks.smelt_item import get_available_smelt_item_tasks,gen_smelt_item_task_config
from openagents.envs.tasks.interact_block import get_available_interact_block_tasks,gen_interact_block_task_config
from openagents.utils import file_op

def get_available_tasks():
    all_tasks = []
    all_tasks.extend(get_available_mine_block_tasks())
    all_tasks.extend(get_available_kill_entity_tasks())
    all_tasks.extend(get_available_smelt_item_tasks()) 
    all_tasks.extend(get_available_craft_item_tasks())
    all_tasks.extend(get_available_interact_block_tasks())
    return all_tasks

def gen_task_config(task_name_pattern:str, difficulty:str="zero"):
    task_config = None
        
    if task_name_pattern in get_available_craft_item_tasks():
        task_config = gen_craft_item_task_config(task_name_pattern,difficulty=difficulty)
    elif task_name_pattern in get_available_smelt_item_tasks():
        task_config = gen_smelt_item_task_config(task_name_pattern,difficulty=difficulty)
    elif "smelt_item" in task_name_pattern:
        task_name_pattern.replace("smelt_item","craft_item")
        task_config = gen_smelt_item_task_config(task_name_pattern,difficulty=difficulty)
    elif "kill_entity" in task_name_pattern:
        task_config = gen_kill_entity_task_config(task_name_pattern,difficulty=difficulty)
    elif 'mine_block' in task_name_pattern:
        task_config = gen_mine_block_task_config(task_name_pattern,difficulty=difficulty)
    elif "interact_with" in task_name_pattern:
        task_config = gen_interact_block_task_config(task_name_pattern,difficulty=difficulty)
    
    if task_config is None:
        raise NameError("wrong task_name_pattern: ", task_name_pattern)
    return task_config

def choose_available_task(task_name_pattern:Union[dict, str], difficulty:str="zero") -> Dict[str, Any]:
    
    all_avaliable_tasks = get_available_tasks()
    #print(all_avaliable_tasks)
    
    assert isinstance(task_name_pattern, str), "task_name_pattern should be a string"
    
    # 如果是random,随机选一个
    if task_name_pattern == "random":
        task_name_pattern = random.choice(all_avaliable_tasks)
    else:
        task_name_patterns = task_name_pattern.split(",")
        task_name_pattern = random.choice([p.strip() for p in task_name_patterns])    
        
        matched_tasks = []

        regex = "^" + re.escape(task_name_pattern).replace(r"\*", ".*") + "$"
        compiled = re.compile(regex)

        # 匹配所有任务名
        for task_name in all_avaliable_tasks:
            if compiled.match(task_name):
                matched_tasks.append(task_name)

        if "smelt_item*" == task_name_pattern:
            matched_tasks =  get_available_smelt_item_tasks()
            matched_tasks = [matched_task.replace("craft_item","smelt_item") for matched_task in matched_tasks]

        if not matched_tasks:
            raise ValueError(f"No task matched pattern(s): {task_name_pattern}")
        
        task_name_pattern = random.choice(matched_tasks)
        
    return gen_task_config(task_name_pattern,difficulty=difficulty)

if __name__ == "__main__":
    task_name = "kill_entity:*"
    task_config = choose_available_task(task_name)
    print(task_config)

