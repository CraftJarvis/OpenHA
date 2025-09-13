'''
Date: 2025-01-05 22:26:22
LastEditors: Muyao 2350076251@qq.com
LastEditTime: 2025-09-10 21:57:41
FilePath: /MineStudio/minestudio/simulator/callbacks/init_inventory.py
'''

from minestudio.simulator.callbacks.callback import MinecraftCallback
from typing import Union
import random
import json
import re
from pathlib import Path
from copy import deepcopy
from time import sleep
from rich import console,print
import uuid
from openagents.assets import MCD, TAG_INFO, ITEM_PREFIX, ITEM_PREFIX_LENGTH

EQUIP_SLOTS = {
    "mainhand": 0,
    "offhand": 40,
    "head": 39,
    "chest": 38,
    "legs": 37,
    "feet": 36,
}
REVERSE_EQUIP_SLOTS_MAP = {
    0:"mainhand",40:"offhand",39:"head",38:"chest",37:"legs",36:"feet"
}
MIN_SLOT_IDX = 0
MAX_INVENTORY_IDX = 35
MAX_SLOT_IDX = 40
SLOT_IDX2NAME = {v: k for k, v in EQUIP_SLOTS.items()}
MIN_ITEMS_NUM = 0
MAX_ITEMS_NUM = 64

INVENTORY_DISTRACTION_LEVEL = {"zero":[0],"one":[1],
                     "easy":range(3,7),"middle":range(7,16),"hard":range(16,36),
                     "normal":range(0,36)}
EQUIP_DISTRACTION_LEVEL = {
    "zero":[0],
    "normal":range(0,6),
}

class InitInventoryCallback(MinecraftCallback):
    
    def __init__(self, init_inventory:dict,inventory_distraction_level:Union[list,str]=[0],equip_distraction_level:Union[list,str]=[0],
                 forbidden_slots = []) -> None:
        """
        Examples:
            init_inventory = [{
                    slot: 0
                    type: "oak_planks"
                    quantity: 64  # supporting ">...",">=...","<...","<=...","==...","...",1
                }]
        """
        self.init_inventory = init_inventory
        self.forbidden_slots = set(forbidden_slots)
        self.inventory_distraction_level = INVENTORY_DISTRACTION_LEVEL.get(inventory_distraction_level,[0]) if isinstance(inventory_distraction_level,str) else inventory_distraction_level
        self.equip_distraction_level = EQUIP_DISTRACTION_LEVEL.get(equip_distraction_level,[0]) if isinstance(equip_distraction_level,str) else equip_distraction_level
        
        self.items_library = MCD.items_name
        self.items_names = list(MCD.items_name.keys())
        self.equipments_library = self._get_equipments_library()
        
    def after_reset(self, sim, obs, info):
        return self._set_inventory(sim, obs, info)
    
    def _set_inventory(self, sim, obs, info):
        chats = []
        visited_slots = set()
        uncertain_slots = [] 
        init_inventory = []
        
        for slot_info in self.init_inventory:
            slot = slot_info["slot"]
            if slot == "random":
                uncertain_slots.append(deepcopy(slot_info))
                continue
            visited_slots.add(int(slot))
            init_inventory.append(slot_info)
        unvisited_slots = set(range(MIN_SLOT_IDX, MAX_INVENTORY_IDX + 1)) - visited_slots
        unvisited_slots = unvisited_slots - self.forbidden_slots
        
        # settle uncertain slots
        for uncertain_slot in uncertain_slots:
            slot = int(random.choice(list(unvisited_slots)))
            unvisited_slots.remove(slot)
            visited_slots.add(slot)
            uncertain_slot["slot"] = slot
            init_inventory.append(uncertain_slot)
        
        # settle distraction at inventory slots
        init_inventory,visited_slots,unvisited_slots = self._sample_inventory(init_inventory,visited_slots,unvisited_slots)
        
        # settle distraction at equipments slots
        init_inventory,visited_slots = self._sample_equipments(init_inventory,visited_slots)
        
        # create init inventory
        self.slot_num = len(init_inventory)

        for item_dict in init_inventory:
            slot = item_dict["slot"]
            
            mc_slot =self._map_slot_number_to_cmd_slot(slot)
            item_type = item_dict["type"]
            if item_type not in self.items_names:
                item_type = random.choice(TAG_INFO.get(ITEM_PREFIX+item_type, [ITEM_PREFIX+item_type]))[ITEM_PREFIX_LENGTH:]
                item_dict["type"] = item_type
            assert item_type in self.items_names, f"wrong type: {item_type}"
            
            item_quantity = self._item_quantity_parser(item_dict["quantity"],int(self.items_library[item_type]["stackSize"]))
            
            chat = f"/replaceitem entity @p {mc_slot} minecraft:{item_type} {item_quantity}"
            if "metadata" in item_dict:
                chat += f" {item_dict['metadata']}"
                
            chats.append(chat)
        
        obs, reward, done, info = sim.env.execute_cmd("/gamerule sendCommandFeedback false")
        #obs, reward, done, info = sim.env.execute_cmd("/gamerule commandblockoutput false")
        for chat in chats:
            obs, reward, done, info = sim.env.execute_cmd(chat)
        obs, info = sim._wrap_obs_info(obs, info)
        
        
        # check whether set up
        init_flag = False
        
        kdx = 0
        inventory_infos = []
        for kdx in range(300):
            action = sim.env.noop_action()
            obs, reward, done, info = sim.env.step(action)
            inventory_infos.append({
                "init_inventory":init_inventory,
                "current_inventory":obs["inventory"]})
            init_flag,current_slot_num = self._check(obs)
            if init_flag or current_slot_num > self.slot_num:
                break
            if kdx%40==0:
                sleep(1)
            
        obs, info = self._clean_screen(sim,obs,info)
            
        if not init_flag:
            uuidx = str(uuid.uuid4())
            Path("logs").mkdir(parents=True,exist_ok=True)
            with open(f"logs/file_inventory_init_{uuidx}.json",mode="w") as file:
                json.dump(inventory_infos,file)
            messages = f"[red]can't set up init inventory[/red], need {self.slot_num}, has {current_slot_num} only, and has sample {kdx} steps. log at file_inventory_init_{uuidx}.json"
            console.Console().log(messages)
            message = info.get('message', {})
            message['InitInventoryCallback'] = messages
            info["message"] = message
            
        inventory_infos = []
            
        return obs, info
    
    def _clean_screen(self, sim, obs, info):
        obs, info = sim._wrap_obs_info(obs, info)
        return obs,info
    
    def _get_equipments_library(self):
        mc_equipments_file_path = Path(__file__).resolve().parents[2] / "assets" / "mc_equipments.1.16.json"
        if not mc_equipments_file_path.exists():
            try:
                from huggingface_hub import hf_hub_download
                hf_hub_download(repo_id="CraftJarvis/MinecraftResources", repo_type="dataset",filename="mc_equipments.1.16.json", local_dir=mc_equipments_file_path.parent)
                assert mc_equipments_file_path.exists(), f"File {mc_equipments_file_path} not found after download."
            except Exception as e:
                raise FileNotFoundError(f"Failed to download the file: {e}")

        with mc_equipments_file_path.open("r") as file:
            mc_equipments = json.load(file)
        
        return mc_equipments
    
    def _sample_inventory(self,init_inventory,visited_slots,unvisited_slots):
        distraction_num = min(random.choice(self.inventory_distraction_level),len(unvisited_slots))
        for _ in range(distraction_num):
            item_type = random.choice(self.items_names)
            slot = int(random.choice(list(unvisited_slots)))
            unvisited_slots.remove(slot)
            visited_slots.add(slot)
            init_inventory.append({
                "slot":slot,
                "type":item_type,
                "quantity":"random",
            })
        return init_inventory,visited_slots,unvisited_slots
    
    def _sample_equipments(self,init_inventory, visited_slots):
        unvisited_equipments = set(range(MAX_INVENTORY_IDX+1, MAX_SLOT_IDX+1)) - visited_slots
        sample_num = min(random.choice(self.equip_distraction_level),len(unvisited_equipments))
        for _ in range(sample_num):
            slot = random.choice(list(unvisited_equipments))
            equipment_name = REVERSE_EQUIP_SLOTS_MAP[slot]
            unvisited_equipments.remove(slot)
            visited_slots.add(slot)
            item_type = random.choice(self.equipments_library[equipment_name])
            if slot==40 and random.uniform(0, 1) > 0.25:
                item_type = random.choice(self.items_names)
            init_inventory.append({
                "slot":slot,
                "type":item_type,
                "quantity":1,
            })
        return init_inventory,visited_slots
         
    def _map_slot_number_to_cmd_slot(self,slot_number: Union[int,str]) -> str:
        slot_number = int(slot_number)
        assert MIN_SLOT_IDX <= slot_number <= MAX_SLOT_IDX, f"exceed slot index range:{slot_number}"
        if slot_number in {0, 40}:
            return f"weapon.{SLOT_IDX2NAME[slot_number]}"
        elif 36 <= slot_number <= 39:
            return f"armor.{SLOT_IDX2NAME[slot_number]}"
        elif 1 <= slot_number <= 8:
            return f"hotbar.{slot_number}"
        else:
            return f"inventory.{slot_number - 9}"

    def _item_quantity_parser(self,item_quantity: Union[int,str],max_items_num,one_p:float=0.7) -> int:
        """Function to parse item quantity from either an integer or a string command
        """
        
        if isinstance(item_quantity,str):
            
            candidate_nums=set(range(MIN_ITEMS_NUM+1, max_items_num + 1))
            
            if item_quantity == "random":
                one_flag = random.choices([True, False], weights=[one_p, 1 - one_p], k=1)[0]
                item_quantity = 1 if one_flag else random.choice(list(candidate_nums))
                return item_quantity
            
            item_quantity_commands = item_quantity.split(",")
        
            def apply_command(op, val):
                """Apply a command based on the operator and value provided in the string 
                """
                return {
                    '<': set(range(MIN_ITEMS_NUM,val)),
                    '<=': set(range(MIN_ITEMS_NUM,val+1)),
                    '>': set(range(val+1,max_items_num+1)),
                    '>=': set(range(val,max_items_num+1)),
                    '==': {val}
                }[op]
        
            for item_quantity_command in item_quantity_commands:
                match = re.search(r'([<>]=?|==)\s*(\d+)', item_quantity_command.strip()) #matching "<...", ">...", "<=...", ">=...", "==..."
                if match:
                    operator, number = match.groups()
                    number = int(number)
                    candidate_nums &= apply_command(operator,number)
            if candidate_nums:
                item_quantity = random.choice(list(candidate_nums))
            else:
                item_quantity = 1
                
            
        elif not isinstance(item_quantity, int):
            raise TypeError("Input must be an integer or a string representing conditions")

        return item_quantity
    
    def _check(self,obs):
        # check whether it set up the init inventory
        current_slot_num = 0
        for slot_dict in obs["inventory"].values():
            if slot_dict["type"] != "none":
                current_slot_num+=1
        for slot_name,slot_dict in obs["equipped_items"].items():
            if slot_name != "mainhand" and slot_dict["type"] != "air":
                current_slot_num += 1
        if current_slot_num == self.slot_num:
            return True,current_slot_num
        return False,current_slot_num
    
if __name__ == "__main__":

    import numpy as np
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import (
        SpeedTestCallback, 
        RecordCallback, 
        RewardsCallback, 
        TaskCallback,
        FastResetCallback
    )
    sim = MinecraftSim(
        action_type="env",
        callbacks=[
            SpeedTestCallback(50), 
            TaskCallback([
                {'name': 'craft', 'text': 'craft crafting_table'}, 
            ]),
            RecordCallback(record_path="./output", fps=30,record_actions=True,record_infos=True,record_origin_observation=True),
            RewardsCallback([{
                'event': 'craft_item', 
                'objects': ['crafting_table'], 
                'reward': 1.0, 
                'identity': 'craft crafting_table', 
                'max_reward_times': 1, 
            }]),
            FastResetCallback(
                biomes=['mountains'],
                random_tp_range=1000,
            ), 
            InitInventoryCallback([
                {"slot": 0,
                "type": "oak_planks",
                "quantity":1,},
                {"slot": 1,
                "type": "oak_planks",
                "quantity":">2",},
                {"slot": 2,
                "type": "oak_planks",
                "quantity":"<12,>10",},
                {"slot": "random",
                "type": "oak_planks",
                "quantity":"random",},
            ],inventory_distraction_level="normal",
            equip_distraction_level="normal")
        ]
    )
    obs, info = sim.reset()
    action = sim.noop_action()
    action["inventory"] = 1
    obs, reward, terminated, truncated, info = sim.step(action)
    for i in range(30):
        action = sim.noop_action()
        obs, reward, terminated, truncated, info = sim.step(action)
    sim.close()