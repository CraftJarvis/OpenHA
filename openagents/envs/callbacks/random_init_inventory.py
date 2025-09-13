import minecraft_data # https://github.com/SpockBotMC/python-minecraft-data  Provide easy access to minecraft-data in python
from minestudio.simulator.callbacks.callback import MinecraftCallback
from typing import Union
import random
import json
import re
from pathlib import Path
from copy import deepcopy
from time import sleep
from rich import console

EQUIP_SLOTS = {
    "mainhand": 0,
    "offhand": 40,
    "head": 39,
    "chest": 38,
    "legs": 37,
    "feet": 36,
}
MIN_SLOT_IDX = 0
MAX_INVENTORY_IDX = 35
MAX_SLOT_IDX = 40
SLOT_IDX2NAME = {v: k for k, v in EQUIP_SLOTS.items()}
MIN_ITEMS_NUM = 0
MAX_ITEMS_NUM = 64

DISTRACTION_LEVEL = {"zero":[0],"one":[1],
                     "easy":range(3,7),"middle":range(7,16),"hard":range(16,35),
                     "normal":range(0,35)}



class RandomInitInventoryCallback(MinecraftCallback):
    
    def __init__(self, recipe ,distraction_level:Union[list,str]=[0], item_counts=None, random_item_prob=0) -> None:
        """
        Examples:
            init_inventory = [{
                    slot: 0
                    type: "oak_planks"
                    quantity: 64  # supporting ">...",">=...","<...","<=...","==...","...",1
                }]
        """
        self.goal_recipe = recipe
        self.item_counts = item_counts
        self.random_item_prob = random_item_prob
        if item_counts is not None:
            self.total_item_counts = sum([self.item_counts[key] for key in self.item_counts])
        #self.goal_recipe = recipes[self.recipe_id]
        #self.init_inventory = self.build_inventory(self.goal_recipe)
        self.distraction_level = DISTRACTION_LEVEL.get(distraction_level,[0]) if isinstance(distraction_level,str) else distraction_level
        
        mcd = minecraft_data("1.16")
        self.items_library = mcd.items_name
        self.items_names = list(mcd.items_name.keys())

    def build_inventory(self,recipe):
        init_inventory = []
        used_slots = set()
        used_slots.clear()
        for item in recipe["materials"]:
            item_type = item["type"]
            #quantity = item["max_stack_num"]
            if item["max_stack_num"] == 0:
                item["max_stack_num"] = random.choice([1,16,64])
            quantity = item["max_stack_num"]
            while item["num"] > 0:
                count = 0
                while True:
                    slot = random.randint(MIN_SLOT_IDX, MAX_INVENTORY_IDX)
                    if slot not in used_slots:
                        used_slots.add(slot)
                        break
                    count+=1
                    if count > 100:
                        import pdb;pdb.set_trace()
                init_inventory.append({
                    "slot":slot,
                    "type":item_type,
                    "quantity":quantity,
                })
                item["num"] -= quantity

        if self.random_item_prob>0:
            for slot in range(MIN_SLOT_IDX, MAX_INVENTORY_IDX + 1):
                if slot in used_slots:
                    continue
                if random.random() < self.random_item_prob:
                    keys = list(self.item_counts.keys())
                    weights = list(self.item_counts.values())
                    item_type = random.choices(keys, weights=weights, k=1)[0]
                    #print("random item:",item_type)
                    additional_quantity = random.choice([1,16,64])
                    init_inventory.append({
                        "slot":slot,
                        "type":item_type,
                        "quantity":additional_quantity,
                    })
                    used_slots.add(slot)


        return init_inventory
        
    def after_reset(self, sim, obs, info):
        chats = []
        visited_slots = set()
        uncertain_slots = [] 
        init_inventory = []
        
        self.init_inventory = self.build_inventory(self.goal_recipe)

        for slot_info in self.init_inventory:
            slot = slot_info["slot"]
            if slot == "random":
                uncertain_slots.append(deepcopy(slot_info))
                continue
            visited_slots.add(int(slot))
            init_inventory.append(slot_info)
        unvisited_slots = set(range(MIN_SLOT_IDX, MAX_INVENTORY_IDX + 1)) - visited_slots
        
        # settle uncertain slots
        for uncertain_slot in uncertain_slots:
            slot = int(random.choice(list(unvisited_slots)))
            unvisited_slots.remove(slot)
            uncertain_slot["slot"] = slot
            init_inventory.append(uncertain_slot)
        
        # settle distraction slot
        distraction_num = min(random.choice(self.distraction_level),len(unvisited_slots))
        for _ in range(distraction_num):
            item_type = random.choice(self.items_names)
            slot = int(random.choice(list(unvisited_slots)))
            unvisited_slots.remove(slot)
            init_inventory.append({
                "slot":slot,
                "type":item_type,
                "quantity":"random",
            })
        self.slot_num = len(init_inventory)
        for item_dict in init_inventory:
            slot = item_dict["slot"]
            mc_slot =self._map_slot_number_to_cmd_slot(slot)
            item_type = item_dict["type"]
            if item_type not in self.items_names:
                continue
            assert item_type in self.items_names
            item_quantity = self._item_quantity_parser(item_dict["quantity"],int(self.items_library[item_type]["stackSize"]))
            chat = f"/replaceitem entity @p {mc_slot} minecraft:{item_type} {item_quantity}"
            if "metadata" in item_dict:
                chat += f" {item_dict['metadata']}"
            chats.append(chat)
        for chat in chats:
            obs, reward, done, info = sim.env.execute_cmd(chat)
        obs, info = sim._wrap_obs_info(obs, info)
        init_flag = False
        
        for _ in range(self.slot_num*2):
            action = sim.env.noop_action()
            obs, reward, done, info = sim.env.step(action)
            init_flag = self._check(obs)
            if init_flag:
                break
        if not init_flag:
            console.Console().log("[red]can't set up init inventory[/red]")
        return obs, info
    
    
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
            
            candidate_nums=set(range(MIN_ITEMS_NUM, max_items_num + 1))
            
            if item_quantity == "random":
                one_flag = random.choices([True, False], weights=[one_p, 1 - one_p], k=1)[0]
                if one_flag:
                    return 1
                else:
                    return random.choice(list(candidate_nums))
            
            
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
            
        elif not isinstance(item_quantity, int):
            raise TypeError("Input must be an integer or a string representing conditions")

        return item_quantity
    
    def _check(self,obs):
        "check whether it set up the init inventory"
        current_slot_num = 0
        for slot_dict in obs["inventory"].values():
            if slot_dict["type"] != "none":
                current_slot_num+=1
        if current_slot_num >= self.slot_num:
            return True
        return False
  