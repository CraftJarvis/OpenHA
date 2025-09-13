from pathlib import Path
import json
import numpy as np
import minecraft_data
import csv

FILE_DIR = Path(__file__).parent

# 一些资源 folder
RECIPES_DIR = FILE_DIR / "recipes"

# 种子文件
MINE_BLOCK_SPAWN_FILE = FILE_DIR / "spawns" / "mine_block.json"
KILL_ENTITY_SPAWN_FILE = FILE_DIR / "spawns" / "kill_entity.json"
INTERACT_BLOCK_SPAWN_FILE = FILE_DIR / "spawns" / "interact_block.json"
CRAFT_ITEM_SPAWN_FILE = FILE_DIR / "spawns" / "craft_item.json"
SMELT_ITEM_SPAWN_FILE = FILE_DIR / "spawns" / "smelt_item.json"
LONG_HORIZON_SPAWN_FILE = FILE_DIR / "spawns" / "long_horizon.json"

# 特殊的文件
CONSTANT_FILE = FILE_DIR / "constants.json"
TAG_ITEMS_FILE = FILE_DIR / "tag_items.json"
with open(TAG_ITEMS_FILE) as file:
    TAG_INFO = json.load(file)
EVENT_DESCRIPTION_FILE_PATH = FILE_DIR / "event_description.json"
with open(EVENT_DESCRIPTION_FILE_PATH, "r") as f:
    EVENT_DESCRIPTION = json.load(f)
MCD = minecraft_data("1.16")
INSTRUCTION_FILE_PATH = FILE_DIR / "instructions.json"
with open(INSTRUCTION_FILE_PATH, "r") as f:
    INSTRUCTION_FILE = json.load(f)
FONT_FILE = FILE_DIR / "fonts" / "SimHei.ttf"
        
OPEN_GUI_ACTIONS_FILE = FILE_DIR / "open_gui_actions.json"


ENV_NULL_ACTION = {
    'hotbar.1': 0, 'hotbar.2': 0, 'hotbar.3': 0, 'hotbar.4': 0, 'hotbar.5': 0,
    'hotbar.6': 0, 'hotbar.7': 0, 'hotbar.8': 0, 'hotbar.9': 0,
    'forward': 0, 'back': 0, 'left': 0, 'right': 0, 'sprint': 0, 'sneak': 0,
    'use': 0, 'drop': 0, 'attack': 0, 'jump': 0, 'inventory': 0,
    'camera': np.array([0.0, 0.0])
}
BASE_WIDTH, BASE_HEIGHT = 640, 360
POLICY_RESOLUTION = (224,224)
ITEM_PREFIX = "minecraft:"
ITEM_PREFIX_LENGTH = len(ITEM_PREFIX)
