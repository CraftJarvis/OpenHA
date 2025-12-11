
from openagents.assets import INSTRUCTION_FILE_PATH
import json
from tqdm import tqdm

with open(INSTRUCTION_FILE_PATH, "r") as f:
    INSTRUCTIONS = f.read()
    INSTRUCTIONS = json.loads(INSTRUCTIONS)

    
def get_task_type(ins):
    for key in INSTRUCTIONS:
        if ins in INSTRUCTIONS[key]:
            if "craft_item" in key.lower():
                return "craft_item"
            elif "kill_entity" in key.lower():
                return "kill_entity"
            elif "mine_block" in key.lower():
                return "mine_block"
            else:
                continue
    return "unknown"

def get_task_from_ins(ins):
    for key in INSTRUCTIONS:
        if ins in INSTRUCTIONS[key]:
            return key
    
    if len(key.split(" ")) == 4:
        return key[0]+ "_" + key[1] + ":" + key[2]+ "_" + key[3]


    return "unknown"

def get_instruction(messages):
    instruction = messages[0]["content"][0]["text"]
    if instruction.endswith("User Instruction\n"):
        instruction = messages[0]["content"][1]["text"]
    return instruction


if __name__ == "__main__":
    print(get_task_type("Mine a cornflower from the ground."))