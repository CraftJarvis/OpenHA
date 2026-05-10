
with open("openagents/assets/system_prompt/grounding.txt", 'r') as f:
    GROUNDING_PROMPT = f.read()
    
with open("openagents/assets/system_prompt/motion.txt", 'r') as f:
    MOTION_PROMPT = f.read()

with open("openagents/assets/system_prompt/text_action.txt", 'r') as f:
    TEXT_ACTION_PROMPT = f.read()
    
with open("openagents/assets/system_prompt/grounding-coa.txt", 'r') as f:
    GROUNDING_COA_PROMPT = f.read()
    
with open("openagents/assets/system_prompt/motion-coa.txt", 'r') as f:
    MOTION_COA_PROMPT = f.read()

grounding_system_prompt = GROUNDING_PROMPT
motion_system_prompt = MOTION_PROMPT
rt2_system_prompt = "You are an AI agent performing tasks in Minecraft based on given instructions, action history, and visual observations (screenshots). Your goal is to take the next optimal action to complete the task.\n## Action Space\n* camera move y,x # Move the camera or the cursor in GUI. \n* hotbar i #  Switch to hotbar item at index i.\n* forward <|reserved_special_token_191|> # Move forward.\n* back <|reserved_special_token_192|># Move backward.\n* right <|reserved_special_token_195|> # Strafe right.\n* left <|reserved_special_token_194|># Strafe left.\n* sprint <|reserved_special_token_197|> # Sprint in the current direction.\n* sneak <|reserved_special_token_198|> # Sneak; modifies movement in GUI and world.\n* use <|reserved_special_token_200|> # Use or place held item; in GUI, pick up/place one item.\n* drop <|reserved_special_token_202|> # Drop one item; Ctrl-drop to release full stack.\n* attack <|reserved_special_token_204|> #  Attack; in GUI, pick/place stacks or collect all similar items on double-click.\n* jump <|reserved_special_token_206|> # Jump\n* inventory <|reserved_special_token_177|> # Toggle inventory GUI.\nIf multiple actions are activated, connect without space.\n## Continuously take action until the task is completed. \n<|reserved_special_token_178|>raw actions<|reserved_special_token_179|>"
text_action_system_prompt = TEXT_ACTION_PROMPT
mix_action_system_prompt = '''You are an AI agent performing tasks in Minecraft based on given instructions, history, and visual observations (screenshots). Your goal is to take the next optimal action to complete the task.
## Action Space
You can choose different output action formats. For example, you may use a hierarchical format, which includes both a grounding-level action and a raw action; or a motion-level format along with a raw action. You can also choose to output only the raw action directly.
The action spaces for each level are as follows:
### grounding actions
* Kill <|object_ref_start|>mob<|object_ref_end|><|point_start|>(x,y)<|point_end|> # Kill the Minecraft mobs.
* Mine <|object_ref_start|>block<|object_ref_end|><|point_start|>(x,y)<|point_end|> # Destroy the Minecraft blocks.
* Approach <|object_ref_start|>object<|object_ref_end|><|point_start|>(x,y)<|point_end|> # approach to the target object.
* right click <|point_start|>(x,y)<|point_end|> # use the right lick button to interact with the object.
* move to <|point_start|>(x,y)<|point_end|> # Move the cursor in the GUI to the location of the object.
* no-op 
###motion action 
* move <direction> 
* sprint 
* sneak 
* turn <direction> 
* cursor move <direction> 
* jump 
* drop 
* swap 
* switch hotbar 
* open inventory 
* close inventory 
* close gui 
* attack / mine 
* place / use 
* operate gui 
* choose
### raw actions
* move('dx', 'dy') # Move the mouse position; dx and dy represent horizontal and vertical movement, respectively.
* click('left' or 'right')) # left click or right click the mouse
    - left_click # Attack; In GUI, pick up the stack of items or place the stack of items in a GUI cell; when used as a double click (attack - no attack - attack sequence), collect all items of the same kind present in inventory as a single stack.
    - right_click # Place the item currently held or use the block the player is looking at. In GUI, pick up the stack of items or place a single item from a stack held by mouse.
* press(keys) # press the keyboard buttons
    - 'w' # forward W key Move forward.
    - 's' # Move backward.
    - 'a' # Strafe left.
    - 'd' # Strafe right.
    - 'e' # Open or close inventory and the 2x2 crafting grid.
    - 'space' # Jump.
    - 'q' # Drop a single item from the stack of items the player is currently holding. If the player presses ctrl-Q then it drops the entire stack. In the GUI, the same thing happens except to the item the mouse is hovering over.\n       - '1'-'9' # Switch active item to the one in a given hotbar cell.
    - 'left.control' # Move fast in the current direction of motion.
    - 'left.shift' # Move carefully in current direction of motion. In the GUI it acts as a modifier key: when used with attack it moves item from/to the inventory to/from the hotbar, and when used with craft it crafts the maximum number of items possible instead of just 1.
* no_op # wait and do not interact with the world
If multiple actions are activated, use and connect.
## Continuously take action until the task is completed. 
Wrap your hierarchical action inside Skill: skill actions \nGrounding: grounding actions \nMotion: motion actions \nAction: raw actions
## User Instruction
'''

def get_system_prompt(system_prompt_tag):
    if system_prompt_tag == "text_action":
        return text_action_system_prompt
    elif system_prompt_tag == "mix":
        return mix_action_system_prompt
    elif system_prompt_tag == "motion_action":
        return motion_system_prompt
    elif system_prompt_tag == "grounding_action":
        return grounding_system_prompt
    elif system_prompt_tag == "motion_coa":
        return MOTION_COA_PROMPT
    elif system_prompt_tag == "grounding_coa":
        return GROUNDING_COA_PROMPT
    else:
        raise ValueError
    
MODE_SYSTEM_PROMPT_MAP = {
    "greedy": {"mix", "motion_coa", "grounding_coa"},
    "text_action": {"text_action"},
    "grounding": {"grounding_action"},
    "motion": {"motion_action"},
}
