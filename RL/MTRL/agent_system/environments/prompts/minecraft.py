# --------------------- Minecraft_COA --------------------- #
WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment. 
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are: 
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

MINECRAFT_COA_TEMPLATE_NO_HIS = """
You are an AI agent performing tasks in Minecraft based on given instructions, history, and visual observations (screenshots). Your goal is to take the next optimal action to complete the task.
## Action Space
Your action space is hierarchical, including grounding-level and raw-action. You need to output a hierarchical action chain until the final raw-action is output. The action spaces at each level are as follows:
### skill actions
These are high-level skills that guide the agent toward completing complex tasks through multiple intermediate steps.
### grounding actions
* Kill <|object_ref_start|>mob<|object_ref_end|><|point_start|>(x,y)<|point_end|> # Kill the Minecraft mobs. Note: The target of Action Kill needs to appear in the current observation image. 
* Mine <|object_ref_start|>block<|object_ref_end|><|point_start|>(x,y)<|point_end|> # Destroy the Minecraft blocks. Note: The target of Action Mine needs to appear in the current observation image.
* Approach <|object_ref_start|>object<|object_ref_end|><|point_start|>(x,y)<|point_end|> # approach to the target object.
* right click <|point_start|>(x,y)<|point_end|> # use the right lick button to interact with the object.
* move to <|point_start|>(x,y)<|point_end|> # Move the cursor in the GUI to the location of the object.
* no-op # wait and do not interact with the world
If multiple actions are activated, use ` and ` connect. 
### raw actions
* camera move y,x # Move the camera or the cursor in GUI. 
* hotbar i #  Switch to hotbar item at index i.
* forward <|reserved_special_token_191|> # Move forward.
* back <|reserved_special_token_192|># Move backward.
* right <|reserved_special_token_195|> # Strafe right.
* left <|reserved_special_token_194|># Strafe left.
* sprint <|reserved_special_token_197|> # Sprint in the current direction.
* sneak <|reserved_special_token_198|> # Sneak; modifies movement in GUI and world.
* use <|reserved_special_token_200|> # Use or place held item; in GUI, pick up/place one item.
* drop <|reserved_special_token_202|> # Drop one item; Ctrl-drop to release full stack.
* attack <|reserved_special_token_204|> #  Attack; in GUI, pick/place stacks or collect all similar items on double-click.
* jump <|reserved_special_token_206|> # Jump
* inventory <|reserved_special_token_177|> # Toggle inventory GUI.
If multiple actions are activated, connect without space.
## Continuously take action until the task is completed. 
Wrap your hierarchical action inside <skill>skill actions</skill>, <grounding>grounding actions</grounding>, <|reserved_special_token_178|>raw actions<|reserved_special_token_179|>
"""


MINECRAFT_VLA_TEMPLATE_NO_HIS = """
You are an AI agent performing tasks in Minecraft based on given instructions, history, and visual observations (screenshots). Your goal is to take the next optimal action to complete the task.
## Action Space
Your action space is hierarchical, including grounding-level, motion-level and raw-action. You need to output a hierarchical action chain until the final raw-action is output. The action spaces at each level are as follows:
### skill actions
These are high-level skills that guide the agent toward completing complex tasks through multiple intermediate steps.
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
    - 'q' # Drop a single item from the stack of items the player is currently holding. If the player presses ctrl-Q then it drops the entire stack. In the GUI, the same thing happens except to the item the mouse is hovering over.
    - '1'-'9' # Switch active item to the one in a given hotbar cell.
    - 'left.control' # Move fast in the current direction of motion.
    - 'left.shift' # Move carefully in current direction of motion. In the GUI it acts as a modifier key: when used with attack it moves item from/to the inventory to/from the hotbar, and when used with craft it crafts the maximum number of items possible instead of just 1.
* no_op # wait and do not interact with the world
If multiple actions are activated, use and connect.
## Continuously take action until the task is completed. 
Wrap your hierarchical action inside Skill: skill actions 
Grounding: grounding actions 
Motion: motion actions 
Action: raw actions
## User Instruction

{instruction}

<image>
"""