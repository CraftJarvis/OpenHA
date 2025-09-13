
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

def get_system_prompt(system_prompt_tag):
    if system_prompt_tag == "text_action":
        return text_action_system_prompt
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
    "greedy": {"motion_coa", "grounding_coa"},
    "text_action": {"text_action"},
    "grounding": {"grounding_action"},
    "motion": {"motion_action"},
}