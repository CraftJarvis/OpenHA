import re

GUI_COORDINATE = (640,360)

pattern_point = re.compile(r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|point_start\|>\((-?\d+),\s*(-?\d+)\)<\|point_end\|>')
pattern_skill = re.compile(r'<skill>(.*?)</skill>')
pattern_grounding = re.compile(r'<grounding>\s*(.*?)\s*<\|point_start\|>\((-?\d+),\s*(-?\d+)\)<\|point_end\|>\s*</grounding>'   )
pattern_motion = re.compile(r'<motion>(.*?)</motion>')
pattern_action = re.compile(r'<raw>(.*?)</raw>')


def extract_point(text:str):
    matches = pattern_point.findall(text)
    contents = []
    for obj, x, y in matches:
        coordinates = [int(float(x)/1000*GUI_COORDINATE[0]),int(float(y)/1000*GUI_COORDINATE[1])]
        contents.append({
            "type": "point",
            "point":[coordinates],
            "label":obj,
        })
    return contents

def extract_skill(text):
    matches = pattern_skill.findall(text)
    contents = []
    for skill in matches:
        contents.append({
            "type":"skill",
            "action":skill.rstrip(),
        })
    return contents

def extract_grounding(text: str):
    matches = pattern_grounding.findall(text)
    contents = []
    for action, x, y in matches:
        coordinates = [
            int(float(x) / 1000 * GUI_COORDINATE[0]),
            int(float(y) / 1000 * GUI_COORDINATE[1])
        ]
        contents.append({
            "type": "grounding",
            "action": action.strip(),
            "point": [coordinates],
        })
    return contents


def extract_motion(text):
    matches = pattern_motion.findall(text)
    contents = []
    for action in matches:
        contents.append({
            "type":"motion",
            "action":action.rstrip(),
        })
    return contents

def extract_action(text):
    matches = pattern_action.findall(text)
    contents = []
    for action in matches:
        contents.append({
            "type":"raw_action",
            "action":action.rstrip(),
        })
    return contents

def extract_hierarchical_action(text):
    hierarchical_action = {}
    hierarchical_action["point"] = extract_point(text)
    hierarchical_action["skill"] = extract_skill(text)
    hierarchical_action["grounding"] = extract_grounding(text)
    hierarchical_action["motion"] = extract_motion(text)
    hierarchical_action["action"] = extract_action(text)
    return hierarchical_action