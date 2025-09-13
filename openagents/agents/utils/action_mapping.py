'''
Author: Muyao 2350076251@qq.com
Date: 2025-02-18 15:57:29
LastEditors: Muyao 2350076251@qq.com
LastEditTime: 2025-08-29 17:33:52
'''

import math
import re
import numpy as np
from collections import OrderedDict
from typing import Union,List,Dict,Any,Optional
from numbers import Number
import torch
import copy
import pickle
import json
from tqdm import tqdm
from pathlib import Path
from rich import console
from abc import ABC, abstractmethod
import random
from minestudio.utils.vpt_lib.actions import  ActionTransformer,Buttons
from minestudio.utils.vpt_lib.action_mapping import CameraHierarchicalMapping
from minestudio.simulator.entry import CameraConfig

ENV_NULL_ACTION = {
    'hotbar.1': 0, 'hotbar.2': 0, 'hotbar.3': 0, 'hotbar.4': 0, 'hotbar.5': 0,
    'hotbar.6': 0, 'hotbar.7': 0, 'hotbar.8': 0, 'hotbar.9': 0,
    'forward': 0, 'back': 0, 'left': 0, 'right': 0, 'sprint': 0, 'sneak': 0,
    'use': 0, 'drop': 0, 'attack': 0, 'jump': 0, 'inventory': 0,
    'camera': np.array([0.0, 0.0])
}
MOTION_NULL_ACTION = {"move":None, "move_state":None, "camera":None, "mouse_op":None, "keyboard_op":None}
MOTION_KEYS = list(MOTION_NULL_ACTION.keys())

# Given constants
KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape" :"ESC",
    "key.keyboard.s" :"back",
    "key.keyboard.q" :"drop",
    "key.keyboard.w" :"forward",
    "key.keyboard.1" :"hotbar.1",
    "key.keyboard.2" :"hotbar.2",
    "key.keyboard.3" :"hotbar.3",
    "key.keyboard.4" :"hotbar.4",
    "key.keyboard.5" :"hotbar.5",
    "key.keyboard.6" :"hotbar.6",
    "key.keyboard.7" :"hotbar.7",
    "key.keyboard.8" :"hotbar.8",
    "key.keyboard.9" :"hotbar.9",
    "key.keyboard.e" :"inventory",
    "key.keyboard.space" :"jump",
    "key.keyboard.a" :"left",
    "key.keyboard.d" :"right",
    "key.keyboard.left.shift" :"sneak",
    "key.keyboard.left.control" :"sprint",
    "key.keyboard.f" :"swapHands",
}
CAMERA_SCALER = 360.0 / 2400.0
# We need a reverse mapping to find the JSON key from the env_action key
REVERSE_KEYBOARD_MAPPING = {v: k for k, v in KEYBOARD_BUTTON_MAPPING.items()}


def merge_dict(list_dict:List[Dict[str,Any]]) -> Dict[str,List[Any]]:
    if len(list_dict) == 0:
        return {}
    dict_keys = Buttons.ALL + ["camera"]

    dict_list = {key:[] for key in dict_keys}
    for dict_ in list_dict:
        for key in dict_keys:
            if key in dict_:
                dict_list[key].append(dict_[key])
            else:
                dict_list[key].append(0)
    for key in dict_keys:
        dict_list[key] = np.array(dict_list[key])
    return dict_list

#--------------------
# TOOLS

def split_continuous_segments(frame_ids: List[int]) -> List[List[int]]:
    """
    将已排序的 frame_ids 拆成若干连续段。
    """
    if not frame_ids:
        return []

    segments = []
    current_segment = [frame_ids[0]]

    for idx in range(1, len(frame_ids)):
        # 如果当前帧与前一帧连续，就放进同一段
        if frame_ids[idx] == frame_ids[idx - 1] + 1:
            current_segment.append(frame_ids[idx])
        else:
            segments.append(current_segment)
            current_segment = [frame_ids[idx]]
    segments.append(current_segment)
    return segments

def sliding_windows(
        segment: List[int],
        chunk_len: int,
        sliding_window_len: int
) -> List[List[int]]:
    """
    在一个连续段上做滑动窗口抽取。
    """

    windows = []
    i = 0
    n = len(segment)
    while i < n:
        windows.append(segment[i:i + chunk_len])
        i += sliding_window_len
    return windows

def env_action_to_json_action(env_action):
        """
        Converts a MineRL environment action into a JSON action.
        """
        # Initialize the basic json_action structure
        json_action = {
            "mouse": {
                #"x": 0.0,
                #"y": 0.0,
                "dx": 0.0,
                "dy": 0.0,
                #"dwheel": 0.0,
                "buttons": [],
                #"newButtons": []
            },
            "keyboard": {
                "keys": [],
                #"newKeys": [],
                #"chars": ""
            }
        }
        # --- 1. Process Mouse Actions ---
        # Convert camera movement back to mouse dx/dy
        camera_action = env_action.get("camera", np.array([0., 0.]))
        if CAMERA_SCALER != 0:
            json_action["mouse"]["dy"] = round(float(camera_action[0] / CAMERA_SCALER),3)
            json_action["mouse"]["dx"] = round(float(camera_action[1] / CAMERA_SCALER),3)
        # Convert mouse button presses
        if env_action.get("attack") == 1:
            json_action["mouse"]["buttons"].append(0)
        if env_action.get("use") == 1:
            json_action["mouse"]["buttons"].append(1)
        if env_action.get("pickItem") == 1:
            json_action["mouse"]["buttons"].append(2)
        # --- 2. Process Keyboard Actions ---
        # Iterate through the mappable environment keys
        for env_key, json_key in REVERSE_KEYBOARD_MAPPING.items():
            # If the key is pressed in env_action, add the corresponding JSON key
            if env_action.get(env_key) == 1:
                json_action["keyboard"]["keys"].append(json_key)
        return json_action

def json_action_to_env_action(json_action):
    
    # This might be slow...
    env_action = copy.deepcopy(ENV_NULL_ACTION)
    # As a safeguard, make camera action again so we do not override anything
    env_action["camera"] = np.array([0., 0.])
    
    is_null_action = True
    keyboard_keys = json_action["keyboard"]["keys"]
    for key in keyboard_keys:
        # You can have keys that we do not use, so just skip them
        # NOTE in original training code, ESC was removed and replaced with
        #      "inventory" action if GUI was open.
        #      Not doing it here, as BASALT uses ESC to quit the game.
        if key in KEYBOARD_BUTTON_MAPPING:
            env_action[KEYBOARD_BUTTON_MAPPING[key]] = 1
            is_null_action = False
            
    mouse = json_action["mouse"]
    camera_action = env_action["camera"]
    camera_action[0] = mouse["dy"] * CAMERA_SCALER
    camera_action[1] = mouse["dx"] * CAMERA_SCALER

    if mouse["dx"] != 0 or mouse["dy"] != 0:
        is_null_action = False
    else:
        if abs(camera_action[0]) > 180:
            camera_action[0] = 0
        if abs(camera_action[1]) > 180:
            camera_action[1] = 0
    env_action["camera"] = camera_action.tolist()
    
            
    mouse_buttons = mouse["buttons"]
    if 0 in mouse_buttons:
        env_action["attack"] = 1
        is_null_action = False
    if 1 in mouse_buttons:
        env_action["use"] = 1
        is_null_action = False
    if 2 in mouse_buttons:
        env_action["pickItem"] = 1
        is_null_action = False

    return env_action, is_null_action

def get_non_markov_sequences(
        frame_ids: List[int],
        chunk_len: int,
        sliding_window_len: int
) -> List[List[int]]:
    """
    主入口：先拆段，再做滑窗，最大限度利用所有帧。
    """
    if chunk_len <= 0 or sliding_window_len <= 0:
        raise ValueError("chunk_len 和 sliding_window_len 必须为正整数")
    # 去重 & 排序，确保算法假设成立
    cleaned = sorted(set(frame_ids))
    segments = split_continuous_segments(cleaned)

    all_windows = []
    for seg in segments:
        all_windows.extend(sliding_windows(seg, chunk_len, sliding_window_len))
    return all_windows

#--------------------

def _to_list_if_ndarray(x):
    return x.tolist() if isinstance(x, np.ndarray) else x

def _equal(a, b, *, rel_tol=1e-7, abs_tol=1e-9):
    """递归相等判断：支持 ndarray/list/tuple/标量；float 用容忍度。"""
    # 先把 ndarray 转成 list，便于递归
    a = _to_list_if_ndarray(a)
    b = _to_list_if_ndarray(b)

    # 都是数字（含 bool 也算 number 的子类，这里额外处理）
    if isinstance(a, Number) and isinstance(b, Number):
        # bool 用严格相等；float/int 用 isclose
        if isinstance(a, bool) or isinstance(b, bool):
            return a == b
        # 用 isclose（对整数也安全）
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    # 字符串/None 等不可迭代简单类型
    if isinstance(a, (str, type(None))) or isinstance(b, (str, type(None))):
        return a == b

    # 列表/元组：长度+元素逐个比较
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_equal(x, y, rel_tol=rel_tol, abs_tol=abs_tol) for x, y in zip(a, b))

    # 字典：键集合和对应值递归比较
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_equal(a[k], b[k], rel_tol=rel_tol, abs_tol=abs_tol) for k in a.keys())

    # 其他类型（比如自定义对象），退化为直接相等
    return a == b

class ActionTokenizer(ABC):

    def __init__(self):
        self._env_null_action = ENV_NULL_ACTION
        self.keyboard_buttons = [
            'forward', 'back', 'left', 'right', 'sprint', 'sneak', 'jump',
            'hotbar.1', 'hotbar.2', 'hotbar.3', 'hotbar.4', 'hotbar.5',
            'hotbar.6', 'hotbar.7', 'hotbar.8', 'hotbar.9',
            'inventory', 'drop', 
        ]
        self.mouse_buttons = ['use','attack',]
        self.buttons = self.keyboard_buttons + self.mouse_buttons
        self.camera = "camera"
        
        self.act_beg_token = ""
        self.act_end_token = ""

    @abstractmethod
    def encode(self, trajectory: Dict, **kwargs) -> list[tuple[int]]:
        pass

    @abstractmethod
    def decode(self, tokens: Union[torch.Tensor, list], **kwargs) -> List[OrderedDict]:
        pass
    
    @property
    @abstractmethod
    def encode_null_action(self):
        pass
    
    @property
    def env_null_action(self):
        return copy.deepcopy(self._env_null_action)
    
    @property
    def json_null_action(self):
        return env_action_to_json_action(self.env_null_action)

    def is_same_env_action(self, action_a, action_b, *, 
                        rel_tol=1e-7, abs_tol=1e-9, ignore_extra_keys=False, verbose=False):
        """
        比较两个 env_action 是否等同：
        - 缺省键用 env_null_action 补齐
        - 浮点用容忍度比较
        - 可选是否忽略 a/b 中的额外键
        """
        # 1) 计算要比较的键集合
        env_null_action = self.env_null_action
        keys = set(env_null_action)
        if not ignore_extra_keys:
            keys |= set(action_a) | set(action_b)  # 不忽略额外键：一并比较

        # 2) 逐键取值，缺则用 env_null_action 默认
        for key in keys:
            va = action_a.get(key, env_null_action.get(key))
            vb = action_b.get(key, env_null_action.get(key))

            # 3) 递归比较
            same = _equal(va, vb, rel_tol=rel_tol, abs_tol=abs_tol)
            if not same:
                if verbose:
                    print(f"[DIFF] key={key!r}\n  A={va}\n  B={vb}")
                return False

        return True
    
    def is_same_json_action(self, action_a, action_b, *,
                        rel_tol=1e-7, abs_tol=1e-9,
                        ignore_extra_keys=False, verbose=False):
        """
        比较两个 json_action 是否相同
        - mouse.dx, mouse.dy 用浮点近似比较
        - mouse.buttons: int 列表，顺序一致才算相等
        - keyboard.keys: str 列表，顺序一致才算相等
        - 其余字段按普通 == 比较
        """
        json_null_action = self.json_null_action

        def compare_values(key, v1, v2):
            # 数值类：float
            if isinstance(v1, (float, int)) and isinstance(v2, (float, int)):
                return math.isclose(v1, v2, rel_tol=rel_tol, abs_tol=abs_tol)

            # list: 可能是 int 按钮 / str 键
            if isinstance(v1, list) and isinstance(v2, list):
                return v1 == v2

            # 其他：直接比较
            return v1 == v2

        # 获取所有需要检查的 keys
        keys_a = set(action_a.keys())
        keys_b = set(action_b.keys())
        all_keys = keys_a | keys_b | set(json_null_action.keys())

        for key in all_keys:
            if key not in action_a:
                val_a = json_null_action.get(key)
            else:
                val_a = action_a[key]

            if key not in action_b:
                val_b = json_null_action.get(key)
            else:
                val_b = action_b[key]

            # 如果是 dict，递归比较
            if isinstance(val_a, dict) and isinstance(val_b, dict):
                sub_keys = set(val_a.keys()) | set(val_b.keys()) | set(json_null_action.get(key, {}))
                for sub_key in sub_keys:
                    sub_val_a = val_a.get(sub_key, json_null_action.get(key, {}).get(sub_key))
                    sub_val_b = val_b.get(sub_key, json_null_action.get(key, {}).get(sub_key))
                    if not compare_values(f"{key}.{sub_key}", sub_val_a, sub_val_b):
                        if verbose:
                            print(f"Mismatch at {key}.{sub_key}: {sub_val_a} != {sub_val_b}")
                        return False
            else:
                if not compare_values(key, val_a, val_b):
                    if verbose:
                        print(f"Mismatch at {key}: {val_a} != {val_b}")
                    return False

        # 额外 keys 检查
        if not ignore_extra_keys:
            extra_a = keys_a - keys_b
            extra_b = keys_b - keys_a
            if extra_a or extra_b:
                if verbose:
                    print(f"Extra keys detected: {extra_a} vs {extra_b}")
                return False

        return True
    
class TextActionTokenizer(ActionTokenizer):
    """Tokenizer: env action  <->  text instruction"""

    env_action_to_keyboard_dict = {v: k.replace("key.", "").replace("keyboard.", "") for k,v in KEYBOARD_BUTTON_MAPPING.items()}
    keyboard_to_env_action_dict = {v: k for k, v in env_action_to_keyboard_dict.items()}
    mouse_button_to_env_action_dict = {
        "right":"use",
        "left":"attack"
    }
    NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

    camera_re = re.compile(
        rf"""move\(\s*['"]?({NUM})['"]?\s*,\s*['"]?({NUM})['"]?\s*\)"""
    )
    #camera_re       = re.compile(r"move\('?([-+]?\d*\.?\d+)'?,\s*'?([-+]?\d*\.?\d+)'?\)")
    keyboard_re     = re.compile(r"press\((.*?)\)") 
    #mouse_click_re  = re.compile(r"click\('(\w+)'\)") 
    mouse_click_re = re.compile(r"click\(\s*['\"]?(\w+)['\"]?\s*\)")


    def __init__(self, action_chunk_len: int = 1,sliding_window_len:Optional[int]=None, 
        reserved_camera:bool=False,reserved_keyboard:bool=False, 
        precision:int=0, 
        act_beg_token:Optional[str]=None,act_end_token:Optional[str]=None,
        keep_no_op_p:float = 1.0,
    ):
        super().__init__()
        self.act_beg_token = "Action:" if act_beg_token is None else act_beg_token
        self.act_end_token = "" if act_end_token is None else act_end_token #"</action>"
        self.action_re = re.compile(rf"{self.act_beg_token}(.*?){self.act_end_token}(?=\n|{self.act_beg_token}|\Z)", re.DOTALL)
        
        assert action_chunk_len > 0
        self.action_chunk_len = action_chunk_len 
        self.sliding_window_len = action_chunk_len if sliding_window_len is None else sliding_window_len
        
        self.reserved_camera = reserved_camera
        self.reserved_keyboard = reserved_keyboard
        self.keep_no_op_p = keep_no_op_p
        assert 0.0 <= self.keep_no_op_p <= 1.0
        assert isinstance(precision,int)
        self.precision = precision

    def encode(self, trajectory: Dict, camera_scale:float=1,replace_keyboard:Optional[Dict[str,str]]=None) -> List[Dict]:
        camera_scale = float(camera_scale)
        if replace_keyboard is None:
            replace_keyboard = {}
        encoded_trajectory = []
        actions = trajectory.get("actions", [])
        use_env_action = True
        null_action = self.env_null_action
        if "json_actions" in trajectory:
            actions = trajectory["json_actions"]
            null_action = self.json_null_action
            use_env_action = False
        if not actions:
            return encoded_trajectory

        trajectory_len = len(actions)

        observations       = trajectory.get("observations", [""] * trajectory_len)
        frame_ids = trajectory.get("frame_ids",   list(range(trajectory_len)))
        frame_idx_real_idx_map = {frame_idx:idx for idx,frame_idx in enumerate(frame_ids)}
        
        if not _equal(self.keep_no_op_p,1.0): #假如需要去掉一些keep_no_op_p
            new_frame_ids = []
            for adx, action in enumerate(actions):
                if ((use_env_action and self.is_same_env_action(null_action,action) ) \
                    or (not use_env_action and self.is_same_json_action(null_action,action) )) \
                    and random.random() > self.keep_no_op_p:
                    pass
                else:
                    new_frame_ids.append(frame_ids[adx])
            frame_ids = new_frame_ids

        
        interact_ids  = trajectory.get("interact_ids",[""] * trajectory_len)
        chunks = get_non_markov_sequences(frame_ids, chunk_len=self.action_chunk_len,sliding_window_len=self.sliding_window_len)
        for chunk in chunks:
            raw_chunk = [actions[frame_idx_real_idx_map[i]] for i in chunk]
            encoded_action = " ".join(
                    self._env_to_text(a,camera_scale=camera_scale,replace_keyboard=replace_keyboard,use_env_action=use_env_action) for a in raw_chunk
                )
            encoded_trajectory.append({
                "action":       encoded_action,
                "raw_action":   raw_chunk,
                "observations": [observations[frame_idx_real_idx_map[i]] for i in chunk],
                "interact_id":  interact_ids[frame_idx_real_idx_map[chunk[0]]],
                "frames":       (frame_ids[frame_idx_real_idx_map[chunk[0]]], len(chunk), frame_ids[frame_idx_real_idx_map[chunk[-1]]]),
            })
        #print([i["frames"] for i in encoded_trajectory])
        return encoded_trajectory

    def decode(self, tokens: str, replace_keyboard:Optional[Dict[str,str]]=None):
        """
        replace_keyboard: is the reverse of replace_keyboard in encode() 
            text_key -> env_key
        """
        matches = self.action_re.findall(tokens)
        env_actions = [self._text_to_env(t.strip(),replace_keyboard=replace_keyboard) for t in matches]
        env_actions = env_actions[-self.action_chunk_len:]
        if not env_actions:
            env_actions = [copy.deepcopy(self._env_null_action)]
        return env_actions

    # -----------------------------------------------------------------
    def _env_to_text(self, action: Dict,camera_scale:float,replace_keyboard:dict, use_env_action:bool) -> str:
        if use_env_action:
            json_action = env_action_to_json_action(action)
        else:
            json_action = action
        return self._json_to_text(json_action,camera_scale,replace_keyboard=replace_keyboard)
    
    def _text_to_env(self, text_action: str,replace_keyboard:Optional[Dict[str,str]]=None) -> Dict:
        env_action = copy.deepcopy(self._env_null_action)
        if "no_op" in text_action:
            return env_action

        # camera -------------------------------------------------------
        cam = self.camera_re.search(text_action)
        if cam:
            dx, dy = map(float, cam.groups())
            env_action[self.camera] = np.array([dy * CAMERA_SCALER, dx * CAMERA_SCALER])

        # keyboard -----------------------------------------------------
        kb = self.keyboard_re.search(text_action)
        if kb:
            for k in kb.group(1).split(","):
                k = k.strip().strip("'")
                if k:
                    new_k = k
                    if replace_keyboard is not None:
                        new_k = self.replace(keyboard_key=k,replace_keyboard=replace_keyboard)
                    env_key = self.keyboard_to_env_action_dict.get(new_k)
                    if env_key:
                        env_action[env_key] = 1

        # mouse --------------------------------------------------------
        for tok in re.split(r"\s*and\s*", text_action): 
            mc = self.mouse_click_re.match(tok.strip())    
            if mc:
                btn = mc.group(1)
                env_key = self.mouse_button_to_env_action_dict.get(btn)
                if env_key:
                    env_action[env_key] = 1
        return env_action

    def _json_to_text(self,json_action:Dict,camera_scale:float,replace_keyboard:dict) -> str:
        '''
        {'mouse': {'x': 640.0,
            'y': 360.0,
            'dx': 0.5,
            'dy': -6.5,
            'dwheel': 0.0,
            'buttons': [],
            'newButtons': []},
            'keyboard': {'keys': [], 'newKeys': [], 'chars': ''}}
        '''
        
        dx = json_action['mouse']['dx']*camera_scale
        dy = json_action['mouse']['dy']*camera_scale
        zero_x, zero_y = 0.0, 0.0
        if self.precision == 0:
            dx = int(round(dx,self.precision))
            dy = int(round(dy,self.precision))
            zero_x, zero_y = 0, 0
        else:
            dx = round(dx,self.precision)
            dy = round(dy,self.precision)
        buttons = json_action['mouse']['buttons']
        keys = json_action['keyboard']['keys']
        action_list = []
        # add mouse 
        if dx == zero_x and dy == zero_y and not self.reserved_camera:
            pass
        else:
            #action_list.append(f"move('{dx}', '{dy}')")
            action_list.append(f"move({dx}, {dy})")
        
        # add keyboards
        if keys or self.reserved_keyboard:
            #keys = [f"'{self.replace(key,replace_keyboard)}'" for key in keys] 
            keys = [f"{self.replace(key,replace_keyboard)}" for key in keys] 
            keyboard_action = f"press({', '.join(keys)})".replace("key.", "").replace("keyboard.", "")
            action_list.append(keyboard_action)

        # add mouse buttons
        if buttons:
            if 0 in buttons:
                #action_list.append("click('left')")
                action_list.append("click(left)")
            if 1 in buttons:
                #action_list.append("click('right')")
                action_list.append("click(right)")
            if 2 in buttons:
                #action_list.append("click('middle')")
                action_list.append("click(middle)")

        # return final action
        action_text = ' and '.join(action_list) if action_list else "no_op" 
        return f"{self.act_beg_token} {action_text} {self.act_end_token}"
    
    def replace(self, keyboard_key:str,replace_keyboard:dict):
        new_keyboard_key = keyboard_key
        if keyboard_key in replace_keyboard:
            new_keyboard_key = replace_keyboard[keyboard_key]
        return new_keyboard_key

    # -----------------------------------------------------------------
    @property
    def encode_null_action(self):
        return f"{self.act_beg_token} no_op {self.act_end_token}"

class MotionTokenizer(ActionTokenizer):
    # 最小动作触发阈值
    MIN_MOVE_TICKS = 0
    MIN_MOVE_DISTANCE = 0  # 曼哈顿距离阈值
    MIN_TURN_TICKS = 0
    MIN_TURN_ANGLE = 0
    MIN_CURSOR_MOVE_TICKS = 0
    MIN_CURSOR_MOVE_DISTANCE = 0

    def __init__(self, act_beg_token: Optional[str] = None, act_end_token: Optional[str] = None,
                 if_pass_same_motion: bool = False, keep_no_op_p: float = 0):
        super().__init__()
        self._motion_null_action = MOTION_NULL_ACTION
        self.motion_key = list(self._motion_null_action.keys())
        self.act_beg_token = "Motion:" if act_beg_token is None else act_beg_token
        self.act_end_token = "" if act_end_token is None else act_end_token
        self.action_re = re.compile(
            rf"{self.act_beg_token}(.*?){self.act_end_token}(?=\n|{self.act_beg_token}|\Z)",
            re.DOTALL
        )
        self.is_gui_open = False  # 缓存 GUI 状态
        self.if_pass_same_motion = if_pass_same_motion  # 是否合并连续相同动作
        self.keep_no_op_p = keep_no_op_p  # 保留 no-op 动作的概率

    def encode(self, trajectory: Dict, is_gui_open=False):
        # encode 主入口：将原始动作轨迹转为编码后的 motion token 序列
        self.is_gui_open = is_gui_open
        encoded_trajectory = []
        env_actions = trajectory["actions"]
        infos = trajectory.get("infos")
        if infos is None:
            infos = [dict()] * len(env_actions)
        if not env_actions:
            return encoded_trajectory
        motions = self.action_to_motion(
            env_actions=env_actions,
            infos=infos,
        )
        
        encoded_trajectory = self.encode_motion(
            env_actions = env_actions,
            motions=motions,
            observations=trajectory.get("observations"),
            interact_ids=trajectory.get("interact_ids"),
            frame_ids=trajectory.get("frame_ids"),
        )

        return encoded_trajectory

    def decode(self):
        pass

    def action_to_motion(self, env_actions, infos):
        # 将原始 env_action 序列转为 motion 语义结构
        self.init_count()
        len_reduce_one = len(env_actions)-1
        for idx, (env_action, info) in enumerate(zip(env_actions, infos)):
            # 初始化每帧的 motion 字典
            self.traj.append({
                "jump": "jump" if env_action["jump"] and not info.get("is_gui_open", self.is_gui_open) else None,
                "drop": "drop" if env_action.get("drop") else None,
            })

            # 分别处理每类动作
            if_not_final = len_reduce_one!=idx
            self.fb_move_process(env_action, info, if_not_final=if_not_final)      # 前后移动
            self.lr_move_process(env_action, info, if_not_final=if_not_final)      # 左右移动
            self.move_state_process(env_action, info)   # sneak/sprint 状态
            self.yaw_process(env_action, info, if_not_final=if_not_final)          # 左右转头
            self.pitch_process(env_action, info, if_not_final=if_not_final)        # 上下转头
            self.lr_cursor_process(env_action, info, if_not_final=if_not_final)    # GUI 左右鼠标移动
            self.ud_cursor_process(env_action, info, if_not_final=if_not_final)    # GUI 上下鼠标移动
            self.mouse_process(env_action, info)        # 鼠标点击类动作
            self.swap_process(env_action, info)         # 主副手切换
            self.hotbar_process(env_action, info)       # 快捷栏切换
            self.gui_process(env_action, info)          # GUI 打开关闭及 inventory 状态

        motions = []
        for motion in self.traj:
            current_motion = self.organize_motion(motion)
            motions.append(current_motion)
        return motions

    def encode_motion(self,
                      env_actions: List[Dict[str, str]],
                      motions: List[Dict[str, str]],
                      observations: List[Union[Path, str]] = None,
                      interact_ids: List[str] = None,
                      frame_ids: Optional[List[int]] = None):
        # 将 motion 序列转为 token 编码结构，包含 prompt、frame range 等信息
        trajectory_len = len(motions)
        if observations is None:
            observations = [""] * trajectory_len
        if interact_ids is None:
            interact_ids = [""] * trajectory_len
        if frame_ids is None:
            frame_ids = list(range(trajectory_len))

        previous_motion = copy.deepcopy(self._motion_null_action)
        chunks = [[]]  # 每个 chunk 是一组 frame id，表示连续动作段
        self.final_traj = []
        for current_index, current_motion in enumerate(motions):
            # 判断是否为 no-op 动作且不保留
            if current_motion == self._motion_null_action and random.random() > self.keep_no_op_p:
                if not chunks[-1]:
                    chunks.append([])
                previous_motion = copy.deepcopy(self._motion_null_action)
                continue

            # 连续相同动作合并
            if previous_motion == current_motion and self.if_pass_same_motion:
                chunks[-1].append(current_index)
                continue

            # 否则开启新段
            self.final_traj.append(current_motion)
            previous_motion = current_motion
            chunks.append([current_index])

        # 清理空段
        encode_trajectory = []
        chunks = [c for c in chunks if c]
        if not chunks:
            return encode_trajectory

        for idx, chunk in enumerate(chunks):
            motion_index = chunk[0]
            first_motion_idx = frame_ids[motion_index]
            last_motion_idx = frame_ids[chunk[-1]]
            motion = self.final_traj[idx]
            motion_prompt = self._motion_to_text(motion)

            # 整合编码结构
            data = {
                "interact_id": interact_ids[motion_index],
                "observations": [observations[index] for index in chunk],
                "motion_prompt": [motion_prompt] + [None] * (len(chunk) - 1),
                "frames": [first_motion_idx, last_motion_idx - first_motion_idx + 1, last_motion_idx],
                "motion": [motion],
                "action": [ env_actions[idx] for idx in chunk]
            }
            encode_trajectory.append(data)

        return encode_trajectory

    def init_count(self):
        # 初始化各类 buffer / 状态缓存
        self.traj = []
        self.offhand_buffer = None
        self.gui_buffer = self.is_gui_open
        self.inventory_buffer = 0

        # 移动状态缓存
        self.lr_move_buffer = 0
        self.lr_start_pos = None
        self.lr_duration = 0
        self.fb_move_buffer = 0
        self.fb_start_pos = None
        self.fb_duration = 0

        # 镜头转动状态缓存
        self.yaw_turn_buffer = 0
        self.yaw_start_angle = None
        self.yaw_duration = 0
        self.pitch_turn_buffer = 0
        self.pitch_start_angle = None
        self.pitch_duration = 0

        # GUI 内鼠标状态缓存
        self.lr_cursor_buffer = 0
        self.lr_accum_dx = 0
        self.lr_cursor_duration = 0
        self.ud_cursor_buffer = 0
        self.ud_accum_dy = 0
        self.ud_cursor_duration = 0

    def fb_move_process(self,env_action,info,if_not_final:bool=True):
        forward = env_action["forward"]
        back = env_action["back"]
        xpos = info.get("location_stats",{}).get("xpos")
        ypos = info.get("location_stats",{}).get("ypos")
        is_gui_open = info.get("is_gui_open",self.is_gui_open)
        if self.fb_move_buffer == -1:
            if forward and not is_gui_open and if_not_final:
                self.fb_duration += 1
            else:
                if ( xpos is None or ypos is None or abs(xpos - self.fb_start_pos[0]) + abs(ypos - self.fb_start_pos[1]) >= self.MIN_MOVE_DISTANCE) and self.fb_duration >= self.MIN_MOVE_TICKS :
                    for i in range(-1,-self.fb_duration-2, -1):
                        self.traj[i]["fb_move"] = "forward"
                else:
                    for i in range(-1,-self.fb_duration-2, -1):
                        self.traj[i]["fb_move"] = None
                self.fb_duration = 0
                self.fb_move_buffer = 0
                
                if back and not is_gui_open:
                    self.fb_move_buffer = 1
                    self.fb_start_pos = (xpos, ypos)
                    self.fb_duration += 1
                else:
                    self.traj[-1]["fb_move"] = None
                
        elif self.fb_move_buffer == 1:
            if back and not is_gui_open and if_not_final:
                self.fb_duration += 1
            else:
                if (xpos is None or ypos is None or abs(xpos - self.fb_start_pos[0]) + abs(ypos - self.fb_start_pos[1]) >= self.MIN_MOVE_DISTANCE) and self.fb_duration >= self.MIN_MOVE_TICKS:
                    for i in range(-1,-self.fb_duration-2, -1):
                        self.traj[i]["fb_move"] = "back"
                else:
                    for i in range(-1,-self.fb_duration-2, -1):
                        self.traj[i]["fb_move"] = None
                self.fb_duration = 0
                self.fb_move_buffer = 0
                
                if forward and not is_gui_open:
                    self.fb_move_buffer = -1
                    self.fb_start_pos = (xpos, ypos)
                    self.fb_duration += 1
                else:
                    self.traj[-1]["fb_move"] = None
        else:
            if not is_gui_open:
                # Deal with move start tick
                if forward and back:
                    self.fb_move_buffer = -1 if random.random() <= 0.5 else 1 # Use random to handle corner case.
                    self.fb_start_pos = (xpos, ypos)
                    self.fb_duration += 1
                elif forward:
                    self.fb_move_buffer = -1
                    self.fb_start_pos = (xpos, ypos)
                    self.fb_duration += 1
                elif back:
                    self.fb_move_buffer = 1
                    self.fb_start_pos = (xpos, ypos)
                    self.fb_duration += 1
                else:
                    self.traj[-1]["fb_move"] = None
            else:
                self.traj[-1]["fb_move"] = None
    
    def lr_move_process(self,env_action,info,if_not_final:bool=True):
        left = env_action["left"]
        right = env_action["right"]
        xpos = info.get("location_stats",{}).get("xpos",0)
        ypos = info.get("location_stats",{}).get("ypos",0)
        is_gui_open = info.get("is_gui_open",self.is_gui_open)
         
        if self.lr_move_buffer == -1:
            if left and not is_gui_open and if_not_final:
                self.lr_duration += 1
            else:
                if ( xpos is None or ypos is None or abs(xpos - self.lr_start_pos[0]) + abs(ypos - self.lr_start_pos[1]) >= self.MIN_MOVE_DISTANCE ) and self.lr_duration >= self.MIN_MOVE_TICKS:
                    for i in range(-1,-self.lr_duration-2, -1):
                        self.traj[i]["lr_move"] = "left"
                else:
                    for i in range(-1,-self.lr_duration-2, -1):
                        self.traj[i]["lr_move"] = None
                        
                self.lr_duration = 0
                self.lr_move_buffer = 0
                
                if right and not is_gui_open:
                    self.lr_move_buffer = 1
                    self.lr_start_pos = (xpos, ypos)
                    self.lr_duration += 1
                else:
                    self.traj[-1]["lr_move"] = None
                
        elif self.lr_move_buffer == 1:
            if right and not is_gui_open  and if_not_final:
                self.lr_duration += 1
            else:
                if (xpos is None or ypos is None or abs(xpos - self.lr_start_pos[0]) + abs(ypos - self.lr_start_pos[1]) >= self.MIN_MOVE_DISTANCE) and self.lr_duration >= self.MIN_MOVE_TICKS:
                    for i in range(-1,-self.lr_duration-2, -1):
                        self.traj[i]["lr_move"] = "right"
                else:
                    for i in range(-1,-self.lr_duration-2, -1):
                        self.traj[i]["lr_move"] = None
                        
                self.lr_duration = 0
                self.lr_move_buffer = 0
                
                if left and not is_gui_open:
                    self.lr_move_buffer = -1
                    self.lr_start_pos = (xpos, ypos)
                    self.lr_duration += 1
                else:
                    self.traj[-1]["lr_move"] = None
        
        else:
            if not is_gui_open:
                # Deal with move start tick
                if left and right:
                    self.lr_move_buffer = -1 if random.random() <= 0.5 else 1 # Use random to handle corner case.
                    self.lr_start_pos = (xpos, ypos)
                    self.lr_duration += 1
                elif left:
                    self.lr_move_buffer = -1
                    self.lr_start_pos = (xpos, ypos)
                    self.lr_duration += 1
                elif right:
                    self.lr_move_buffer = 1
                    self.lr_start_pos = (xpos, ypos)
                    self.lr_duration += 1
                else:
                    self.traj[-1]["lr_move"] = None
            else:
                self.traj[-1]["lr_move"] = None

    def move_state_process(self,env_action,info):
        sneak = env_action["sneak"]
        sprint = env_action["sprint"]
        is_gui_open = info.get("is_gui_open",self.is_gui_open)
        motion = None
        if not is_gui_open:
            if sneak and sprint:
                motion = "sneak" if random.random()<=0.5 else "sprint"
            elif sneak:
                motion = "sneak"
            elif sprint:
                motion = "sprint"

        self.traj[-1]["move_state"] = motion
   
    def yaw_process(self,env_action,info,if_not_final:bool=True):
        delta_yaw = env_action["camera"][1]
        yaw = info.get("location_stats",{}).get("yaw")
        is_gui_open = info.get("is_gui_open", self.is_gui_open)
        if self.yaw_turn_buffer == 1:
            if delta_yaw > 0 and not is_gui_open  and if_not_final:
                self.yaw_duration += 1
            else:
                if ( yaw is None or abs(yaw - self.yaw_start_angle) >= self.MIN_TURN_ANGLE ) and self.yaw_duration >= self.MIN_TURN_TICKS:
                    for i in range(-1,-self.yaw_duration-2, -1):
                        self.traj[i]["yaw"] = "right"
                else:
                    for i in range(-1,-self.yaw_duration-2, -1):
                        self.traj[i]["yaw"] =  None
                self.yaw_duration = 0
                self.yaw_turn_buffer = 0
                
                if delta_yaw < 0 and not is_gui_open:
                    self.yaw_duration += 1
                    self.yaw_turn_buffer = -1
                    self.yaw_start_angle = yaw
                else:
                    self.traj[-1]["yaw"] =  None
        elif self.yaw_turn_buffer == -1:
            if delta_yaw < 0 and not is_gui_open  and if_not_final:
                self.yaw_duration += 1
            else:
                if (yaw is None or  abs(yaw - self.yaw_start_angle) >= self.MIN_TURN_ANGLE)  and self.yaw_duration >= self.MIN_TURN_TICKS:
                    for i in range(-1,-self.yaw_duration-2, -1):
                        self.traj[i]["yaw"] = "left"
                else:
                    for i in range(-1,-self.yaw_duration-2, -1):
                        self.traj[i]["yaw"] =  None
                self.yaw_duration = 0
                self.yaw_turn_buffer = 0
                
                if delta_yaw > 0 and not is_gui_open:
                    self.yaw_duration += 1
                    self.yaw_turn_buffer = 1
                    self.yaw_start_angle = yaw
                else:
                    self.traj[-1]["yaw"] =  None
        else:
            if not is_gui_open:
                if delta_yaw > 0:
                    self.yaw_turn_buffer = 1
                    self.yaw_start_angle = yaw
                    self.yaw_duration += 1
                elif delta_yaw < 0:
                    self.yaw_turn_buffer = -1
                    self.yaw_start_angle = yaw
                    self.yaw_duration += 1
                else:
                    self.traj[-1]["yaw"] =  None
            else:
                self.traj[-1]["yaw"] =  None

    def pitch_process(self,env_action,info,if_not_final:bool=True):
        delta_pitch = env_action["camera"][0]
        pitch = info.get("location_stats",{}).get("pitch")
        is_gui_open = info.get("is_gui_open", self.is_gui_open)
        if self.pitch_turn_buffer == 1:
            if delta_pitch > 0 and not is_gui_open  and if_not_final:
                self.pitch_duration += 1
            else:
                if ( pitch is None or abs(pitch - self.pitch_start_angle) >= self.MIN_TURN_ANGLE ) and self.pitch_duration >= self.MIN_TURN_TICKS:
                    for i in range(-1,-self.pitch_duration-2, -1):
                        self.traj[i]["pitch"] = "down"
                else:
                    for i in range(-1,-self.pitch_duration-2, -1):
                        self.traj[i]["pitch"] =  None
                    
                self.pitch_duration = 0
                self.pitch_turn_buffer = 0
                
                if delta_pitch < 0 and not is_gui_open:
                    self.pitch_duration += 1
                    self.pitch_turn_buffer = -1
                    self.pitch_start_angle = pitch
                else:
                    self.traj[-1]["pitch"] =  None
        elif self.pitch_turn_buffer == -1:
            if delta_pitch < 0 and not is_gui_open  and if_not_final:
                self.pitch_duration += 1
            else:
                if ( pitch is None or abs(pitch - self.pitch_start_angle) >= self.MIN_TURN_ANGLE )  and self.pitch_duration >= self.MIN_TURN_TICKS:
                    for i in range(-1,-self.pitch_duration-2, -1):
                        self.traj[i]["pitch"] = "up"
                else:
                    for i in range(-1,-self.pitch_duration-2, -1):
                        self.traj[i]["pitch"] =  None
                self.pitch_duration = 0
                self.pitch_turn_buffer = 0
                
                if delta_pitch > 0 and not is_gui_open:
                    self.pitch_duration += 1
                    self.pitch_turn_buffer = 1
                    self.pitch_start_angle = pitch
                else:
                    self.traj[-1]["pitch"] =  None
        else:
            if not is_gui_open:
                if delta_pitch > 0:
                    self.pitch_turn_buffer = 1
                    self.pitch_start_angle = pitch
                    self.pitch_duration += 1
                elif delta_pitch < 0:
                    self.pitch_turn_buffer = -1
                    self.pitch_start_angle = pitch
                    self.pitch_duration += 1
                else:
                    self.traj[-1]["pitch"] =  None
            else:
                self.traj[-1]["pitch"] =  None

    def lr_cursor_process(self,env_action,info,if_not_final:bool=True):
        dx = env_action["camera"][1]
        is_gui_open = info.get("is_gui_open", self.is_gui_open)
        if self.lr_cursor_buffer == 1:
            if dx > 0 and is_gui_open  and if_not_final:
                self.lr_cursor_duration += 1
                self.lr_accum_dx += dx
            else:
                if abs(self.lr_accum_dx) >= self.MIN_CURSOR_MOVE_DISTANCE and self.lr_cursor_duration >= self.MIN_CURSOR_MOVE_TICKS:
                    for i in range(-1,-self.lr_cursor_duration-2, -1):
                        self.traj[i]["lr_cursor"] = "right"
                else:
                    for i in range(-1,-self.lr_cursor_duration-2, -1):
                        self.traj[i]["lr_cursor"] =  None
                self.lr_cursor_duration = 0
                self.lr_accum_dx = 0
                self.lr_cursor_buffer = 0
                
                if dx < 0 and is_gui_open:
                    self.lr_cursor_duration += 1
                    self.lr_accum_dx += dx
                    self.lr_cursor_buffer = -1
                else:
                    self.traj[-1]["lr_cursor"] =  None
        elif self.lr_cursor_buffer == -1:
            if dx < 0 and is_gui_open  and if_not_final:
                self.lr_cursor_duration += 1
                self.lr_accum_dx += dx
            else:
                if abs(self.lr_accum_dx) >= self.MIN_CURSOR_MOVE_DISTANCE and self.lr_cursor_duration >= self.MIN_CURSOR_MOVE_TICKS:
                    for i in range(-1,-self.lr_cursor_duration-2, -1):
                        self.traj[i]["lr_cursor"] = "left"
                else:
                    for i in range(-1,-self.lr_cursor_duration-2, -1):
                        self.traj[i]["lr_cursor"] =  None
                self.lr_cursor_duration = 0
                self.lr_accum_dx = 0
                self.lr_cursor_buffer = 0
                
                if dx > 0 and is_gui_open:
                    self.lr_cursor_duration += 1
                    self.lr_accum_dx += dx
                    self.lr_cursor_buffer = 1
                else:
                    self.traj[-1]["lr_cursor"] =  None
        else:
            if is_gui_open:
                if dx > 0:
                    self.lr_cursor_buffer = 1
                    self.lr_accum_dx += dx
                    self.lr_cursor_duration += 1
                elif dx < 0:
                    self.lr_cursor_buffer = -1
                    self.lr_accum_dx += dx
                    self.lr_cursor_duration += 1
                else:
                    self.traj[-1]["lr_cursor"] =  None
            else:
                self.traj[-1]["lr_cursor"] =  None

    def ud_cursor_process(self,env_action,info,if_not_final:bool=True):
        dy = env_action["camera"][0]
        is_gui_open = info.get("is_gui_open", self.is_gui_open)
        if self.ud_cursor_buffer == 1:
            if dy > 0 and is_gui_open  and if_not_final:
                self.ud_cursor_duration += 1
                self.ud_accum_dy += dy
            else:
                if abs(self.ud_accum_dy) >= self.MIN_CURSOR_MOVE_DISTANCE and self.ud_cursor_duration >= self.MIN_CURSOR_MOVE_TICKS:
                    for i in range(-1,-self.ud_cursor_duration-2, -1):
                        self.traj[i]["ud_cursor"] = "down"
                else:
                    for i in range(-1,-self.ud_cursor_duration-2, -1):
                        self.traj[i]["ud_cursor"] =  None
                    
                self.ud_cursor_duration = 0
                self.ud_accum_dy = 0
                self.ud_cursor_buffer = 0
                
                if dy < 0 and is_gui_open:
                    self.ud_cursor_duration += 1
                    self.ud_accum_dy += dy
                    self.ud_cursor_buffer = -1
                else:
                    self.traj[-1]["ud_cursor"] =  None
        elif self.ud_cursor_buffer == -1:
            if dy < 0 and is_gui_open  and if_not_final:
                self.ud_cursor_duration += 1
                self.ud_accum_dy += dy
            else:
                if abs(self.ud_accum_dy) >= self.MIN_CURSOR_MOVE_DISTANCE and self.ud_cursor_duration >= self.MIN_CURSOR_MOVE_TICKS:
                    for i in range(-1,-self.ud_cursor_duration-2, -1):
                        self.traj[i]["ud_cursor"] = "up"
                else:
                    for i in range(-1,-self.ud_cursor_duration-2, -1):
                        self.traj[i]["ud_cursor"] =  None
                self.ud_cursor_duration = 0
                self.ud_accum_dy = 0
                self.ud_cursor_buffer = 0
                
                if dy > 0 and is_gui_open:
                    self.ud_cursor_duration += 1
                    self.ud_accum_dy += dy
                    self.ud_cursor_buffer = 1
                else:
                    self.traj[-1]["ud_cursor"] =  None
        else:
            if is_gui_open:
                if dy > 0:
                    self.ud_cursor_buffer = 1
                    self.ud_accum_dy += dy
                    self.ud_cursor_duration += 1
                elif dy < 0:
                    self.ud_cursor_buffer = -1
                    self.ud_accum_dy += dy
                    self.ud_cursor_duration += 1
                else:
                    self.traj[-1]["ud_cursor"] =  None
            else:
                self.traj[-1]["ud_cursor"] =  None

    def mouse_process(self,env_action,info):
        attack = env_action["attack"]
        use = env_action["use"]
        is_gui_open = info.get("is_gui_open", self.is_gui_open)
        motion = None
        if is_gui_open:
            if attack and use:
                motion = "operate gui" if random.random()<=0.5 else "choose"
            elif attack:
                motion = "operate gui"
            elif use:
                motion = "choose"
            else:
                motion = None
        else:
            if attack and use:
                motion = "attack/mine and use/place"
            elif attack:
                motion = "attack/mine"
            elif use:
                motion = "use/place"
            else:
                motion = None
        self.traj[-1]["mouse"] = motion 
   
    def swap_process(self,env_action,info):
        temp_offhand = env_action.get("equipped_item",{}).get("offhand")
        motion = "swap" if temp_offhand and self.offhand_buffer is not None and self.offhand_buffer != temp_offhand else None
        self.offhand_buffer = temp_offhand
        self.traj[-1]["swap"] = motion
     
    def hotbar_process(self,env_action,info):
        hotbars = []
        for i in range(1, 10):
            if env_action[f"hotbar.{i}"]:
                hotbars.append(str(i))
        motion = f"switch to hotbar {', '.join(hotbars)}" if len(hotbars) != 0 else None
        self.traj[-1]["hotbar"] = motion
        
    def gui_process(self,env_action,info):
        inventory = env_action["inventory"]
        is_gui_open = info.get("is_gui_open", self.is_gui_open)
    
        motion = None
        if self.gui_buffer == 0:
            if inventory and self.inventory_buffer == 0:
                self.inventory_buffer = 1
                motion = "open inventory"
            elif inventory and self.inventory_buffer == 1:
                motion = "open inventory"
            elif inventory and self.inventory_buffer == 2:
                self.inventory_buffer = 3
                motion = "close inventory"
            elif inventory and self.inventory_buffer == 3:
                motion = "close inventory"
            elif not inventory and self.inventory_buffer == 1:
                self.inventory_buffer = 2
            elif not inventory and self.inventory_buffer == 3:
                self.inventory_buffer = 0
        
        # If inventory remain unchanged but is_gui_open is changed.
        if self.inventory_buffer == 0:
            if is_gui_open and self.gui_buffer == 0:
                self.gui_buffer = 1
                motion = "open gui"
            elif not is_gui_open and self.gui_buffer == 1:
                self.gui_buffer = 0
                motion = "close gui"
        self.traj[-1]["gui"] = motion
    
    def organize_motion(self, motion_entity):
        final_motion_dict = copy.deepcopy(self._motion_null_action)
        # 合并move
        if motion_entity.get("fb_move") and motion_entity.get("lr_move"):
            final_motion_dict["move"] = f"move {motion_entity['fb_move']} and move {motion_entity['lr_move']}"
        elif motion_entity.get("fb_move"):
            final_motion_dict["move"] = f"move {motion_entity['fb_move']}"
        elif motion_entity.get("lr_move"):
            final_motion_dict["move"] = f"move {motion_entity['lr_move']}"

        final_motion_dict["move_state"] = motion_entity.get("move_state")
        final_motion_dict["mouse_op"] = motion_entity.get("mouse")

        if motion_entity.get("yaw") and motion_entity.get("pitch"):
            final_motion_dict["camera"] = f"turn {motion_entity['yaw']} and {motion_entity['pitch']}"
        elif motion_entity.get("yaw"):
            final_motion_dict["camera"] = f"turn {motion_entity['yaw']}"
        elif motion_entity.get("pitch"):
            final_motion_dict["camera"] = f"turn {motion_entity['pitch']}"

        if motion_entity.get("lr_cursor") and motion_entity.get("ud_cursor"):
            final_motion_dict["camera"] = f"cursor move {motion_entity['lr_cursor']} and {motion_entity['ud_cursor']}"
        elif motion_entity.get("lr_cursor"):
            final_motion_dict["camera"] = f"cursor move {motion_entity['lr_cursor']}"
        elif motion_entity.get("ud_cursor"):
            final_motion_dict["camera"] = f"cursor move {motion_entity['ud_cursor']}"

        keyboard_op = []
        if motion_entity.get("jump"):
            keyboard_op.append(motion_entity["jump"])
        if motion_entity.get("drop"):
            keyboard_op.append(motion_entity["drop"])
        if motion_entity.get("swap"):
            keyboard_op.append(motion_entity["swap"])
        if motion_entity.get("hotbar"):
            keyboard_op.append(motion_entity["hotbar"])
        if motion_entity.get("gui"):
            keyboard_op.append(motion_entity["gui"])
        keyboard_op_text = ", ".join(keyboard_op)
        final_motion_dict["keyboard_op"] = keyboard_op_text if keyboard_op_text else None
        return final_motion_dict

    def _motion_to_text(self, motion):
        # 将 motion 字典转换为 motion_prompt 字符串，用于最终输出
        motion_action_list = []
        for key in self.motion_key:
            current_motion = motion.get(key)
            if current_motion:
                motion_action_list.append(current_motion)

        # 替换 '/' 为 ' / ' 避免语义歧义
        motion_action_list = [motion_action_str.replace("/", " / ") for motion_action_str in motion_action_list]
        if motion_action_list:
            motion_prompt = ', '.join(motion_action_list)
        else:
            motion_prompt = 'no_op'
        motion_prompt = f"{self.act_beg_token} {motion_prompt} {self.act_end_token}"
        return motion_prompt
    @property
    def encode_null_action(self):
        return f"{self.act_beg_token} {self.act_end_token}"

class ReservedActionTokenizer(ActionTokenizer):
    def __init__(self, tokenizer_type="llama-3",
                 camera_quantization_scheme="mu_law",
                 camera_mu=20,
                 camera_binsize=1,
                 camera_maxval=10,
                 keep_no_op_p:float = 0.0,
                 ):
        
        super().__init__()
        self.tokenizer_type = tokenizer_type

        self.act_beg_id = self.tag_token(0, self.tokenizer_type, return_type=1)
        self.act_end_id = self.tag_token(1, self.tokenizer_type, return_type=1)

        self.act_beg_token = self.tag_token(0, self.tokenizer_type, return_type=0)
        self.act_end_token = self.tag_token(1, self.tokenizer_type, return_type=0)

        camera_config = CameraConfig(
            camera_maxval=camera_maxval,
            camera_binsize=camera_binsize,
            camera_quantization_scheme=camera_quantization_scheme,
            camera_mu=camera_mu,
        )
        self.camera_config = camera_config

        self.n_camera_bins = camera_config.n_camera_bins

        self.action_transformer = ActionTransformer(**camera_config.action_transformer_kwargs)
        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=camera_config.n_camera_bins)
        self.keep_no_op_p = keep_no_op_p
        assert 0.0 <= self.keep_no_op_p <= 1.0
    
    def map_control_token(self,num:int, place:int, tokenizer_type:str = "llama-2",not_text=False) -> str:
    # [10,3,3,3,2,2,2,2,2] 2
    # sprint use drop attack jump camera
    # 3 sprint sneak
    # 2 use
    # 2 drop
    # 2 attack
    # 2 jump
    # 2 camera
        if tokenizer_type == "llama-2":
            special_tokens = [
                (('진', 31536),('জ', 31537),('천', 31563),('년', 31571),('세', 31578),('민', 31582),('ർ', 31585),('ἡ', 31598),('호', 31603),('ਰ', 31604),
                ("동", 31000), ("Υ", 31001), ("┌", 31002), ("ボ", 31003), ("宮", 31004), ("』", 31005), ("ম", 31006), ("『", 31007), ("ļ", 31008), ("श", 31009), ("ป", 31010), ("Ա", 31011), ("ब", 31012), ("자", 31013), ("政", 31014), ("ா", 31015), ("间", 31016), ("ﬁ", 31017), ("松", 31018), ("ṃ", 31019), ("始", 31020), ("息", 31021), ("少", 31022), ("教", 31023), ("获", 31024), ("列", 31025), ("开", 31026), ("ტ", 31027), ("ワ", 31028), ("კ", 31029), ("科", 31030), ("春", 31031), ("治", 31032), ("吉", 31033), ("ས", 31034), ("ศ", 31035), ("ɒ", 31036), ("台", 31037), ("ネ", 31038), ("း", 31039), ("ĩ", 31040), ("工", 31041), ("ά", 31042), ("知", 31043),
                ("八", 31044), ("場", 31045), ("画", 31046), ("百", 31047), ("☆", 31048), ("記", 31049), 
                ("得", 31050), ("ソ", 31051), ("氏", 31052), ("ာ", 31053), ("에", 31054), ("ল", 31055), ("ṛ", 31056), ("关", 31057), ("ġ", 31058), 
                ("έ", 31059), ("∑", 31060), ("ベ", 31061), ("标", 31062), ("니", 31063), ("ὴ", 31064), ("ֵ", 31065), ("外", 31066), ("♠", 31067), ("わ", 31068), ("間", 31069), ("ภ", 31070), ("校", 31071), ("制", 31072), ("แ", 31073), ("力", 31074), ("門", 31075), ("好", 31076), ("ғ", 31077), ("Ù", 31078), ("ℓ", 31079), ("ֶ", 31080), ("는", 31081), ("┐", 31082), ("∗", 31083), ("指", 31084), ("色", 31085), ("返", 31086), ("馬", 31087), ("请", 31088), ("≫", 31089), ("風", 31090), ("ό", 31091), ("接", 31092), ("서", 31093), ("↳", 31094), ("せ", 31095), ("志", 31096), ("̲", 31097), ("魔", 31098), ("ң", 31099), ("更", 31100), ("程", 31101), ("김", 31102), ("郡", 31103), 
                ("ོ", 31104), ("ũ", 31105), ("ച", 31106), ("利", 31107), ("県", 31108), ("周", 31109), ("そ", 31110), ("や", 31111), ("谷", 31112), ("香", 31113), ("♯", 31114), ("じ", 31115), ("،", 31116), ("期", 31117), ("∅", 31118), ("┘", 31119), ("初", 31120), ("福", 31121), ("片", 31122), ("ザ", 31123), ("動", 31124), ("参", 31125), ("성", 31126), ("Ə", 31127), ("╦", 31128), ("어", 31129), ("ხ", 31130), ("義", 31131), ("च", 31132), ("象", 31133), ("功", 31134), ("♂", 31135), ("도", 31136), ("고", 31137), ("过", 31138), ("վ", 31139), ("皇", 31140), ("特", 31141), ("ậ", 31142), ("长", 31143), ("英", 31144), ("ấ", 31145), ("ണ", 31146), ("Ъ", 31147), ("স", 31148),
                ("其", 31149), ("ত", 31150), ("流", 31151), ("除", 31152), ("일", 31153), ("ু", 31154), ("្", 31155), ("永", 31156), ("直", 31157), ("상", 31158), ("千", 31159), ("ắ", 31160), ("館", 31161), ("Ť", 31162), ("朝", 31163), ("ட", 31164), ("ɣ", 31165), ("单", 31166), ("ʀ", 31167), ("格", 31168), ("德", 31169), ("전", 31170), ("☺", 31171), ("ピ", 31172), ("歌", 31173), ("进", 31174), ("限", 31175), ("夫", 31176), ("트", 31177), ("⊢", 31178), ("園", 31179), ("量", 31180), ("土", 31181), ("放", 31182), ("码", 31183), ("等", 31184), ("系", 31185), ("∼", 31186), ("華", 31187), ("↵", 31188), ("소", 31189), ("常", 31190), ("否", 31191), ("見", 31192), ("源", 31193), 
                ("实", 31195), ("博", 31196), ("라", 31197), ("원", 31198), ("보", 31199), ("⊕", 31200), ("解", 31201), ("〜", 31202), ("男", 31203), ("দ", 31204), ("ポ", 31205), ("ろ", 31206), ("나", 31207), ("ག", 31208), ("無", 31209), ("Û", 31210), ("̥", 31211), ("ұ", 31212), ("查", 31213), ("̣", 31214), ("╗", 31215), ("╩", 31216), ("条", 31217), ("য", 31218), ("ὁ", 31219), ("後", 31220), ("他", 31221), ("网", 31222), ("ல", 31223), ("≃", 31224), ("화", 31225), ("ە", 31226), ("阿", 31227), ("ေ", 31228), ("户", 31229), ("∫", 31230), ("구", 31231), ("ར", 31232), ("မ", 31233), ("▸", 31234), ("լ", 31235), ("○", 31236), ("命", 31237), ("就", 31238), ("龍", 31239), ("君", 31240), 
                ("夏", 31241), ("言", 31243), ("先", 31244), ("➜", 31245), ("შ", 31246), ("ძ", 31247), ("ਾ", 31248), ("வ", 31249), ("ど", 31250), ("ヒ", 31251), ("ไ", 31252), ("ன", 31253), ("ば", 31254), ("ギ", 31255), ("գ", 31256), ("ἄ", 31257), ("ヤ", 31258), ("典", 31259), ("府", 31260), ("̄", 31261), ("신", 31262), ("组", 31263), ("改", 31264), ("ὲ", 31265),("华", 31266), ("与", 31267), ("调", 31268), ("╝", 31269), ("ヴ", 31270), ("ქ", 31271), ("由", 31272), ("修", 31273), ("學", 31274), ("♣", 31275), ("消", 31276), ("符", 31277), ("ʌ", 31278), ("부", 31279), ("ớ", 31280), ("‾", 31281), ("▲", 31282), ("录", 31283), ("ള", 31284), ("연", 31285), ("을", 31286), ("ひ", 31287), 
                ("영", 31288), ("┤", 31289), ("已", 31290), ("陽", 31291), ("င", 31292), ("국", 31293), ("容", 31294), ("未", 31295), ("宗", 31296), ("ᴇ", 31297), ("び", 31298), ("장", 31299), ("龙", 31300), ("්", 31301), ("提", 31302), ("ĝ", 31303), ("六", 31304), ("形", 31305), ("제", 31306), ("Հ", 31307), ("伊", 31308), ("ϵ", 31309), ("ข", 31310), ("Ű", 31311), ("ゃ", 31312), ("火", 31313), ("Ṣ", 31314), ("佐", 31315), ("⊥", 31316), ("̪", 31317), ("ứ", 31318), ("□", 31319), ("结", 31320), ("九", 31321), ("雄", 31322), ("թ", 31323), ("ា", 31324), ("而", 31325), ("བ", 31326), ("우", 31327), ("张", 31328), ("ट", 31329), ("ष", 31330), ("向", 31331), ("ῥ", 31332), ("选", 31333), 
                ("공", 31334), ("ゲ", 31335), ("ʐ", 31336), ("仁", 31337), ("堂", 31338), ("ך", 31339), ("ု", 31340), ("ἔ", 31341), ("അ", 31342), ("ề", 31343), ("ད", 31344), ("선", 31345), ("오", 31346), ("久", 31347), ("", 31348), ("义", 31349), ("अ", 31350), ("╔", 31351), ("无", 31352), ("", 31353), ("은", 31354), ("ʷ", 31355), ("那", 31356), ("線", 31357), ("务", 31358), ("基", 31359), ("属", 31360), ("配", 31361), ("미", 31362), ("軍", 31363), ("โ", 31364), ("津", 31365), ("完", 31366), ("研", 31367), ("注", 31368), ("失", 31369), ("应", 31370), ("က", 31371), ("╚", 31372), ("友", 31373), ("章", 31374), ("Ψ", 31375), ("求", 31376), ("ण", 31377), ("경", 31378),  ("भ", 31380), 
                ("们", 31381), ("模", 31382), ("需", 31383), ("ச", 31384), ("電", 31385), ("প", 31386), ("դ", 31387), ("へ", 31388), ("此", 31389), ("夜", 31390), ("或", 31391), ("橋", 31392), ("根", 31393), ("Ī", 31394), ("玉", 31395), ("ู", 31396), ("ṅ", 31397), ("交", 31398), ("品", 31399), ("良", 31400), ("ང", 31401), ("ォ", 31402), ("则", 31403), ("開", 31404), ("Ζ", 31405), ("문", 31406), ("被", 31407), ("조", 31408), ("株", 31409), ("记", 31410), ("會", 31411), ("经", 31412), ("ू", 31413), ("ょ", 31414), ("转", 31415), ("崎", 31416), ("마", 31417), ("⌘", 31418), ("比", 31419), ("造", 31420), ("ܐ", 31421), ("ื", 31422), ("没", 31423), ("现", 31424), ("七", 31425), ("Ά", 31426), 
                ("商", 31427), ("ை", 31428), ("机", 31429), ("阳", 31430), ("ĉ", 31431), ("角", 31432), ("站", 31433), ("բ", 31434), ("해", 31435), ("及", 31436),("ध", 31437), ("術", 31438), ("认", 31439), ("", 31440), ("创", 31441), ("編", 31442), ("ղ", 31443), ("ḩ", 31444), ("伝", 31445), ("岡", 31446), ("ड", 31447), ("ホ", 31448), ("港", 31449), ("任", 31450), ("登", 31451), ("ི", 31452), ("็", 31453), ("布", 31454), ("究", 31455), ("帝", 31456), ("여", 31457), ("산", 31458), ("န", 31459), ("◦", 31460), ("密", 31461), ("变", 31462), ("序", 31463), ("♀", 31464), ("∣", 31465), ("计", 31466), ("曲", 31467), ("Ă", 31468), ("ύ", 31469), ("ʋ", 31470), ("传", 31471), ("】", 31472), ("包", 31473),
                ("意", 31474), ("去", 31475), ("沙", 31476), ("⸮", 31477), ("【", 31478), ("写", 31479), ("超", 31480), ("ய", 31481), ("今", 31482), ("┈", 31483), ("森", 31484), ("ි", 31485), ("⊗", 31486), ("비", 31487), ("հ", 31488), ("Ḩ", 31489), ("ǫ", 31490), ("黄", 31491), ("∙", 31492), ("드", 31493), ("🌍", 31494), ("景", 31495), ("湖", 31496), ("ք", 31497), ("ိ", 31498), ("ⁿ", 31499), ("̂", 31500), ("ペ", 31501), ("何", 31502), ("宇", 31503), ("張", 31504), ("语", 31505), ("老", 31506), ("例", 31507), ("Ṭ", 31508), ("鉄", 31509), ("克", 31510), ("☉", 31511), ("", 31512), ("ɹ", 31513), ("ἱ", 31514), ("ⴰ", 31515), ("然", 31516), ("를", 31517), ("ǧ", 31518), ("報", 31519), ("服", 31520),
                ("Ď", 31521), ("想", 31522), ("‖", 31523), ("ユ", 31524), ("実", 31525), ("载", 31526),
                ),
                (('그', 31607),('න', 31609),('ན', 31614),),
                (('ゆ', 31621),('ご', 31622),('현', 31680),),
                (('군', 31699), ('무', 31716), ('위', 31724),),
                (('안', 31734), ('박', 31736),),
                (('용', 31737), ('단', 31746),),
                (('면', 31747), ('남', 31754),),
                (('강', 31774), ('씨', 31781),),
                (('개', 31789), ('들', 31804),),
                (('차', 31817), ('학', 31822), ('만', 31826), ('터', 31856), ('식', 31895), ('과', 31906), ('타', 31925), ('종', 31930), ('내', 31940), ('중', 31941), ('방', 31945)),
                (('월', 31950), ('회', 31953), ('모', 31962), ('바', 31963), ('음', 31966), ('재', 31973), ('명', 31976), ('합', 31980), ('역', 31987), ('백', 31989), ('왕', 31996)),
            ]
        elif tokenizer_type == "mistral":
            special_tokens = [
                (('朱', 31947),('ǝ', 31948),('Ḩ', 31949),('担', 31950),('灰', 31951), ('讲', 31952), ('롤', 31953),('😤', 31955),('ោ', 31956),('애', 31957),),
                (('였', 31958),('질', 31959),('振', 31960),),
                (('灯', 31961),('ĉ', 31962),('ස', 31963),),
                (('閉', 31964),('램', 31965),('ಂ', 31966),),
                (('げ', 31967),('ふ', 31896),),
                (('狂', 31969),('融', 31970),),
                (('仍', 31971),('實', 31972),),
                (('楽', 31973),('範', 31974),),
                (('వ', 31976),('嵌', 31977),),
                (('摩', 31978),('袁', 31979),('ষ', 31980),('乎', 31981),('규', 31982),('岗', 31983),('糊', 31984),('క', 31985),('雲', 31986),('심', 31987),('ई', 31988),('庭', 31926), ('苗', 31927),('闲', 31929), ('독', 31930), ('ɹ', 31931), ('ҽ', 31932), ('ថ', 31933), ('宏', 31934), ('尊', 31935), ('總', 31936),),
                (('འ', 31989),('ἡ', 31990),('丝', 31991),('Ħ', 31992),('ٍ', 31993),('ٓ', 31994),('အ', 31995),('執', 31996),('벨', 31997),('ゼ', 31998),('梦', 31999), ('裝', 31937), ('ම', 31938), ('▸', 31939), ('測', 31940), ('勇', 31920), ('徐', 31921), ('轩', 31943), ('兄', 31944), ('剑', 31945), ('ન', 31946),),
            ]
        elif tokenizer_type == "llama-3":
            special_tokens = [
                (('<|reserved_special_token_180|>', 128185), ('<|reserved_special_token_181|>', 128186), ('<|reserved_special_token_182|>', 128187), ('<|reserved_special_token_183|>', 128188), ('<|reserved_special_token_184|>', 128189), ('<|reserved_special_token_185|>', 128190), ('<|reserved_special_token_186|>', 128191), ('<|reserved_special_token_187|>', 128192), ('<|reserved_special_token_188|>', 128193), ('<|reserved_special_token_189|>', 128194), ('<|reserved_special_token_5|>', 128010), ('<|reserved_special_token_6|>', 128011), ('<|reserved_special_token_7|>', 128012), ('<|reserved_special_token_8|>', 128013), ('<|reserved_special_token_9|>', 128014), ('<|reserved_special_token_10|>', 128015), ('<|reserved_special_token_11|>', 128016), ('<|reserved_special_token_12|>', 128017), ('<|reserved_special_token_13|>', 128018), ('<|reserved_special_token_14|>', 128019), ('<|reserved_special_token_15|>', 128020), 
                ('<|reserved_special_token_16|>', 128021), ('<|reserved_special_token_17|>', 128022), ('<|reserved_special_token_18|>', 128023), ('<|reserved_special_token_19|>', 128024), ('<|reserved_special_token_20|>', 128025), ('<|reserved_special_token_21|>', 128026), ('<|reserved_special_token_22|>', 128027), ('<|reserved_special_token_23|>', 128028), ('<|reserved_special_token_24|>', 128029), ('<|reserved_special_token_25|>', 128030), ('<|reserved_special_token_26|>', 128031), ('<|reserved_special_token_27|>', 128032), ('<|reserved_special_token_28|>', 128033), ('<|reserved_special_token_29|>', 128034), ('<|reserved_special_token_30|>', 128035), ('<|reserved_special_token_31|>', 128036), ('<|reserved_special_token_32|>', 128037), ('<|reserved_special_token_33|>', 128038), ('<|reserved_special_token_34|>', 128039), ('<|reserved_special_token_35|>', 128040), ('<|reserved_special_token_36|>', 128041), 
                ('<|reserved_special_token_37|>', 128042), ('<|reserved_special_token_38|>', 128043), ('<|reserved_special_token_39|>', 128044), ('<|reserved_special_token_40|>', 128045), ('<|reserved_special_token_41|>', 128046), ('<|reserved_special_token_42|>', 128047), ('<|reserved_special_token_43|>', 128048), ('<|reserved_special_token_44|>', 128049), ('<|reserved_special_token_45|>', 128050), ('<|reserved_special_token_46|>', 128051), ('<|reserved_special_token_47|>', 128052), ('<|reserved_special_token_48|>', 128053), ('<|reserved_special_token_49|>', 128054), ('<|reserved_special_token_50|>', 128055), ('<|reserved_special_token_51|>', 128056), ('<|reserved_special_token_52|>', 128057), ('<|reserved_special_token_53|>', 128058), ('<|reserved_special_token_54|>', 128059), ('<|reserved_special_token_55|>', 128060), ('<|reserved_special_token_56|>', 128061), ('<|reserved_special_token_57|>', 128062), 
                ('<|reserved_special_token_58|>', 128063), ('<|reserved_special_token_59|>', 128064), ('<|reserved_special_token_60|>', 128065), ('<|reserved_special_token_61|>', 128066), ('<|reserved_special_token_62|>', 128067), ('<|reserved_special_token_63|>', 128068), ('<|reserved_special_token_64|>', 128069), ('<|reserved_special_token_65|>', 128070), ('<|reserved_special_token_66|>', 128071), ('<|reserved_special_token_67|>', 128072), ('<|reserved_special_token_68|>', 128073), ('<|reserved_special_token_69|>', 128074), ('<|reserved_special_token_70|>', 128075), ('<|reserved_special_token_71|>', 128076), ('<|reserved_special_token_72|>', 128077), ('<|reserved_special_token_73|>', 128078), ('<|reserved_special_token_74|>', 128079), ('<|reserved_special_token_75|>', 128080), ('<|reserved_special_token_76|>', 128081), ('<|reserved_special_token_77|>', 128082), ('<|reserved_special_token_78|>', 128083), 
                ('<|reserved_special_token_79|>', 128084), ('<|reserved_special_token_80|>', 128085), ('<|reserved_special_token_81|>', 128086), ('<|reserved_special_token_82|>', 128087), ('<|reserved_special_token_83|>', 128088), ('<|reserved_special_token_84|>', 128089), ('<|reserved_special_token_85|>', 128090), ('<|reserved_special_token_86|>', 128091), ('<|reserved_special_token_87|>', 128092), ('<|reserved_special_token_88|>', 128093), ('<|reserved_special_token_89|>', 128094), ('<|reserved_special_token_90|>', 128095), ('<|reserved_special_token_91|>', 128096), ('<|reserved_special_token_92|>', 128097), ('<|reserved_special_token_93|>', 128098), ('<|reserved_special_token_94|>', 128099), ('<|reserved_special_token_95|>', 128100), ('<|reserved_special_token_96|>', 128101), ('<|reserved_special_token_97|>', 128102), ('<|reserved_special_token_98|>', 128103), ('<|reserved_special_token_99|>', 128104), 
                ('<|reserved_special_token_100|>', 128105), ('<|reserved_special_token_101|>', 128106), ('<|reserved_special_token_102|>', 128107), ('<|reserved_special_token_103|>', 128108), ('<|reserved_special_token_104|>', 128109), ('<|reserved_special_token_105|>', 128110), ('<|reserved_special_token_106|>', 128111), ('<|reserved_special_token_107|>', 128112), ('<|reserved_special_token_108|>', 128113), ('<|reserved_special_token_109|>', 128114), ('<|reserved_special_token_110|>', 128115), ('<|reserved_special_token_111|>', 128116), ('<|reserved_special_token_112|>', 128117), ('<|reserved_special_token_113|>', 128118), ('<|reserved_special_token_114|>', 128119), ('<|reserved_special_token_115|>', 128120), ('<|reserved_special_token_116|>', 128121), ('<|reserved_special_token_117|>', 128122), ('<|reserved_special_token_118|>', 128123), ('<|reserved_special_token_119|>', 128124), ('<|reserved_special_token_120|>', 128125), 
                ('<|reserved_special_token_121|>', 128126), ('<|reserved_special_token_122|>', 128127), ('<|reserved_special_token_123|>', 128128), ('<|reserved_special_token_124|>', 128129), ('<|reserved_special_token_125|>', 128130), ('<|reserved_special_token_126|>', 128131), ('<|reserved_special_token_127|>', 128132), ('<|reserved_special_token_128|>', 128133), ('<|reserved_special_token_129|>', 128134), ('<|reserved_special_token_130|>', 128135), ('<|reserved_special_token_131|>', 128136), ('<|reserved_special_token_132|>', 128137), ('<|reserved_special_token_133|>', 128138), ('<|reserved_special_token_134|>', 128139), ('<|reserved_special_token_135|>', 128140), ('<|reserved_special_token_136|>', 128141), ('<|reserved_special_token_137|>', 128142), ('<|reserved_special_token_138|>', 128143), ('<|reserved_special_token_139|>', 128144), ('<|reserved_special_token_140|>', 128145), ('<|reserved_special_token_141|>', 128146), 
                ('<|reserved_special_token_142|>', 128147), ('<|reserved_special_token_143|>', 128148), ('<|reserved_special_token_144|>', 128149), ('<|reserved_special_token_145|>', 128150), ('<|reserved_special_token_146|>', 128151), ('<|reserved_special_token_147|>', 128152), ('<|reserved_special_token_148|>', 128153), ('<|reserved_special_token_149|>', 128154), ('<|reserved_special_token_150|>', 128155), ('<|reserved_special_token_151|>', 128156), ('<|reserved_special_token_152|>', 128157), ('<|reserved_special_token_153|>', 128158), ('<|reserved_special_token_154|>', 128159), ('<|reserved_special_token_155|>', 128160), ('<|reserved_special_token_156|>', 128161), ('<|reserved_special_token_157|>', 128162), ('<|reserved_special_token_158|>', 128163), ('<|reserved_special_token_159|>', 128164), ('<|reserved_special_token_160|>', 128165), ('<|reserved_special_token_161|>', 128166), ('<|reserved_special_token_162|>', 128167), 
                ('<|reserved_special_token_163|>', 128168), ('<|reserved_special_token_164|>', 128169), ('<|reserved_special_token_165|>', 128170), ('<|reserved_special_token_166|>', 128171), ('<|reserved_special_token_167|>', 128172), ('<|reserved_special_token_168|>', 128173), ('<|reserved_special_token_169|>', 128174), ('<|reserved_special_token_170|>', 128175), ('<|reserved_special_token_171|>', 128176), ('<|reserved_special_token_172|>', 128177), ('<|reserved_special_token_173|>', 128178), ('<|reserved_special_token_174|>', 128179), ('<|reserved_special_token_175|>', 128180), ('<|reserved_special_token_176|>', 128181), ('<|reserved_special_token_177|>', 128182),
                ),
                (('<|reserved_special_token_190|>', 128195), ('<|reserved_special_token_191|>', 128196), ('<|reserved_special_token_192|>', 128197), ),
                (('<|reserved_special_token_193|>', 128198), ('<|reserved_special_token_194|>', 128199), ('<|reserved_special_token_195|>', 128200), ),
                (('<|reserved_special_token_196|>', 128201), ('<|reserved_special_token_197|>', 128202), ('<|reserved_special_token_198|>', 128203), ),
                (('<|reserved_special_token_199|>', 128204), ('<|reserved_special_token_200|>', 128205),),
                (('<|reserved_special_token_201|>', 128206), ('<|reserved_special_token_202|>', 128207), ),
                (('<|reserved_special_token_203|>', 128208), ('<|reserved_special_token_204|>', 128209), ),
                (('<|reserved_special_token_205|>', 128210), ('<|reserved_special_token_206|>', 128211),),
                (('<|reserved_special_token_207|>', 128212), ('<|reserved_special_token_208|>', 128213), ),
                (('<|reserved_special_token_176|>', 128181),('<|reserved_special_token_177|>', 128182),),
                (('<|reserved_special_token_209|>', 128214), ('<|reserved_special_token_210|>', 128215), ('<|reserved_special_token_211|>', 128216), ('<|reserved_special_token_212|>', 128217), ('<|reserved_special_token_213|>', 128218), ('<|reserved_special_token_214|>', 128219), ('<|reserved_special_token_215|>', 128220), ('<|reserved_special_token_216|>', 128221), ('<|reserved_special_token_217|>', 128222), ('<|reserved_special_token_218|>', 128223), ('<|reserved_special_token_219|>', 128224), ('<|reserved_special_token_220|>', 128225), ('<|reserved_special_token_221|>', 128226), ('<|reserved_special_token_222|>', 128227), ('<|reserved_special_token_223|>', 128228), ('<|reserved_special_token_224|>', 128229), ('<|reserved_special_token_225|>', 128230), ('<|reserved_special_token_226|>', 128231), ('<|reserved_special_token_227|>', 128232), ('<|reserved_special_token_228|>', 128233), ('<|reserved_special_token_229|>', 128234), ),
                (('<|reserved_special_token_230|>', 128235), ('<|reserved_special_token_231|>', 128236), ('<|reserved_special_token_232|>', 128237), ('<|reserved_special_token_233|>', 128238), ('<|reserved_special_token_234|>', 128239), ('<|reserved_special_token_235|>', 128240), ('<|reserved_special_token_236|>', 128241), ('<|reserved_special_token_237|>', 128242), ('<|reserved_special_token_238|>', 128243), ('<|reserved_special_token_239|>', 128244), ('<|reserved_special_token_240|>', 128245), ('<|reserved_special_token_241|>', 128246), ('<|reserved_special_token_242|>', 128247), ('<|reserved_special_token_243|>', 128248), ('<|reserved_special_token_244|>', 128249), ('<|reserved_special_token_245|>', 128250), ('<|reserved_special_token_246|>', 128251), ('<|reserved_special_token_247|>', 128252), ('<|reserved_special_token_248|>', 128253), ('<|reserved_special_token_249|>', 128254), ('<|reserved_special_token_250|>', 128255), ),
            ]
        elif tokenizer_type == "qwen2_vl":#151657  151657
            special_tokens = [
                [["<|reserved_special_token_180|>", 151837], ["<|reserved_special_token_181|>", 151838], ["<|reserved_special_token_182|>", 151839], ["<|reserved_special_token_183|>", 151840], ["<|reserved_special_token_184|>", 151841], ["<|reserved_special_token_185|>", 151842], ["<|reserved_special_token_186|>", 151843], ["<|reserved_special_token_187|>", 151844], ["<|reserved_special_token_188|>", 151845], ["<|reserved_special_token_189|>", 151846]], 
                [["<|reserved_special_token_190|>", 151847], ["<|reserved_special_token_191|>", 151848], ["<|reserved_special_token_192|>", 151849]], 
                [["<|reserved_special_token_193|>", 151850], ["<|reserved_special_token_194|>", 151851], ["<|reserved_special_token_195|>", 151852]], 
                [["<|reserved_special_token_196|>", 151853], ["<|reserved_special_token_197|>", 151854], ["<|reserved_special_token_198|>", 151855]], 
                [["<|reserved_special_token_199|>", 151856], ["<|reserved_special_token_200|>", 151857]],  # use
                [["<|reserved_special_token_201|>", 151858], ["<|reserved_special_token_202|>", 151859]],  # drop
                [["<|reserved_special_token_203|>", 151860], ["<|reserved_special_token_204|>", 151861]],  # attack
                [["<|reserved_special_token_205|>", 151862], ["<|reserved_special_token_206|>", 151863]],  # jump
                [["<|reserved_special_token_207|>", 151864], ["<|reserved_special_token_208|>", 151865],],  # camera
                [["<|reserved_special_token_176|>", 151833], ["<|reserved_special_token_177|>", 151834],],
                [["<|reserved_special_token_209|>", 151866], ["<|reserved_special_token_210|>", 151867], ["<|reserved_special_token_211|>", 151868], ["<|reserved_special_token_212|>", 151869], ["<|reserved_special_token_213|>", 151870], ["<|reserved_special_token_214|>", 151871], ["<|reserved_special_token_215|>", 151872], ["<|reserved_special_token_216|>", 151873], ["<|reserved_special_token_217|>", 151874], ["<|reserved_special_token_218|>", 151875], ["<|reserved_special_token_219|>", 151876], ["<|reserved_special_token_220|>", 151877], ["<|reserved_special_token_221|>", 151878], ["<|reserved_special_token_222|>", 151879], ["<|reserved_special_token_223|>", 151880], ["<|reserved_special_token_224|>", 151881], ["<|reserved_special_token_225|>", 151882], ["<|reserved_special_token_226|>", 151883], ["<|reserved_special_token_227|>", 151884], ["<|reserved_special_token_228|>", 151885], ["<|reserved_special_token_229|>", 151886]], 
                [["<|reserved_special_token_230|>", 151887], ["<|reserved_special_token_231|>", 151888], ["<|reserved_special_token_232|>", 151889], ["<|reserved_special_token_233|>", 151890], ["<|reserved_special_token_234|>", 151891], ["<|reserved_special_token_235|>", 151892], ["<|reserved_special_token_236|>", 151893], ["<|reserved_special_token_237|>", 151894], ["<|reserved_special_token_238|>", 151895], ["<|reserved_special_token_239|>", 151896], ["<|reserved_special_token_240|>", 151897], ["<|reserved_special_token_241|>", 151898], ["<|reserved_special_token_242|>", 151899], ["<|reserved_special_token_243|>", 151900], ["<|reserved_special_token_244|>", 151901], ["<|reserved_special_token_245|>", 151902], ["<|reserved_special_token_246|>", 151903], ["<|reserved_special_token_247|>", 151904], ["<|reserved_special_token_248|>", 151905], ["<|reserved_special_token_249|>", 151906], ["<|reserved_special_token_250|>", 151907]],
                
            ]
        else:
            raise ValueError(f"The tokenizer type {tokenizer_type} is not supported in control tokens.")
        try:
            token = special_tokens[place][num][not_text]
        except Exception as e:
            print("place:", place,"num:", num, "not_text:",not_text,e)
        return token

    def remap_control_token(self, token:str,use_num=True, tokenizer_type:str = "llama-2")->tuple:
        """由token映射到action，注意，虽然把camera从token中去掉，但是还需要它 """
        re_tokens = {}
        if tokenizer_type == "llama-2":
            if use_num:
                re_tokens = {31536: (0, 0), 31537: (0, 1), 31563: (0, 2), 31571: (0, 3), 31578: (0, 4), 31582: (0, 5), 31585: (0, 6), 31598: (0, 7), 
                            31603: (0, 8), 31604: (0, 9), 31000: (0, 10), 31001: (0, 11), 31002: (0, 12), 31003: (0, 13), 31004: (0, 14), 31005: (0, 15), 
                            31006: (0, 16), 31007: (0, 17), 31008: (0, 18), 31009: (0, 19), 31010: (0, 20), 31011: (0, 21), 31012: (0, 22), 31013: (0, 23), 
                            31014: (0, 24), 31015: (0, 25), 31016: (0, 26), 31017: (0, 27), 31018: (0, 28), 31019: (0, 29), 31020: (0, 30), 31021: (0, 31), 31022: (0, 32), 31023: (0, 33), 31024: (0, 34), 31025: (0, 35), 31026: (0, 36), 31027: (0, 37), 
                            31028: (0, 38), 31029: (0, 39), 31030: (0, 40), 31031: (0, 41), 31032: (0, 42), 31033: (0, 43), 31034: (0, 44), 31035: (0, 45), 31036: (0, 46), 31037: (0, 47), 31038: (0, 48), 31039: (0, 49), 31040: (0, 50), 31041: (0, 51), 
                            31042: (0, 52), 31043: (0, 53), 31044: (0, 54), 31045: (0, 55), 31046: (0, 56), 31047: (0, 57), 31048: (0, 58), 31049: (0, 59), 31050: (0, 60), 31051: (0, 61), 31052: (0, 62), 31053: (0, 63), 31054: (0, 64), 31055: (0, 65), 
                            31056: (0, 66), 31057: (0, 67), 31058: (0, 68), 31059: (0, 69), 31060: (0, 70), 31061: (0, 71), 31062: (0, 72), 31063: (0, 73), 31064: (0, 74), 31065: (0, 75), 31066: (0, 76), 31067: (0, 77), 31068: (0, 78), 31069: (0, 79), 
                            31070: (0, 80), 31071: (0, 81), 31072: (0, 82), 31073: (0, 83), 31074: (0, 84), 31075: (0, 85), 31076: (0, 86), 31077: (0, 87), 31078: (0, 88), 31079: (0, 89), 31080: (0, 90), 31081: (0, 91), 31082: (0, 92), 31083: (0, 93), 
                            31084: (0, 94), 31085: (0, 95), 31086: (0, 96), 31087: (0, 97), 31088: (0, 98), 31089: (0, 99), 31090: (0, 100), 31091: (0, 101), 31092: (0, 102), 31093: (0, 103), 31094: (0, 104), 31095: (0, 105), 31096: (0, 106), 31097: (0, 107), 
                            31098: (0, 108), 31099: (0, 109), 31100: (0, 110), 31101: (0, 111), 31102: (0, 112), 31103: (0, 113), 31104: (0, 114), 31105: (0, 115), 31106: (0, 116), 31107: (0, 117), 31108: (0, 118), 31109: (0, 119), 31110: (0, 120), 31111: (0, 121), 
                            31112: (0, 122), 31113: (0, 123), 31114: (0, 124), 31115: (0, 125), 31116: (0, 126), 31117: (0, 127), 31118: (0, 128), 31119: (0, 129), 31120: (0, 130), 31121: (0, 131), 31122: (0, 132), 31123: (0, 133), 31124: (0, 134), 31125: (0, 135), 31126: (0, 136), 31127: (0, 137), 31128: (0, 138), 31129: (0, 139), 31130: (0, 140), 31131: (0, 141), 31132: (0, 142), 31133: (0, 143), 31134: (0, 144), 31135: (0, 145), 31136: (0, 146), 31137: (0, 147), 31138: (0, 148), 31139: (0, 149), 
                            31140: (0, 150), 31141: (0, 151), 31142: (0, 152), 31143: (0, 153), 31144: (0, 154), 31145: (0, 155), 31146: (0, 156), 31147: (0, 157), 31148: (0, 158), 31149: (0, 159), 31150: (0, 160), 31151: (0, 161), 31152: (0, 162), 31153: (0, 163), 31154: (0, 164), 31155: (0, 165), 31156: (0, 166), 31157: (0, 167), 31158: (0, 168), 31159: (0, 169), 31160: (0, 170), 31161: (0, 171), 31162: (0, 172), 31163: (0, 173), 31164: (0, 174), 31165: (0, 175), 31166: (0, 176), 31167: (0, 177), 31168: (0, 178), 
                            31169: (0, 179), 31170: (0, 180), 31171: (0, 181), 31172: (0, 182), 31173: (0, 183), 31174: (0, 184), 31175: (0, 185), 31176: (0, 186), 31177: (0, 187), 31178: (0, 188), 31179: (0, 189), 31180: (0, 190), 31181: (0, 191), 31182: (0, 192), 31183: (0, 193), 31184: (0, 194), 31185: (0, 195), 31186: (0, 196), 31187: (0, 197), 31188: (0, 198), 31189: (0, 199), 31190: (0, 200), 31191: (0, 201), 31192: (0, 202), 31193: (0, 203), 31195: (0, 204), 31196: (0, 205), 31197: (0, 206), 31198: (0, 207), 
                            31199: (0, 208), 31200: (0, 209), 31201: (0, 210), 31202: (0, 211), 31203: (0, 212), 31204: (0, 213), 31205: (0, 214), 31206: (0, 215), 31207: (0, 216), 31208: (0, 217), 31209: (0, 218), 31210: (0, 219), 31211: (0, 220), 31212: (0, 221), 31213: (0, 222), 31214: (0, 223), 31215: (0, 224), 31216: (0, 225), 31217: (0, 226), 31218: (0, 227), 31219: (0, 228), 31220: (0, 229), 31221: (0, 230), 31222: (0, 231), 31223: (0, 232), 31224: (0, 233), 31225: (0, 234), 31226: (0, 235), 31227: (0, 236), 
                            31228: (0, 237), 31229: (0, 238), 31230: (0, 239), 31231: (0, 240), 31232: (0, 241), 31233: (0, 242), 31234: (0, 243), 31235: (0, 244), 31236: (0, 245), 31237: (0, 246), 31238: (0, 247), 31239: (0, 248), 31240: (0, 249), 31241: (0, 250), 31243: (0, 251), 31244: (0, 252), 31245: (0, 253), 31246: (0, 254), 31247: (0, 255), 31248: (0, 256), 31249: (0, 257), 31250: (0, 258), 31251: (0, 259), 31252: (0, 260), 31253: (0, 261), 31254: (0, 262), 31255: (0, 263), 31256: (0, 264), 31257: (0, 265), 
                            31258: (0, 266), 31259: (0, 267), 31260: (0, 268), 31261: (0, 269), 31262: (0, 270), 31263: (0, 271), 31264: (0, 272), 31265: (0, 273), 31266: (0, 274), 31267: (0, 275), 31268: (0, 276), 31269: (0, 277), 31270: (0, 278), 31271: (0, 279), 31272: (0, 280), 31273: (0, 281), 31274: (0, 282), 31275: (0, 283), 31276: (0, 284), 31277: (0, 285), 31278: (0, 286), 31279: (0, 287), 31280: (0, 288), 31281: (0, 289), 31282: (0, 290), 31283: (0, 291), 31284: (0, 292), 31285: (0, 293), 31286: (0, 294), 
                            31287: (0, 295), 31288: (0, 296), 31289: (0, 297), 31290: (0, 298), 31291: (0, 299), 31292: (0, 300), 31293: (0, 301), 31294: (0, 302), 31295: (0, 303), 31296: (0, 304), 31297: (0, 305), 31298: (0, 306), 31299: (0, 307), 31300: (0, 308), 31301: (0, 309), 31302: (0, 310), 31303: (0, 311), 31304: (0, 312), 31305: (0, 313), 31306: (0, 314), 31307: (0, 315), 31308: (0, 316), 31309: (0, 317), 31310: (0, 318), 31311: (0, 319), 31312: (0, 320), 31313: (0, 321), 31314: (0, 322), 31315: (0, 323), 
                            31316: (0, 324), 31317: (0, 325), 31318: (0, 326), 31319: (0, 327), 31320: (0, 328), 31321: (0, 329), 31322: (0, 330), 31323: (0, 331), 31324: (0, 332), 31325: (0, 333), 31326: (0, 334), 31327: (0, 335), 31328: (0, 336), 31329: (0, 337), 31330: (0, 338), 31331: (0, 339), 31332: (0, 340), 
                            31333: (0, 341), 31334: (0, 342), 31335: (0, 343), 31336: (0, 344), 31337: (0, 345), 31338: (0, 346), 31339: (0, 347), 31340: (0, 348), 31341: (0, 349), 31342: (0, 350), 31343: (0, 351), 31344: (0, 352), 31345: (0, 353), 31346: (0, 354), 31347: (0, 355), 31348: (0, 356), 31349: (0, 357), 31350: (0, 358), 31351: (0, 359), 31352: (0, 360), 31353: (0, 361), 31354: (0, 362), 31355: (0, 363), 31356: (0, 364), 31357: (0, 365), 31358: (0, 366), 31359: (0, 367), 31360: (0, 368), 31361: (0, 369), 
                            31362: (0, 370), 31363: (0, 371), 31364: (0, 372), 31365: (0, 373), 31366: (0, 374), 31367: (0, 375), 31368: (0, 376), 31369: (0, 377), 31370: (0, 378), 31371: (0, 379), 31372: (0, 380), 31373: (0, 381), 31374: (0, 382), 31375: (0, 383), 31376: (0, 384), 31377: (0, 385), 31378: (0, 386), 
                            31380: (0, 387), 31381: (0, 388), 31382: (0, 389), 31383: (0, 390), 31384: (0, 391), 31385: (0, 392), 31386: (0, 393), 31387: (0, 394), 31388: (0, 395), 31389: (0, 396), 31390: (0, 397), 31391: (0, 398), 31392: (0, 399), 31393: (0, 400), 31394: (0, 401), 31395: (0, 402), 31396: (0, 403), 31397: (0, 404), 31398: (0, 405), 31399: (0, 406), 31400: (0, 407), 31401: (0, 408), 31402: (0, 409), 31403: (0, 410), 31404: (0, 411), 
                            31405: (0, 412), 31406: (0, 413), 31407: (0, 414), 31408: (0, 415), 31409: (0, 416), 31410: (0, 417), 31411: (0, 418), 31412: (0, 419), 31413: (0, 420), 31414: (0, 421), 31415: (0, 422), 31416: (0, 423), 31417: (0, 424), 31418: (0, 425), 31419: (0, 426), 31420: (0, 427), 31421: (0, 428), 31422: (0, 429), 31423: (0, 430), 31424: (0, 431), 31425: (0, 432), 31426: (0, 433), 31427: (0, 434), 31428: (0, 435), 31429: (0, 436), 31430: (0, 437), 31431: (0, 438), 31432: (0, 439), 31433: (0, 440), 
                            31434: (0, 441), 31435: (0, 442), 31436: (0, 443), 31437: (0, 444), 31438: (0, 445), 31439: (0, 446), 31440: (0, 447), 31441: (0, 448), 31442: (0, 449), 31443: (0, 450), 31444: (0, 451), 31445: (0, 452), 31446: (0, 453), 31447: (0, 454), 31448: (0, 455), 31449: (0, 456), 31450: (0, 457), 
                            31451: (0, 458), 31452: (0, 459), 31453: (0, 460), 31454: (0, 461), 31455: (0, 462), 31456: (0, 463), 31457: (0, 464), 31458: (0, 465), 31459: (0, 466), 31460: (0, 467), 31461: (0, 468), 31462: (0, 469), 31463: (0, 470), 31464: (0, 471), 31465: (0, 472), 31466: (0, 473), 31467: (0, 474), 31468: (0, 475), 31469: (0, 476), 31470: (0, 477), 31471: (0, 478), 31472: (0, 479), 31473: (0, 480), 31474: (0, 481), 31475: (0, 482), 31476: (0, 483), 31477: (0, 484), 31478: (0, 485), 31479: (0, 486), 
                            31480: (0, 487), 31481: (0, 488), 31482: (0, 489), 31483: (0, 490), 31484: (0, 491), 31485: (0, 492), 31486: (0, 493), 31487: (0, 494), 31488: (0, 495), 31489: (0, 496), 31490: (0, 497), 31491: (0, 498), 31492: (0, 499), 31493: (0, 500), 31494: (0, 501), 31495: (0, 502), 31496: (0, 503), 31497: (0, 504), 31498: (0, 505), 31499: (0, 506), 31500: (0, 507), 31501: (0, 508), 31502: (0, 509), 31503: (0, 510), 31504: (0, 511)}
            else:
                re_tokens = {
                    '진': (0, 0),'জ': (0, 1),'천': (0, 2),'년': (0, 3),'세': (0, 4),'민': (0, 5),'ർ': (0, 6),'ἡ': (0, 7),'호': (0, 8),'ਰ': (0, 9),
                    '그': (1, 0),'න': (1, 1),'ན': (1, 2),
                    'ゆ': (2, 0),'ご': (2, 1),'현': (2, 2),
                    '군': (3, 0),'무': (3, 1),'위': (3, 2),
                    '안': (4, 0),'박': (4, 1),
                    '용': (5, 0),'단': (5, 1),
                    '면': (6, 0),'남': (6, 1),
                    '강': (7, 0),'씨': (7, 1),
                    '개': (8, 0),'들': (8, 1),
                    '차': (9, 0),'학': (9, 1),'만': (9, 2),'터': (9, 3),'식': (9, 4),'과': (9, 5),'타': (9, 6),'종': (9, 7),'내': (9, 8),'중': (9, 9),'방': (9, 10),
                    '월': (10, 0),'회': (10, 1),'모': (10, 2),'바': (10, 3),'음': (10, 4),'재': (10, 5),'명': (10, 6),'합': (10, 7),'역': (10, 8),'백': (10, 9),'왕': (10, 10)
                }
        elif tokenizer_type=="mistral":
            if use_num:
                re_tokens = {31947: (0, 0), 31948: (0, 1), 31949: (0, 2), 31950: (0, 3), 31951: (0, 4), 31952: (0, 5), 31953: (0, 6), 31955: (0, 7), 31956: (0, 8), 31957: (0, 9), 31958: (1, 0), 31959: (1, 1), 31960: (1, 2), 31961: (2, 0), 31962: (2, 1), 31963: (2, 2), 31964: (3, 0), 31965: (3, 1), 31966: (3, 2), 31967: (4, 0), 31896: (4, 1), 31969: (5, 0), 31970: (5, 1), 31971: (6, 0), 31972: (6, 1), 31973: (7, 0), 31974: (7, 1), 31976: (8, 0), 31977: (8, 1), 31978: (9, 0), 31979: (9, 1), 31980: (9, 2), 31981: (9, 3), 31982: (9, 4), 31983: (9, 5), 31984: (9, 6), 31985: (9, 7), 31986: (9, 8), 31987: (9, 9), 31988: (9, 10), 31926: (9, 11), 31927: (9, 12), 31929: (9, 13), 31930: (9, 14), 31931: (9, 15), 31932: (9, 16), 31933: (9, 17), 31934: (9, 18), 31935: (9, 19), 31936: (9, 20), 31989: (10, 0), 31990: (10, 1), 31991: (10, 2), 31992: (10, 3), 31993: (10, 4), 31994: (10, 5), 31995: (10, 6), 31996: (10, 7), 31997: (10, 8), 31998: (10, 9), 31999: (10, 10), 31937: (10, 11), 31938: (10, 12), 31939: (10, 13), 31940: (10, 14), 31920: (10, 15), 31921: (10, 16), 31943: (10, 17), 31944: (10, 18), 31945: (10, 19), 31946: (10, 20)}
            else:
                re_tokens = {
                    '朱': (0, 0),'ǝ': (0, 1),'Ḩ': (0, 2),'担': (0, 3),'灰': (0, 4),'讲': (0, 5),'롤': (0, 6),'😤': (0, 7),'ោ': (0, 8),'애': (0, 9),
                    '였': (1, 0),'질': (1, 1),'振': (1, 2),
                    '灯': (2, 0),'ĉ': (2, 1),'ස': (2, 2),
                    '閉': (3, 0),'램': (3, 1),'ಂ': (3, 2),
                    'げ': (4, 0),'ふ': (4, 1),
                    '狂': (5, 0),'融': (5, 1),
                    '仍': (6, 0),'實': (6, 1),
                    '楽': (7, 0),'範': (7, 1),
                    'వ': (8, 0),'嵌': (8, 1),
                    '摩': (9, 0),'袁': (9, 1),'ষ': (9, 2),'乎': (9, 3),'규': (9, 4),'岗': (9, 5),'糊': (9, 6),'క': (9, 7),'雲': (9, 8),'심': (9, 9),'ई': (9, 10),'庭': (9, 11), '苗': (9, 12), '闲': (9, 13), '독': (9, 14),'ɹ': (9, 15), 'ҽ': (9, 16), 'ថ': (9, 17), '宏': (9, 18), '尊': (9, 19),'總': (9, 20),
                    'འ': (10, 0),'ἡ': (10, 1),'丝': (10, 2),'Ħ': (10, 3),'伝': (10, 4),'컨': (10, 5),'အ': (10, 6),'執': (10, 7),'벨': (10, 8),'ゼ': (10, 9),'梦': (10, 10),'裝': (10, 11), 'ම': (10, 12), '▸': (10, 13), '測': (10, 14),'勇': (10, 15), '徐': (10, 16), '轩': (10, 17), '兄': (10, 18), '剑': (10, 19),'ન': (10, 20)
                }
        elif tokenizer_type=="llama-3":
            if use_num:
                re_tokens={128185: (0, 0), 128186: (0, 1), 128187: (0, 2), 128188: (0, 3), 128189: (0, 4), 128190: (0, 5), 128191: (0, 6), 128192: (0, 7), 128193: (0, 8), 128194: (0, 9), 128195: (1, 0), 128196: (1, 1), 128197: (1, 2), 128198: (2, 0), 128199: (2, 1), 128200: (2, 2), 128201: (3, 0), 128202: (3, 1), 128203: (3, 2), 128204: (4, 0), 128205: (4, 1), 128206: (5, 0), 128207: (5, 1), 128208: (6, 0), 128209: (6, 1), 128210: (7, 0), 128211: (7, 1), 128212: (8, 0), 128213: (8, 1), 128181: (9, 0),128182: (9, 1),    
                    128214: (10, 0), 128215: (10, 1), 128216: (10, 2), 128217: (10, 3), 128218: (10, 4),
                    128219: (10, 5), 128220: (10, 6), 128221: (10, 7), 128222: (10, 8), 128223: (10, 9),
                    128224: (10, 10), 128225: (10, 11), 128226: (10, 12), 128227: (10, 13), 128228: (10, 14),
                    128229: (10, 15), 128230: (10, 16), 128231: (10, 17), 128232: (10, 18), 128233: (10, 19),
                    128234: (10, 20), 128235: (11, 0), 128236: (11, 1), 128237: (11, 2), 128238: (11, 3),
                    128239: (11, 4), 128240: (11, 5), 128241: (11, 6), 128242: (11, 7), 128243: (11, 8),
                    128244: (11, 9), 128245: (11, 10), 128246: (11, 11), 128247: (11, 12), 128248: (11, 13),
                    128249: (11, 14), 128250: (11, 15), 128251: (11, 16), 128252: (11, 17), 128253: (11, 18),
                    128254: (11, 19), 128255: (11, 20),}
            else:
                raise ValueError("can't use text as tokens")
        elif tokenizer_type == "qwen2_vl":
            if use_num:
                re_tokens={151837: [0, 0], 151838: [0, 1], 151839: [0, 2], 151840: [0, 3], 151841: [0, 4], 151842: [0, 5], 151843: [0, 6], 151844: [0, 7], 151845: [0, 8], 151846: [0, 9], 151847: [1, 0], 151848: [1, 1], 151849: [1, 2], 151850: [2, 0], 151851: [2, 1], 151852: [2, 2], 151853: [3, 0], 151854: [3, 1], 151855: [3, 2], 151856: [4, 0], 151857: [4, 1], 151858: [5, 0], 151859: [5, 1], 151860: [6, 0], 151861: [6, 1], 151862: [7, 0], 151863: [7, 1], 151864: [8, 0], 151865: [8, 1], 
                    151833: (9, 0),151834: (9, 1),
                    151866: [10, 0], 151867: [10, 1], 151868: [10, 2], 151869: [10, 3], 151870: [10, 4],
                    151871: [10, 5], 151872: [10, 6], 151873: [10, 7], 151874: [10, 8], 151875: [10, 9],
                    151876: [10, 10], 151877: [10, 11], 151878: [10, 12], 151879: [10, 13], 151880: [10, 14],
                    151881: [10, 15], 151882: [10, 16], 151883: [10, 17], 151884: [10, 18], 151885: [10, 19],
                    151886: [10, 20], 151887: [11, 0], 151888: [11, 1], 151889: [11, 2], 151890: [11, 3],
                    151891: [11, 4], 151892: [11, 5], 151893: [11, 6], 151894: [11, 7], 151895: [11, 8],
                    151896: [11, 9], 151897: [11, 10], 151898: [11, 11], 151899: [11, 12], 151900: [11, 13],
                    151901: [11, 14], 151902: [11, 15], 151903: [11, 16], 151904: [11, 17], 151905: [11, 18],
                    151906: [11, 19], 151907: [11, 20],
                }
            else:
                raise ValueError("can't use text as tokens")
        else:
            raise ValueError(f"The tokenizer type {tokenizer_type} is not supported in control tokens.")
        return re_tokens.get(token,(-1,-1))

    def tag_token(self, place, tokenizer_type:str = "llama-2",return_type:int=0):
        """引入头标记和尾标记 """
        assert place in {0,1}
        if tokenizer_type == "llama-2":
            special_tokens = [('유', 31533),('요', 31527)]
        elif tokenizer_type == "mistral":
            special_tokens = [('ಮ', 31941),('አ', 31942)]
        elif tokenizer_type=="llama-3":
            special_tokens = [('<|reserved_special_token_178|>', 128183), ('<|reserved_special_token_179|>', 128184),]
        elif tokenizer_type=="qwen2_vl":
            special_tokens = [('<|reserved_special_token_178|>', 151835), ('<|reserved_special_token_179|>', 151836),]
        else:
            raise ValueError(f"The tokenizer type {tokenizer_type} is not supported in control tokens.")
        return special_tokens[place][return_type]

class OneActionTokenizer(ReservedActionTokenizer):
    BUTTONS_GROUPS = [
        "hotbar", "fore or back", "left or right", "sprint or sneak", "use", "drop", "attack", "jump", "camera"
    ]
    
    def __init__(self,tokenizer_type="llama-2",
                 bases=[10,3,3,3,2,2,2,2,2,2,21,21],
                 camera_quantization_scheme:str="mu_law",
                 camera_mu:float=20,
                 camera_binsize:int=1,
                 action_chunk_len:int=1,
                 sliding_window_len:Optional[int] = None,
                 keep_no_op_p:float = 1.0,
                 **kwargs,
                 ):
        super().__init__(tokenizer_type=tokenizer_type,
                         camera_quantization_scheme=camera_quantization_scheme,
                         camera_mu=camera_mu,
                         camera_binsize=camera_binsize,
                         keep_no_op_p=keep_no_op_p,)
        console.Console().log(f"tokenizer_type: {tokenizer_type}")
        console.Console().log(f"bases: {bases}, camera_mu: {camera_mu}, n_camera_bins: {self.n_camera_bins}, camera_binsize: {camera_binsize}")
        self.bases = bases
        self.action_chunk_len = action_chunk_len
        self.sliding_window_len = action_chunk_len if sliding_window_len is None else sliding_window_len
        self.NULL_ACTION = [0,(bases[-2]//2)*bases[-2]+(bases[-1]//2)]        
    
    def decode(self,tokens:Union[torch.Tensor,List]):
        """decode the tokens to action
        """
        group_actions = self.token_2_group_action(tokens,)
        
        actions = [self.group_action_2_decimal_action(group_action) for group_action in group_actions ]
        action_dicts = []
        for action in  actions:
            action_dict = {
                "buttons":np.array([action[0]]),
                "camera":np.array([action[1]]),  #返回一个工作
            }
            action_dict = OrderedDict({key: value[0] for key, value in action_dict.items()})
            action_dicts.append(action_dict)
        
        action_dicts = action_dicts[-self.action_chunk_len:]
        if not action_dicts:
            action_dicts = [copy.deepcopy(self.NULL_ACTION)]
        
        return action_dicts
    
    def encode(self,trajectory:dict,**kwargs) -> list[tuple[int]]:
        """encode an action to tokens
        action: tuple -- (button,camera)
        output: str, 多个token
        """
        encoded_trajectory = []
        
        actions = trajectory.get("actions", [])
        null_action = self.env_null_action
        
        if "json_actions" in trajectory:
            actions = [ json_action_to_env_action(action) for action in trajectory["json_actions"]]
        
        traj_len = len(actions) 
        if not traj_len:
            return encoded_trajectory
        
        observations = trajectory.get('observations',[""]*traj_len)
        frame_ids = trajectory.get('frame_ids',list(range(traj_len)))
        frame_idx_real_idx_map = {frame_idx:idx for idx,frame_idx in enumerate(frame_ids)} #要跟在frame_ids后面
        if not _equal(self.keep_no_op_p,1.0): #假如需要去掉一些keep_no_op_p
            new_frame_ids = []
            for adx, action in enumerate(actions):
                if self.is_same_env_action(null_action,action) \
                    and random.random() > self.keep_no_op_p:
                    pass
                else:
                    new_frame_ids.append(frame_ids[adx])
            frame_ids = new_frame_ids
        
        interact_ids = trajectory.get('interact_ids',[""]*traj_len)
        minerl_actions = merge_dict(trajectory['actions'])
        encode_minerl_actions = self.action_transformer.env2policy(minerl_actions) 
        actions = self.action_mapper.from_factored(encode_minerl_actions)
        
        action_list = []
        for idx in range(traj_len):
            action_list.append((actions["buttons"][idx][0],actions["camera"][idx][0]))
        encoded_trajectory = []
        
        chunks = get_non_markov_sequences(frame_ids, chunk_len=self.action_chunk_len,sliding_window_len=self.sliding_window_len)
        for chunk in chunks:
            encoded_raw_actions = [action_list[frame_idx_real_idx_map[i]] for i in chunk]
            raw_actions = [trajectory['actions'][frame_idx_real_idx_map[i]] for i in chunk ]
            encoded_action = " ".join(
                    [self.encode_action(raw_action) for raw_action in encoded_raw_actions]
                )
            encoded_trajectory.append({
                "action":       encoded_action,
                "raw_action":   raw_actions,
                "observations": [observations[frame_idx_real_idx_map[i]] for i in chunk],
                "interact_id":  interact_ids[frame_idx_real_idx_map[chunk[0]]],
                "frames":       (frame_ids[frame_idx_real_idx_map[chunk[0]]], len(chunk), frame_ids[frame_idx_real_idx_map[chunk[-1]]]),
            })
                
        return encoded_trajectory
    
    def encode_action(self,action:tuple)->str:
        """encode an action to tokens
        action: tuple -- (button,camera)
        output: str, 多个token
        """
        assert len(action)==2
        # map to groups of action 
        group_action = self.decimal_action_2_group_action(action)
        # from action to token
        tokens = self.group_action_2_token(group_action)
        
        return tokens
    
    def group_action_2_token(self,group_action):
        zero_include_token_list = [self.map_control_token(num, i, self.tokenizer_type) for i, num in enumerate(group_action)]
        control_token = ''.join((s for x,s in zip(group_action[:-4],zero_include_token_list[:-4]) if x != 0 )) #camera键在这里没有意义
        control_token = control_token + "".join((s for s in zero_include_token_list[-2:]))  #camera必须保存
        tag_control_token = self.act_beg_token + control_token + self.act_end_token
        return tag_control_token
    
    def token_2_group_action(self,tokens:Union[torch.Tensor,list]):
        actions = []
        action_base = [0]*len(self.bases) #初始化
        camera_null = [self.bases[-1]//2,self.bases[-2]//2]
        action_base[-2:] = camera_null

        if isinstance(tokens, torch.Tensor):
            # 如果是二维张量 (shape == 2)，则需要 squeeze
            if tokens.ndim == 2:
                tokens = tokens.squeeze()
            tokens = tokens.tolist()
        elif not isinstance(tokens, list):
            raise ValueError("wrong type!")

        start_idx = 0
        while start_idx < len(tokens):
            try:
                first_index_n1 = tokens.index(self.act_beg_id, start_idx)
                first_index_n2 = tokens.index(self.act_end_id, first_index_n1 + 1)
            except ValueError:
                break
            
            control_tokens = tokens[first_index_n1 + 1:first_index_n2]
            action = copy.copy(action_base)
            # 对 control_tokens 中每个 token 进行 remap 并更新 action

            for token in control_tokens:
                place, num = self.remap_control_token(token, use_num=True, tokenizer_type=self.tokenizer_type)
                if place != -1:
                    action[place] = num
            
            if action[-2:] != camera_null:
                action[-4] = 1
            
            actions.append(copy.copy(action))
            start_idx = first_index_n2 + 1

        if len(actions)  == 0:
            actions.append(action_base)

        return actions

    def decimal_action_2_group_action(self,inputs):
        """ 
        Params:
        * output: set, 返回一个元组, 元组中的每个元素表示一个位的值
        * inputs: tuple, 两个十进制整数
        * bases: list, 每位的基数     

        Function: 将一个十进制整数转换为具有不同基数的数字系统(每位的基数分别为 [8, 8, 8, 6, 5]), 需要编写一个Python函数来执行逆向计算。这个转换涉及将十进制数逐位除以对应的基数并取余数, 然后再继续处理商。
        """
        decimals = list(inputs)
        result = [0] * len(self.bases)
        inventory_flag = False
        if decimals[0]==8640:
            inventory_flag = True
            decimals[0] = 0
        else:
            # 用于存储转换结果的列表
            # 从最低位到最高位逐位计算
            for i in range(len(self.bases)-4, -1, -1):
                # 求当前位的值
                result[i] = decimals[0] % self.bases[i]
                # 更新十进制数为下一位的处理
                decimals[0] //= self.bases[i]
            # 确保转换过程中十进制数被完全转换
        
        result[-1] = decimals[1] % self.bases[-1]
        decimals[1] //= self.bases[-1]
        result[-2] = decimals[1] % self.bases[-2]
        decimals[1] //= self.bases[-2]

        if inventory_flag:
            result[-3] = 1
        if decimals != [0,0]:
            print(decimals)
            raise ValueError("The decimal number is too large for the custom base system.")
        return tuple(result)
    
    def group_action_2_decimal_action(self,inputs):
        """ 
        Function: 将一个具有不同基数的数字系统(每位的基数分别为 [8, 8, 8, 6, 5])转换为十进制整数, 需要编写一个Python函数来执行逆向计算。这个转换涉及将每位的值乘以对应的基数的幂, 然后再求和。
        :output: int, 十进制整数
        :number_tuple: tuple, 每位的值
        :bases: list, 每位的基数
        """
        # 确保输入的长度与基数匹配
        if len(inputs) != len(self.bases):
            raise ValueError("The input number does not match the expected number of digits.")
        # 初始化十进制结果
        decimal_results = [0,0]
        # 计算十进制值
        mid = len(inputs)-3
        for i, digit in enumerate(inputs):
            if digit >= self.bases[i]:
                raise ValueError(f"Digit at position {i} exceeds the base limit of {self.bases[i]-1}.")
            if i < mid:
                decimal_results[0] = decimal_results[0] * self.bases[i] + digit
            elif i == mid and digit:
                decimal_results[0] = 8640
            else:
                decimal_results[1] = decimal_results[1] * self.bases[i] + digit
        return tuple(decimal_results)
    
    def encode_null_action(self):
        return self.encode_action(self.NULL_ACTION)

class BPEActionTokenizer(ReservedActionTokenizer):

    SAVE_CHECKPOINT_INTERVAL = 1000
    LOG_INTERVAL = 20
    
    
    def __init__(self, checkpoint_path= Path("ultron/model/inference/checkpoints/bpe_model_512.pkl"), 
                 train=False,
                 documents_dir =None, 
                 vocab_size: int=512,
                 tokenizer_type="llama-2",
                 camera_quantization_scheme="mu_law",
                 camera_mu=10,
                 camera_binsize=2
                ):
        
        
        super().__init__(tokenizer_type=tokenizer_type,
            camera_quantization_scheme=camera_quantization_scheme,
            camera_mu=camera_mu,
            camera_binsize=camera_binsize)
        
        self.checkpoint_path=checkpoint_path
        self.camera_null_number = (self.action_transformer.discretize_camera(np.array([0.])) * (self.n_camera_bins + 1)).item()
        
        self.initial_vocab_size = 2 ** (len(self.movements)) + 9 + 2 ** len(self.operations) + 1 + self.n_camera_bins ** 2
        self.vocab_size = vocab_size
        
        self.documents = []
        self.total_len = 0      # length of all documents
        
        if train:
            self.initialize_tokenizer(documents_dir)
        else:
            self.load_from_checkpoint()

    def load_from_checkpoint(self,):
        with open(self.checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
            
            self.total_len = checkpoint["total_len"]
            self.idx = checkpoint["idx"]
            self.bpe_rules = checkpoint["bpe_rules"]
            self.reverse_bpe_rules = checkpoint["reverse_bpe_rules"]
            self.frequency = checkpoint["frequency"]
            self.log = checkpoint["log"]
    
    def save_to_checkpoint(self,):
        checkpoint = {
            'total_len': self.total_len,
            'idx': self.idx,
            'bpe_rules': self.bpe_rules,
            'reverse_bpe_rules': self.reverse_bpe_rules,
            'frequency': self.frequency,
            'log': self.log
        }
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"Checkpoint saved at {self.checkpoint_path}")
    
    def initialize_tokenizer(self, documents_dir):
        self.documents = []
        self.total_len = 0
        self.initial_vocab_size = 2 ** (len(self.movements)) + 9 + 2 ** len(self.operations) + 1 + self.n_camera_bins ** 2
        self.idx = copy.copy(self.initial_vocab_size)
        self.bpe_rules = []
        self.reverse_bpe_rules = {}
        for i in range(self.initial_vocab_size):
            self.reverse_bpe_rules[i] = (i,)
        self.frequency = {}
        
        for document in tqdm(documents_dir.rglob('*.jsonl'), desc="Processing documents"):
            my_document = []
            with open(document, 'rb') as f:
                for line in f:
                    actions = json.loads(line)['actions']
                    traj_len = len(actions['attack'])
                    for i in range(traj_len):
                        action = {key: val[i] for key, val in actions.items()}
                        my_document.extend(self.action2int(action))
            self.documents.append(my_document)
            self.total_len += len(my_document)
        
        for document in tqdm(self.documents, desc="Calculating frequency"):
            for i in range(len(document) - 1):
                pair = (document[i], document[i+1])
                self.frequency[pair] = self.frequency.get(pair, 0) + 1

        self.log = {'vocab_size': [self.initial_vocab_size], 'total_len': [self.total_len]}
    
    def merge(self, pair: tuple, new_token: int):
        new_documents = []
        for document in self.documents:
            new_document = []
            i = 0
            while i < len(document):
                if i < len(document) - 1 and (document[i], document[i+1]) == pair:
                    new_document.append(new_token)
                    self.total_len -= 1
                    # Update frequency
                    if i > 0:
                        self.frequency[(document[i-1], new_token)] = self.frequency.get((document[i-1], new_token), 0) + 1
                    if i < len(document) - 2:
                        self.frequency[(new_token, document[i+2])] = self.frequency.get((new_token, document[i+2]), 0) + 1
                    i += 2
                else:
                    new_document.append(document[i])
                    i += 1
            new_documents.append(new_document)
        self.frequency.pop(pair)
        self.documents = new_documents

    def train(self):
        print(f"Initial vocab size: {self.initial_vocab_size}")
        with tqdm(total=self.vocab_size - self.idx, desc="Training BPE") as pbar:
            while self.idx < self.vocab_size:
                most_frequent_pair = max(self.frequency, key=self.frequency.get)
                self.bpe_rules.append((most_frequent_pair, self.idx))
                self.reverse_bpe_rules[self.idx] = self.reverse_bpe_rules[most_frequent_pair[0]] + self.reverse_bpe_rules[most_frequent_pair[1]]
                self.merge(most_frequent_pair, self.idx)
                self.idx += 1
                pbar.update(1)
                if self.idx % self.SAVE_CHECKPOINT_INTERVAL == 0:
                    with open(checkpoints_dir / f'bpe_checkpoint_{self.idx}.pkl', 'wb') as f:
                        pickle.dump(self, f)
                if self.idx % self.LOG_INTERVAL == 0:
                    self.log['vocab_size'].append(self.idx)
                    self.log['total_len'].append(self.total_len)
        self.save_to_checkpoint()
        print(f"Compress rate is {self.total_len / self.log['total_len'][0]}")

    def encode(self,trajectory:list) -> list[tuple[int]]:
        """将trajectory映射到tokens
        输入 trajectory，必须包含actions，image_path，和这一步的id
        输出 包含action_tokens，image_paths,起始步的id
        """
        actions = trajectory['actions']
        traj_len = len(actions['attack'])
        observations = trajectory.get('observations',[""]*traj_len)
        interact_ids = trajectory.get('interact_ids',[""]*traj_len)
        interact_ids = self.actions2ints(actions)
        encoded_trajectory = []
        
        last_action_id = 0
        last_action_start_index = 0
        current_index = 0
        while current_index < len(interact_ids):
            _, action_id = interact_ids[current_index]
            if last_action_id != action_id:
                control_token = self.interact_ids2token([interact_ids[idx][0] for idx in range(last_action_start_index, current_index)])
                encoded_trajectory.append([control_token, observations[last_action_start_index:current_index], interact_ids[last_action_start_index]])
                last_action_id = action_id
                last_action_start_index = current_index
            current_index += 1
        
        control_token = self.interact_ids2token([interact_ids[idx][0] for idx in range(last_action_start_index,  len(interact_ids))])
        encoded_trajectory.append([control_token, observations[last_action_start_index:len(interact_ids)], interact_ids[last_action_start_index]])
        return encoded_trajectory
    
    def decode(self, tokens: Union[torch.Tensor,list]) -> list[dict]:
        '''
        decode a chunk of action token to a list of actions
        '''
        interact_ids = self.token2action_idx(tokens)
        actions = []
        for action_idx in interact_ids:
            if action_idx==-1:
                continue
            actions.extend([self.int2action(t) for t in self.reverse_bpe_rules[action_idx]])
        new_actions = []
        for action in actions:
            minerl_action_transformed = {key: np.array([val]) for key, val in action.items()}
            minerl_action = self.action_transformer.env2policy(minerl_action_transformed) 
            new_action = self.action_mapper.from_factored(minerl_action)
            new_action = {key: val[0] for key, val in new_action.items()}
            new_actions.append(new_action)
        return new_actions 

    def actions2ints(self,actions:dict)-> tuple[int]:
        traj_len = len(actions['attack'])
        old_document = []
        for i in range(traj_len):
            action = {key: val[i] for key, val in actions.items()}
            result = self.action2int(action)
            old_document.extend((token, i) for token in result)
        
        for pair, new_token in self.bpe_rules:
            new_document = []
            i = 0
            while i < len(old_document):
                if i < len(old_document) - 1 and (old_document[i][0], old_document[i+1][0]) == pair:
                    new_document.append((new_token, old_document[i][1]))
                    i += 2
                else:
                    new_document.append(old_document[i])
                    i += 1
            old_document = new_document
        return old_document

    def action2int(self,action: dict)-> tuple[int]:
        result = []
        mov_num = 0
        for i, mov in enumerate(self.movements):
            if action[mov]:
                mov_num += 2 ** i
        if mov_num > 0:
            result.append(mov_num)

        hotbar_num = 0
        for i in range(9):
            if action[f'hotbar.{i+1}']:
                hotbar_num = i
                break
        if hotbar_num > 0:
            result.append(hotbar_num + 2 ** len(self.movements))

        op_num = 0
        for i, op in enumerate(self.operations):
            if action[op]:
                op_num += 2 ** i
        if op_num > 0:
            result.append(op_num + 2 ** (len(self.movements)) + 9)

        if action['inventory']:
            result.append(2 ** (len(self.movements)) + 9 + 2 ** len(self.operations))

        camera_x = self.action_transformer.discretize_camera(np.array([action['camera'][0]]))
        camera_y = self.action_transformer.discretize_camera(np.array([action['camera'][1]]))
        camera_num = (camera_x * self.n_camera_bins + camera_y).item()
        if camera_num != self.camera_null_number:
            result.append(camera_num + 2 ** (len(self.movements)) + 9 + 2 ** len(self.operations) + 1)

        return tuple(result)
        
    def _null_token(self):
        return self.interact_ids2token([0])

    def int2action(self,token: int) -> dict:
        action = {mov: False for mov in self.movements}
        action.update({f'hotbar.{i+1}': False for i in range(9)})
        action.update({op: False for op in self.operations})
        action['inventory'] = False
        action['camera'] = (0., 0.)

        if token < 2 ** len(self.movements):
            for i, mov in enumerate(self.movements):
                if token & 2 ** i:
                    action[mov] = True
        elif token < 2 ** (len(self.movements)) + 9:
            hotbar_num = token - 2 ** len(self.movements)
            action[f'hotbar.{hotbar_num + 1}'] = True
        elif token < 2 ** (len(self.movements)) + 9 + 2 ** len(self.operations):
            op_num = token - 2 ** (len(self.movements)) - 9
            for i, op in enumerate(self.operations):
                if op_num & 2 ** i:
                    action[op] = True
        elif token < 2 ** (len(self.movements)) + 9 + 2 ** len(self.operations) + 1:
            action['inventory'] = True
        elif token < 2 ** (len(self.movements)) + 9 + 2 ** len(self.operations) + 1 + self.n_camera_bins ** 2:
            camera_num = token - 2 ** (len(self.movements)) - 9 - 2 ** len(self.operations) - 1
            camera_x = camera_num // self.n_camera_bins
            camera_y = camera_num % self.n_camera_bins
            action['camera'] = (self.action_transformer.undiscretize_camera(camera_x).item(), self.action_transformer.undiscretize_camera(camera_y).item())
        else:
            raise ValueError(f"Invalid token {token}")
        return action

    def interact_ids2token(self,interact_ids:list)->str:
        zero_include_token_list = [self.map_control_token( action_idx, 0,self.tokenizer_type) for action_idx in interact_ids]
        control_token = "".join((s for s in zero_include_token_list))  #camera必须保存
        tag_control_token = self.act_beg_token + control_token + self.act_end_token
        return tag_control_token
    
    def token2action_idx(self,tokens:Union[torch.Tensor,list]):
        if isinstance(tokens, torch.Tensor):
            # 如果是二维张量 (shape == 2)，则需要 squeeze
            if tokens.ndim == 2:
                tokens = tokens.squeeze()
            tokens = tokens.tolist()
        elif not isinstance(tokens, list):
            raise ValueError("wrong type!")
        interact_ids = []
        try:
            first_index_n1 = tokens.index(self.act_beg_id, 0)
            first_index_n2 = tokens.index(self.act_end_id, first_index_n1 + 1)
            control_tokens = tokens[first_index_n1 + 1:first_index_n2]
            for token in control_tokens:
                print(token)
                _, num = self.remap_control_token(token, use_num=True, tokenizer_type=self.tokenizer_type)
                interact_ids.append(num)
            return interact_ids
        except ValueError:
            return [0]
         
def make_action_tokenizer(name:str):
    checkpoints_dir = Path('checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)
    with open(checkpoints_dir / name, 'rb') as f:
         loaded_bpe = pickle.load(f)
    return loaded_bpe

def create_point_prompt(point_type:str, points:list, labels:str=""):
    text = ""
    
    if "qwen2_vl" in point_type or 'llama2' in point_type:
        point_text = ""
        for point in points:
            x = 99.9 if point[0] >= 100 else point[0]
            y = 99.9 if point[1] >= 100 else point[1]
            output_point = [int(x * 10),int(y * 10)]
            point_text += f"({output_point[0]},{output_point[1]}),"
        point_text = point_text.rstrip(',')
        if labels:
            text += "<|object_ref_start|>{labels}<|object_ref_end|>".format(labels=labels)
        text += "<|point_start|>{point}<|point_end|>".format(point=point_text)
    elif "kimi_vl" in point_type:
        point_text = ""
        for point in points:
            x = 99.9 if point[0] >= 100 else point[0]
            y = 99.9 if point[1] >= 100 else point[1]
            output_point = [round(x/100,3),round(y/100,3)]
            point_text += f"(x={output_point[0]},y={output_point[1]}),"
        point_text = point_text.rstrip(',')
        if labels:
            text += "<|object_ref_start|>{labels}<|object_ref_end|>".format(labels=labels)
        text += "<|point_start|>{point}<|point_end|>".format(point=point_text)
    else:
        raise ValueError(f"[red]{point_type} is not support for pointing")
    return text


if __name__ == '__main__':
    
    # tokenizer = MotionTokenizer()
    # from ultron.utils import file_utils
    # from collections import Counter
    # encoded = tokenizer.encode(
    #     trajectory={"actions":file_utils.load_jsonl("/DATA/datasets/contractors/10xx/env_actions/woozy-ruby-ostrich-f153ac423f61-20220420-083502.jsonl")}
    #     )
    # file_utils.dump_jsonl(encoded,file_path="/home/lmy/workspace/JARVIS/ultron/visualize/dha/outputs/encoded.jsonl")
    # exit()
    
    
    
    # file = file_utils.load_jsonl("/DATA/datasets/minecraft-trajectory.cache/mc-motion-action-samples_v1-c1i1w1-0713-train.jsonl")
    # new_file = list(set([entity["conversations"][1]["content"][0]["text"] for entity in file]))
    
    # file_utils.dump_jsonl(new_file,"ultron/visualize/dha/assets/motion_type.jsonl")
    # exit()
    
    action_tokenizer = TextActionTokenizer(action_chunk_len=3,sliding_window_len=1)
    a = action_tokenizer.json_null_action
    b = action_tokenizer.json_null_action
    c = action_tokenizer.is_same_json_action(a,b)
    print(c)
    exit()
    inputs = "Action: move('3.0', '0.0') and click('left') \n Action: move('39.0', '6.0') and click('left') \n Action: move('26.0', '2.0') and click('left') \n"
    a = action_tokenizer.decode(tokens=inputs)
    e = action_tokenizer.encode(trajectory={"actions":a,"observations":range(len(a))})
    from ultron.utils import file_utils
    from ultron.dataset.interaction.create_datasets import flatten_dict
    from rich import print
    actions = file_utils.load_jsonl("/DATA/datasets/contractors/7xx/env_actions/woozy-ruby-ostrich-f3ec8c12f786-20220203-023311.jsonl")
    encode_actions = action_tokenizer.encode(trajectory={"actions":actions,"observations":range(len(actions))})
    print([encode_action["action"] for encode_action in encode_actions])
    exit()
    
    #print(prepare_for_remap_control_token(bases=[512,3,3,3,2,2,2,2,2,11,11]))
    #import pdb; pdb.set_trace()
    #exit()
    #documents_dir = Path('/public/share/craft-shell_agent-12_05')
    #heckpoints_dir = Path('checkpoints')
    #checkpoints_dir.mkdir(exist_ok=True)

    #vocab_size = 512
    #bpe = BPEActionTokenizer(documents_dir=documents_dir, vocab_size=vocab_size,train=True,checkpoint_path=checkpoints_dir / 'bpe_model_512.pkl',)
    #bpe.train()

    #Save the BPE model
    #with open(checkpoints_dir / f'bpe_model_{vocab_size}.pkl', 'wb') as f:
        #pickle.dump(bpe, f)

    # # plot log info
    # plt.plot(bpe.log['vocab_size'], bpe.log['total_len'])
    # plt.xlabel('Vocabulary size')
    # plt.ylabel('Total length of documents')
    # plt.title('BPE training process')
    # plt.savefig('bpe_training.png')

    # Load the BPE model and encode a file
    #loaded_bpe = BPEActionTokenizer()

    #A = {
    #    "actions":{'forward': [True], 'back': [False,], 'left': [False,], 'right': [True,], 'sprint': [False,], 'sneak': [False,], 'hotbar.1': [True,], 'hotbar.2': [False,], 'hotbar.3': [False,], 'hotbar.4': [False,], 'hotbar.5': [False,], 'hotbar.6': [False,], 'hotbar.7': [False,], 'hotbar.8': [False,], 'hotbar.9': [False,], 'use': [False,], 'drop': [True,], 'attack': [False,], 'jump': [False,], 'inventory': [False,], 'camera': [(1, 0.0)]}
    #}
    #print(loaded_bpe.encode(A))#utils.load_jsonl(documents_dir / 'craft_craft_table/train/009b0681-9120-4eec-b54b-1e2f23f30885.jsonl')[0]))
    #print(loaded_bpe.decode([1,29871,31533,30705, 31527,13,376, 29913,31472,31197,]))

    #action_map = ActionTokenizer("llama-2")
    
    
    
    #print(action_map.decode(tokens=torch.tensor([128183,128202,128219,128240,128184])))
    #print(action_map.encode([0]))
    
    processor_config = dict(
        do_rescale=False,
        patch_size=14,
        vision_feature_select_strategy="default"
    )
    from transformers import Qwen2VLProcessor,Qwen2VLForConditionalGeneration
    model_path = "/public/models/qwen2-vl-2b-instruct"
    processor = Qwen2VLProcessor.from_pretrained(model_path,**processor_config)
    
    #with open("ultron/model/assets/special_token.json", "r") as file:
    #    special_token = json.load(file)
    #processor.tokenizer.add_special_tokens({"additional_special_tokens":special_token})
    #print(processor.tokenizer("<|reserved_special_token_178|><|reserved_special_token_204|><|reserved_special_token_221|><|reserved_special_token_239|><|reserved_special_token_179|>"))

    #print(model)
    import  numpy as np
    tokenizer = OneActionTokenizer("qwen2_vl")
    #actions = tokenizer.decode(tokens=[151835, 151863, 151878, 151896, 151836])
    #actions = actions[0]
    #print(actions)
    print(tokenizer.encode({'actions':action}))
    #actions = tokenizer.action_mapper.to_factored(actions)
    #actions = tokenizer.action_transformer.policy2env(actions)
    #print(actions)
    