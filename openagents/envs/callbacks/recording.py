import av
from pathlib import Path
from minestudio.simulator.callbacks.callback import MinecraftCallback
from typing import Literal
from rich import print
from copy import deepcopy
import numpy as np
from gymnasium import spaces
from collections import defaultdict
import json
import cv2
from PIL import Image
import io  
import base64 

from openagents.envs import ENV_RESET_STEPS, DEFAULT_MAXIMUM_BUFFER_SIZE

# DEFAULT_MAXIMUM_VIDEO_TIME = 5 # min
# DEFAULT_MAXIMUM_VIDEO_FRAMES = DEFAULT_MAXIMUM_VIDEO_TIME * 60 * 20

# DEFAULT_MAXIMUM_INFO_TIME = 10 # min 
# DEFAULT_MAXIMUM_INFO_STEPS = DEFAULT_MAXIMUM_INFO_TIME * 60 * 20


REALTIME_RECORDING = False

def encode_obs(obs, format="PNG"): 
    '''
    obs: numpy.array (info['pov'])
    '''
    # 将 numpy 数组转换为 PIL 图像
    image = Image.fromarray(np.uint8(obs))
    res = obs.shape
    # 创建一个内存中的字节流
    buffer = io.BytesIO()
    # 将图像保存为 PNG 格式到字节流中
    image.save(buffer, format=format)
    # 获取字节流中的数据
    img_bytes = buffer.getvalue()
    # 对字节数据进行 Base64 编码
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return {"format": format, "resolution": res, "base64": img_base64}


def post_info(info:dict) -> dict:
    processed_info = {'stats': {}}
    for k, v in info.items():
        if k == 'isGuiOpen' or k == 'is_gui_open':
            processed_info['is_gui_open'] = v
        elif k == 'location_stats':
            processed_info['location_stats'] = {
                'biome_id': int(v['biome_id']), 
                'biome_rainfall': float(v['biome_rainfall']), 
                'biome_temperature': float(v['biome_temperature']), 
                'can_see_sky': bool(v['can_see_sky']), 
                'is_raining': bool(v['is_raining']), 
                'light_level': int(v['light_level']),
                'sea_level': int(v['sea_level']),
                'sky_light_level': float(v['sky_light_level']),
                'sun_brightness': float(v['sun_brightness']),
                'pitch': float(v['pitch']), # array(0.), 
                'yaw': float(v['yaw']), # array(0.), 
                'xpos': float(v['xpos']), # array(978.5),
                'ypos': float(v['ypos']), # array(64.), 
                'zpos': float(v['zpos']) # array(922.5)
            }
        elif k == 'health':
            processed_info['health'] = float(v)
        elif k == 'food_level':
            processed_info['food_level'] = float(v)
        elif k == 'pov':
            continue 
        elif k == 'image':
            continue
        elif k == 'voxels':
            continue
        elif k == 'inventory': 
            processed_info['inventory'] = v # keep same to avoid other errors
        elif k == 'equipped_items':
            equipped_items = {
                'chest': v['chest']['type'],
                'feet': v['feet']['type'],
                'head': v['head']['type'], # {'damage': array(0), 'maxDamage': array(0), 'type': 'air'}, 
                'legs': v['legs']['type'], # {'damage': array(0), 'maxDamage': array(0), 'type': 'air'}, 
                'mainhand': v['mainhand']['type'], # {'damage': array(0), 'maxDamage': array(0), 'type': 'air'}, 
                'offhand': v['offhand']['type'] # {'damage': array(0), 'maxDamage': array(0), 'type': 'air'}
            }
            processed_info['equipped_items'] = equipped_items
        elif k in ['pickup', 'break_item', 'craft_item', 'mine_block', 'kill_entity', 'use_item', 'drop', 'entity_killed_by']:
            # break_item and entity_killed_by is special log, which is caused from outside environment
            # mine_block, kill_entity, craft_item is the important info   
            # use_item, pickup, drop is usually can be omitted
            for item_k, item_v in v.items(): # 'acacia_boat': array(0.),
                if int(item_v) > 0:
                    processed_info['stats'][f'{k}:{item_k}'] = int(item_v)
        elif k == "custom":
            for item_k, item_v in v.items():
                if "interact_with" in item_k:
                    processed_info['stats'][f'interact_with:{item_k[14:]}'] =  int(item_v)
        elif k == 'damage_dealt':
            continue
        elif k == 'player_pos':
            continue
        elif k == 'mobs':
            continue
        elif k == 'message':
            continue
        else:
            raise NotImplementedError(f"Key {k} not implemented yet in info processing function.")
    return processed_info

# def post_action(action:dict) -> dict:
#     print(action)
#     processed_action = {}
#     for k,v in action.items():
#         if k == 'camera':
#             processed_action['camera'] = v.tolist()
#         else:
#             processed_action[k] = int(v)
#     return processed_action

class RecordCallback(MinecraftCallback):
    def __init__(self, 
            record_path: str, 
            fps: int = 20, 
            frame_type: Literal['pov', 'obs'] = 'pov', 
            maximum_length: int = DEFAULT_MAXIMUM_BUFFER_SIZE, 
            recording: bool = True, 
            show_actions=False, 
            record_actions=True, 
            record_infos=True, 
            # record_observations=False,
            no_bframe = True,
            before_recording = ENV_RESET_STEPS, 
            **kwargs
            ):
        super().__init__(**kwargs)
        self.record_path = Path(record_path)
        self.record_path.mkdir(parents=True, exist_ok=True)
        self.recording = recording
        self.maximum_length = maximum_length # maximum length of buffer 
        self.record_actions = record_actions
        self.record_infos = record_infos
        # self.record_observations = record_observations
        self.before_recording = before_recording
        self.no_bframe = no_bframe
        if recording:
            print(f'[blue]Recording enabled, saving episodes to {self.record_path}[/blue]')
        self.fps = fps
        self.frame_type = frame_type
        # self.episode_id = 0
        self.step_count = 0

        #self.frames = []
        self.frame_buffer = []
        self.infos = []
        self.actions = []

    def before_reset(self, sim, reset_flag: bool) -> bool:
        if self.recording:
            self._save_episode()
            # self.episode_id += 1
        return reset_flag

    def after_reset(self, sim, obs, info):
        # 注意，第一帧是没有的
        return obs, info
    
    def before_step(self, sim, action):
        if not self.recording:
            return action 
        if self.step_count < self.before_recording:
            return action
        p_action = self._convert_data(action)
        if self.record_actions:
            self.actions.append(p_action)
        if len(self.actions)>=self.maximum_length:
            self._save_action()
        return action
    
    def after_step(self, sim, obs, reward, terminated, truncated, info):
        self.step_count += 1 
        if not self.recording:
            return obs, reward, terminated, truncated, info

        if self.step_count < self.before_recording:
            return obs, reward, terminated, truncated, info
        
        if len(self.frame_buffer) >= self.maximum_length:
            print(f"Reach the maximum frame length {self.maximum_length}, save video ...")
            self._save_episode()

        if self.frame_type == 'obs':
            self.frame_buffer.append(encode_obs(obs['image']))
        elif self.frame_type == 'pov':
            self.frame_buffer.append(encode_obs(info['pov']))
        else:
            raise ValueError(f'Invalid frame_type: {self.frame_type}')
        
        # if self.record_infos:
        #     self.infos.append(info)
        p_info = post_info(info)
        if self.record_infos:
            self.infos.append(p_info)
        if len(self.infos) >= self.maximum_length:
            self._save_info()

        return obs, reward, terminated, truncated, info
    
    def before_close(self, sim):
        if self.recording:
            self._save_episode()
        
        if self.record_infos:
            self._save_info()
        
        if self.record_actions:
            self._save_action()
    
    def _save_info(self):
        info_path = self.record_path / 'info.jsonl'
        with open(info_path, 'a', encoding='utf-8') as f:
            # 将字典对象序列化为 JSON 字符串，并写入文件
            for info in self.infos:
                f.write(json.dumps(info, ensure_ascii=False) + '\n')
        self.infos = []

    def _save_action(self):
        action_path = self.record_path / 'action.jsonl'
        with open(action_path, 'a', encoding='utf-8') as f:
            for action in self.actions:
                f.write(json.dumps(action, ensure_ascii=False) + '\n')
        self.actions = []
    
    def _save_episode(self):
        if len(self.frame_buffer) == 0:
            return 
        output_path = self.record_path / f'episode.jsonl'
        with open(output_path, 'a', encoding='utf-8') as f:
            for frame in self.frame_buffer:
                f.write(json.dumps(frame, ensure_ascii=False) + '\n')
        self.frame_buffer = []
    
    def _convert_data(self, data):
        if isinstance(data, dict):
            # Iterate over items and apply conversion recursively
            return {key: self._convert_data(value) for key, value in data.items()}
        elif isinstance(data, defaultdict):
            return {key: self._convert_data(value) for key, value in data.spaces.items()}
        elif isinstance(data, spaces.Dict):
            return {key: self._convert_data(value) for key, value in data.spaces.items()}
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data
        
    