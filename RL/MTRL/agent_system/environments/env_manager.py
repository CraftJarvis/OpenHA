from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from openagents.agents.utils.system_prompt import mix_action_system_prompt
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory
from datetime import datetime
import uuid
import random
import json

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    def __init__(self, envs, projection_f, config):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        self.memory.store({'text_obs': self.pre_text_obs, 'action': [self.ACTION_LOOKUP[act] for act in actions]})
        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            else:
                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs


class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
    
    def reset(self) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs

# class MinecraftEnvironmentManager(EnvironmentManagerBase):  
#     def __init__(self, envs, projection_f, config):
#         self.memory = SimpleMemory()
#         self.texts = None
#         self.images = None #这两项是二维list，外层是batchsize，内层是历史长度
#         self.maximum_history_length = config.env.maximum_history_length

#         # self.system_prompt_tag = config.env.get("system_prompt_tag", None)
#         # from openagents.agents.utils.system_prompt import get_system_prompt
#         # self.system_prompt = get_system_prompt(self.system_prompt_tag)
#         super().__init__(envs, projection_f, config)

#     def reset(self) -> Dict[str, Any]:
#         import gc
#         gc.collect()
        
#         obs, infos = self.envs.reset()
#         self.texts = None
#         self.images = None
#         actions = None
#         texts, images = self.build_text_obs(obs, actions, infos)
#         # infos = [None] * self.envs.num_envs
#         observations = {'text': texts, 'image': images, 'anchor': obs.copy()}

#         return observations, infos

#     def step(self, text_actions: List[str]):
#         actions, valids = self.projection_f(text_actions)
#         next_obss, rewards, dones, infos = self.envs.step(actions)

#         texts, images = self.build_text_obs(next_obss, actions, infos)
        
#         try: 
#             next_observations = {
#                 'text': texts, #
#                 'image': images, #[next_obs["image"] for next_obs in next_obss],
#                 'anchor': next_obss.copy()
#             }   
#             rewards = to_numpy(rewards)
#             dones = to_numpy(dones)
#         except Exception as e:
#             breakpoint()
#         return next_observations, rewards, dones, infos


#         '''魔改了，返回[{
#             "content": obs_content,
#             "role": "user",
#         }]也行'''
#     def build_text_obs(self, next_obss, actions, infos: Tuple[Dict]=None) -> List[str]:
#         """
#         This function builds the text observation for the agent.
#         """
#         images = next_obss
#         postprocess_text_obs = []

#         if self.texts is None:
#             assert self.texts is None and self.images is None and actions is None
#             self.texts = [[] for _ in range(len(infos))]
#             self.images = [[] for _ in range(len(infos))]

#         # ins, image, action, image, action ,image ...
#         for i in range(len(infos)):
#             if len(self.texts[i]) >= 2*(self.maximum_history_length+1):
#                 breakpoint()
#             if len(self.texts[i]) == 0:
#                 self.texts[i].append({"content": [{"type": "text", "text": mix_action_system_prompt + infos[i]["task_description"]}, {"type": "text", "text": "<image>"}], "role": "user"})
#                 self.images[i].append(next_obss[i])
                
#             elif len(self.texts[i]) == 2*(self.maximum_history_length+1) -1:
#                 self.texts[i] = self.texts[i][2:]
#                 self.images[i] = self.images[i][1:]
#                 self.texts[i][0]["content"] = [{"type": "text", "text":mix_action_system_prompt + infos[i]["task_description"]}] + self.texts[i][0]["content"]
#                 self.texts[i].append({"content": [{"type": "text", "text": "Action:" +actions[i]["thought"].split("Action:")[-1].replace("<|endoftext|>", "").replace("<|im_end|>", "")}], "role": "assistant"})#.replace("<|endoftext|>", "").replace("<|im_end|>", "")
#                 self.texts[i].append({"content": [{"type": "text", "text": "<image>"}], "role": "user"})
#                 self.images[i].append(next_obss[i])
#             else:
#                 self.texts[i].append({"content": [{"type": "text", "text": "Action:" +actions[i]["thought"].split("Action:")[-1].replace("<|endoftext|>", "").replace("<|im_end|>", "")}], "role": "assistant"})#.replace("<|endoftext|>", "").replace("<|im_end|>", "")
#                 self.texts[i].append({"content": [{"type": "text", "text": "<image>"}], "role": "user"})
#                 self.images[i].append(next_obss[i])

#         # with open("debug_texts.json", "a") as f:
#         #     json.dump(self.texts[0], f, indent=4)

#         for image in self.images:
#             if len(image) == 0:
#                 breakpoint()

#         return self.texts, self.images
from dataclasses import dataclass, field
from collections import deque
import threading

# 你已有的全局：mix_action_system_prompt
# from ... import mix_action_system_prompt

@dataclass
class EnvHistory:
    max_hist: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    texts: deque = field(default_factory=deque)    # 每条是对话块 dict
    images: deque = field(default_factory=deque)   # 每条是图像帧

    def reset_with_first_obs(self, task_description: str, first_image):
        with self.lock:
            self.texts.clear()
            self.images.clear()
            self.texts.append({
                "content": [
                    {"type": "text", "text": mix_action_system_prompt + task_description},
                    {"type": "text", "text": "<image>"},
                ],
                "role": "user",
            })
            self.images.append(first_image)

    def append_turn(self, action_thought: str | None, next_image, task_description: str):
        with self.lock:
            # 限长：最多保留 (max_hist+1) 轮（user+assistant 为一轮）
            while len(self.texts) >= 2 * (self.max_hist) + 1:
                if self.texts:
                    self.texts.popleft()  # earliest user
                if self.texts:
                    self.texts.popleft()  # earliest assistant
                if self.images:
                    self.images.popleft()
                if self.texts:
                    self.texts[0]["content"] = [
                        {"type": "text", "text": mix_action_system_prompt + task_description}
                    ] + self.texts[0]["content"]

            # assistant 动作帧
            if action_thought is not None:
                act = "Action:" + str(action_thought).split("Action:")[-1] \
                        .replace("<|endoftext|>", "").replace("<|im_end|>", "")
                self.texts.append({"content": [{"type": "text", "text": act}], "role": "assistant"})

            # 下一张 image 的 user 帧
            self.texts.append({"content": [{"type": "text", "text": "<image>"}], "role": "user"})
            self.images.append(next_image)
            if (len(self.images)>=5):
                import pdb
                pdb.set_trace()
            # print("length of images:", len(self.images))

    def snapshot(self):
        with self.lock:
            return list(self.texts), list(self.images)


class MinecraftEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
        self.maximum_history_length = config.env.maximum_history_length
        self._histories: List[EnvHistory] = []
        self._global_lock = threading.Lock()
        self._ensure_histories_initialized()

    def _num_envs(self):
        return getattr(self.envs, 'num_envs', getattr(self.envs, 'num_processes', None)) or 1

    def _ensure_histories_initialized(self):
        n = self._num_envs()
        with self._global_lock:
            while len(self._histories) < n:
                self._histories.append(EnvHistory(self.maximum_history_length))

    # --------- 工具：从底层 obs 提取一帧图像 ----------
    def _image_from(self, obs_item):
        if isinstance(obs_item, dict) and "image" in obs_item:
            return obs_item["image"]
        return obs_item 

    # --------- 批量 reset（同步路径，保留兼容） ----------
    def reset(self):
        self._ensure_histories_initialized()
        obs_list, infos = self.envs.reset()            # list[obs], list[info]
        for i, (obs_i, info_i) in enumerate(zip(obs_list, infos)):
            first_img = self._image_from(obs_i)
            self._histories[i].reset_with_first_obs(info_i["task_description"], first_img)

        # 组 batch 视图
        texts_batch, images_batch = [], []
        for h in self._histories[:len(obs_list)]:
            t, im = h.snapshot()
            texts_batch.append(t)
            images_batch.append(im)
        observations = {'text': texts_batch, 'image': images_batch, 'anchor': obs_list.copy()}
        return observations, infos

    # --------- 单环境 reset（异步路径用） ----------
    def reset_one(self, env_id: int):
        self._ensure_histories_initialized()
        obs, info = self.envs.reset_single(env_id)
        self._histories[env_id].reset_with_first_obs(info["task_description"], self._image_from(obs))
        obs_view = self.make_obs_view(env_id, obs)
        print("resetting env finish:", env_id)
        
        return obs_view, info

    # --------- 单环境 step（异步路径用） ----------
    def step_one(self, env_id: int, text_action: str):
        self._ensure_histories_initialized()
        actions, _ = self.projection_f([text_action])
        action = actions[0]
        next_obs, reward, done, info = self.envs.step_single(env_id, action)
        # thought 提取（你也可传入 action["thought"]）
        thought = text_action if isinstance(text_action, str) else str(text_action)
        self._histories[env_id].append_turn(thought, self._image_from(next_obs), info["task_description"])
        reward = to_numpy(reward)
        done = to_numpy(done)
        info['is_action_valid'] = to_numpy(info.get('is_action_valid', True))
        return self.make_obs_view(env_id, next_obs), reward, done, info

    # --------- 兼容：批量 build_text_obs（如果旧代码会用到） ----------
    def build_text_obs(self, next_obss, actions, infos: Tuple[Dict, ...] = None):
        self._ensure_histories_initialized()
        # 写 histories
        for i in range(len(infos)):
            thought = None
            if actions is not None:
                thought = actions[i].get("thought") if isinstance(actions[i], dict) else str(actions[i])
            self._histories[i].append_turn(thought, self._image_from(next_obss[i]), infos[i]["task_description"])
        # 返回 batch 视图
        texts_batch, images_batch = [], []
        for i in range(len(infos)):
            t, im = self._histories[i].snapshot()
            texts_batch.append(t)
            images_batch.append(im)
        return texts_batch, images_batch

    # --------- 生成单 env 的 obs 视图，喂给 preprocess ----------
    def make_obs_view(self, env_id: int, anchor_obs: Any):
        self._ensure_histories_initialized()
        texts, images = self._histories[env_id].snapshot()
        return {"text": texts, "image": images, "anchor": anchor_obs} #相比普通的step，这里的texts和images没有放在envs batch里，可以看作都只有一维度，方便后续拼接



def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    if "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False)
        
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': 'eval_in_distribution', # 'eval_in_distribution' or 'eval_out_of_distribution'
        } 
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs)
        
        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "minecraft" in config.env.env_name.lower():
        from agent_system.environments.env_package.minecraft import build_minecraft_envs, minecraft_projection

        tasks = []
        if config.env.task_path is not None:
            with open(config.env.task_path, "r") as f:
                tasks += json.load(f)
        
        if config.env.tasks is not None:
            tasks += config.env.tasks.split(",")
        

        env_kwargs = {
            "tasks": tasks,
            "group_n": group_n,
            "rollout_path": config.env.rollout_path,
        }
        # recipe = copy.deepcopy(recipe_ogn)
        # task_config = gen_craft_item_task_config(recipe, task_group)
        # task_name = task_config['task_name']
        # reward_cfg = task_config['rewards']


        # random_item_prob = random.random()
        # record_path="MC-verl-agent/examples/grpo_trainer/output/videos"
        # rollout_path = os.path.join(
        # record_path, 
        # datetime.now().strftime('%y-%m-%d'),ftayyy
        # datetime.now().strftime('%y-%m-%d-%H-%M-%S')+'-'+task_name.replace(":", "_")+'-difficulty_'+str(random_item_prob))
        # os.makedirs(rollout_path, exist_ok=True)
        
        # env_kwargs = {
        #     # 'observation_mode': 'text',
        #     # 'num_products': None,
        #     # 'human_goals': config.env.minecraft.human_goals,
        #     # 'max_steps': config.env.max_steps,
        #     'group_n': group_n,
        #     "action_type": "env", 
        #     "obs_size": (640, 360),
        #     "render_size": (640, 360),
        #     "preferred_spawn_biome": random.choice(["plains"]), 
        #     "callbacks": [
        #         RecordCallback(record_path=rollout_path, fps=30, frame_type="pov"),
        #         RandomInitInventoryCallback(recipe, item_counts = item_counts, random_item_prob=random_item_prob),
        #         RewardsCallback(reward_cfg),
        #         InitialActionCallback("open_crafting_table")
        #     ],
        #     "camera_config": CameraConfig(
        #         camera_binsize = 1,
        #         camera_maxval = 10,
        #         camera_mu = 20,
        #         camera_quantization_scheme = "mu_law"
        #     )
        # }
        
        _envs = build_minecraft_envs(env_num=config.data.train_batch_size, group_n = group_n, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_minecraft_envs(env_num=config.data.val_batch_size, group_n = 1, is_train=False, env_kwargs=env_kwargs)
        projection_f = partial(minecraft_projection)
        
        envs = MinecraftEnvironmentManager(_envs, projection_f, config)
        val_envs = MinecraftEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    
    else:
        print("Environment not supported")
        exit(1)


from types import SimpleNamespace
from functools import partial
def main():
    def build_test_config():
        """根据你的完整配置提取最小必要项"""
        return SimpleNamespace(
            env=SimpleNamespace(
                env_name="minecraft",
                seed=0,
                task_path="OpenHA/rl/data_processor/utils/task_list.json",
                tasks = None,
                maximum_history_length=3,
                max_steps=10,
                rollout=SimpleNamespace(n=1),
                rollout_path="./rollout_debug"
            ),
            data=SimpleNamespace(
                train_batch_size=1,
                val_batch_size=0
            )
        )

    config = build_test_config()

    from agent_system.environments.env_package.minecraft import build_minecraft_envs, minecraft_projection

    tasks = []
    if config.env.task_path is not None:
        with open(config.env.task_path, "r") as f:
            tasks += json.load(f)
    
    if config.env.tasks is not None:
        tasks += config.env.tasks.split(",")
    

    env_kwargs = {
        "tasks": tasks,
        "group_n": config.env.rollout.n,
        "rollout_path": config.env.rollout_path,
    }
        
    _envs = build_minecraft_envs(env_num=config.data.train_batch_size, group_n = config.env.rollout.n, is_train=True, env_kwargs=env_kwargs)
    projection_f = partial(minecraft_projection)
        
    envs = MinecraftEnvironmentManager(_envs, projection_f, config)

    obs, info = envs.reset_one(0)
    fake_action = "press('w')"
    for i in range(10):
        obs2, reward, done, info2 = envs.step_one(0, fake_action)
        t, im = envs._histories[0].snapshot()
        print("text length:", len(t), "image length:", len(im))


if __name__ == "__main__":
    main()
