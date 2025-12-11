import ray
import gym
import numpy as np
import traceback
import random
from minestudio.simulator.callbacks import RewardsCallback, CommandsCallback, RecordCallback
from minestudio.simulator.entry import CameraConfig
from minestudio.simulator.entry import MinecraftSim
from openagents.utils.render import render_video_enable, save_render_videos
from openagents.envs.callbacks.random_init_inventory import RandomInitInventoryCallback
from openagents.envs.callbacks.initial_action import InitialActionCallback
import copy
import sys
import os
from openagents.envs.tasks.craft_item import get_available_craft_recipes, gen_craft_item_task_config_old, item_counts
from datetime import datetime
import time
from minestudio.simulator.callbacks.reward_move import RewardsMoveCallback
from openagents.envs.tasks.task_manager import choose_available_task
from openagents.envs.env import env_init
# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

@ray.remote(num_cpus=2)#resources={"minecraft_env": 0.98}, 
class MinecraftWorker:
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *WebAgentTextEnv* instance.
    """

    def __init__(self, env_id: int = 0):
        self.env_id = env_id
        self.cur_step = 0
        self.reset_num = 0
        self.won = False
        
    def step(self, action):
        """Execute a step in the environment"""


        #print("step with action:", action)
        raw_action = action["raw_action"]
        thought = action["thought"]
        try:
            obs, reward, terminated, truncated, info = self.env.step(raw_action)
        except:
            print("⏱️ MinecraftWorker step 超时，正在重试...)")
            obs_image, info = self.reset()
            return obs_image, 0,0, info
            

        print(f"env_id:{self.env_id}, Rollout step: {self.cur_step}")
        self.cur_step += 1



        if reward > 0:
            self.won = True

        info["won"] = self.won
        info["step_count"] = self.cur_step
        info["task_name"] = self.task_name
        info["task_description"] = self.task_description
        if reward > 0 and self.won:
            terminated = True
        
        return obs["image"], reward, terminated, info
        
    def reset(self, env_kwargs= None):
        """Reset the environment with given session index"""

        if env_kwargs is not None:
            self.env_kwargs = env_kwargs

        self.task_name = self.env_kwargs["task_name"]
        self.task_description = self.env_kwargs["task_description"]
        self.won = False
        self.record_path = self.env_kwargs.get("rollout_path", None)
        datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.record_path = None if self.record_path is None else os.path.join(self.record_path, f"{datetime_str}_reset_{self.reset_num}_{self.task_name}")       
        
        self.env_kwargs["rollout_path"] = self.record_path
        
        if getattr(self, "env", None) is not None:
            self.env.close()
            time.sleep(2)
            del self.env

        time.sleep(random.randint(0,180))

        while True:
            try:
                self.env, extra_info = env_init(**self.env_kwargs)
                break
            except Exception as e:
                print("⏱️ env_init 超时，正在重试...")
                traceback.print_exc()
                if getattr(self, "env", None) is not None:
                    try:
                        self.env.close()
                        del self.env
                    except Exception:
                        pass
            time.sleep(random.randint(0,10))
            
        obs, info = extra_info["obs"], extra_info["info"]

        info["won"] = self.won
        info["step_count"] = self.cur_step
        info["task_name"] = self.task_name
        info["task_description"] = self.task_description
        self.cur_step = 0
        self.reset_num += 1
        return obs["image"], info
    
    def get_available_actions(self):
        """Get available actions"""
        return self.env.action_space
    
    def close(self):
        """Close the environment"""
        return self.env.close()

    def save_render_videos(self):
        renderred_frames = render_video_enable(
            self.record_path, 
            enable_task_name=True, enable_rawaction=True, enable_thought=True, enable_envaction=True,enable_point_cot=False,
            raw_action_max_length = 5,
        )
        #print(f"saving render video in {self.record_path} with {len(renderred_frames)} frames")
        try:
            save_render_videos(self.record_path, renderred_frames)
            print("保存视频成功:", self.record_path)
        except:
            print("保存视频失败")
            breakpoint()

    def ready(self):
        return True

# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class MinecraftMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def kill_workers(self, close_timeout: int = 60):
        """优雅关闭所有 workers，失败时强制 kill"""
        if getattr(self, "_workers", None) is None:
            return

        try:
            # 1. 并行触发 close()
            close_refs = [w.close.remote() for w in self._workers]

            # 2. 等待所有 worker close 完成（有超时）
            try:
                ray.get(close_refs, timeout=close_timeout)
            except Exception as e:
                print(f"⚠️ Some workers did not close gracefully: {e}")

            # 3. 强制 kill 掉所有 worker，防止残留
            for w in self._workers:
                try:
                    ray.kill(w, no_restart=True)
                except Exception:
                    pass

        finally:
            del self._workers


    def reset_workers(self):
        import gc
        gc.collect()
        self.kill_workers()
        random.shuffle(self.tasks)
        
        while True:
            try:
                self._workers = []
                futures = []
                for idx in range(self.num_processes):
                    task = self.tasks[idx // self.group_n]
                    _env_kwargs = self.format_kwargs(task) #(self.rgs, idx)
                    worker = (MinecraftWorker.remote(idx))
                    self._workers.append(worker)
                    futures.append(worker.reset.remote(_env_kwargs))
                results = ray.get(futures, timeout = 1800)
                break
            except Exception as e:
                print("⏱️ 创建 MinecraftWorker 超时，正在重试...")
                traceback.print_exc()
                self.kill_workers()

        return results


    def __init__(
        self,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
        env_kwargs: dict = None,
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
        self.reset_num = 0
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        self.record_path = env_kwargs.get("rollout_path", None)
        self.tasks = env_kwargs["tasks"]
        if not is_train: assert group_n == 1

        self._env_kwargs = env_kwargs if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}
        self.reset_workers()


    def format_kwargs(self, task): #minicase
        task_config = choose_available_task(task,difficulty="zero")
        env_kwargs = {
            "group_n": self.group_n,
            "action_type": "env",
            "task_name": task_config['task_name'],
            "task_description": task_config["task_description"],
            "task_config": task_config,
            "rollout_path": self.record_path,
            "if_standard_camera_config": True,
        }
        return env_kwargs

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        futures = []
        while True:
            try:        
                for worker, action in zip(self._workers, actions):
                    future = worker.step.remote(action)
                    futures.append(future)
                results = ray.get(futures)
                break
            except Exception as e:
                print("⏱️ MinecraftWorker step 超时，正在重试...")
                traceback.print_exc()
                self.reset_workers()

        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
            
        return obs_list, reward_list, done_list, info_list

    def reset(self):
        results = self.reset_workers()
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        self.reset_num+=1
        return obs_list, info_list

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    # def render(self, mode: str = 'text', env_idx: int = None):
    #     if env_idx is not None:
    #         future = self._workers[env_idx].render.remote(mode)
    #         return ray.get(future)

    #     futures = []
    #     for worker in self._workers:
    #         future = worker.render.remote(mode)
    #         futures.append(future)
        
    #     return ray.get(futures)

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return
        self.kill_workers() 
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()

    
    # ------------------------------------------------------------------
    # Single environment control (for async control) --------------------
    # ------------------------------------------------------------------
    def reset_single(self, env_id: int, env_kwargs: dict | None = None, start_time= 20):
        """
        Reset a single environment by env_id.

        Returns:
            obs (np.ndarray or list)
            info (dict)
        """

        time.sleep(random.random()*start_time)
        print(f"resetting env {env_id}")
        w = self._workers[env_id]
        assert 0 <= env_id < len(self._workers), f"Invalid env_id: {env_id}"
        if env_kwargs is None:
            idx = env_id // self.group_n #random.randint(0, len(self.tasks)-1)
            task = self.tasks[idx]
            env_kwargs = self.format_kwargs(task)

        future = self._workers[env_id].reset.remote(env_kwargs)
        obs, info = ray.get(future, timeout=1800)

        print(f"resetting env {env_id} DONE")
        return obs, info

    def shuffle_tasks(self):
        random.shuffle(self.tasks)

    def step_single(self, env_id: int, action: dict):
        """
        Step only one environment (used for async control).

        Args:
            env_id: int
            action: dict { "raw_action": ..., "thought": ... }

        Returns:
            obs, reward, done, info
        """
        assert 0 <= env_id < len(self._workers), f"Invalid env_id: {env_id}"
        # while True:
        #     try:
        future = self._workers[env_id].step.remote(action)
        obs, reward, done, info = ray.get(future, timeout=1800)
                # break
            # except Exception as e:
            #     print(f"⚠️ step_single({env_id}) 出错，正在重置该 worker...")
            #     traceback.print_exc()
            #     # worker 崩了就重建
            #     try:
            #         ray.kill(self._workers[env_id], no_restart=True)
            #     except Exception:
            #         pass

            #     kwargs = self.format_kwargs(self.tasks[env_id // self.group_n])
            #     self.reset_single(env_id, kwargs)

        return obs, reward, done, info #相比batch step, 没有打包成list




# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_minecraft_envs(
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    env_kwargs: dict = None,
    
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""
    return MinecraftMultiProcessEnv(
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )

