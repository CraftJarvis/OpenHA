import abc
from typing import Dict, Any, Literal, List, Union
import numpy as np
import torch
import copy
from openagents.assets import ENV_NULL_ACTION
from minestudio.simulator.entry import CameraConfig
from minestudio.utils.vpt_lib.actions import ActionTransformer
from minestudio.utils.vpt_lib.action_mapping import CameraHierarchicalMapping

class BaseAgent(abc.ABC):
    def __init__(self,**kwargs):
        self._action_type = "agent"
    
    @abc.abstractmethod
    def get_action(self, obs: Union[Dict[str, Any],np.ndarray], instruction:str=None, info: Dict[str, Any]=None, verbose=False):
        pass
        
    @abc.abstractmethod
    def reset(self, instruction:str=None, task_name:str=None):
        pass
    
    @property
    def action_type(self):
        return self._action_type
    
class MineCraftAgent(BaseAgent):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @classmethod
    def env_null_action(cls) -> Dict[str, Any]:
        return copy.deepcopy(ENV_NULL_ACTION)
    
    @classmethod
    def init_action_mapper_and_transformer(
        cls,
        camera_binsize: int = 2,
        camera_maxval: int = 10,
        camera_mu: float = 10.0,
        camera_quantization_scheme: str = "mu_law",
    ) -> tuple[CameraHierarchicalMapping, ActionTransformer]:
        camera_config = CameraConfig(
            camera_maxval=camera_maxval,
            camera_binsize=camera_binsize,
            camera_mu=camera_mu,
            camera_quantization_scheme=camera_quantization_scheme,
        )
        action_transformer = ActionTransformer(**camera_config.action_transformer_kwargs)
        action_mapper = CameraHierarchicalMapping(n_camera_bins=camera_config.n_camera_bins)
        return action_mapper, action_transformer

    @classmethod
    def agent_action_to_env_action(
        cls, 
        action: Dict[str, Any], 
        action_mapper: CameraHierarchicalMapping, 
        action_transformer: ActionTransformer
    ) -> Dict[str, Any]:
        
        if isinstance(action, tuple):
            action = {"buttons": action[0], "camera": action[1]}
        
        if isinstance(action["buttons"], torch.Tensor):
            action["buttons"] = action["buttons"].cpu().numpy()
        if isinstance(action["camera"], torch.Tensor):
            action["camera"] = action["camera"].cpu().numpy()
        
        action = action_mapper.to_factored(action)
        action = action_transformer.policy2env(action)
        return action
    
    def __str__(self):
        return "mc_agent"
    
class VanillaMCAgent(MineCraftAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.success = None  # 任务是否成功完成
        
    def get_action(self, obs: Union[Dict[str, Any],np.ndarray], instruction:str=None, info: Dict[str, Any]=None, verbose=False):
        self.tick += 1
        if self.tick >= self.num:
            self.success = True
        return self.env_null_action()
    
    def reset(self, instruction = None, task_name = "", num=1):
        self.task_name = task_name
        self.success = None
        self.num = num
        self.tick = 0
        
    def __str__(self):
        return "vanilla_agent"
    