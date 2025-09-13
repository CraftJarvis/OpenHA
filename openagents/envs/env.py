import time
import json
import os
from openagents.envs.callbacks import RecordCallback, SummonMobsCallback,InitInventoryCallback,CommandsCallback
from minestudio.simulator import MinecraftSim
from minestudio.simulator.entry import CameraConfig
from minestudio.simulator.callbacks import SpeedTestCallback, RewardsCallback


def env_init(
        task_config:dict, 
        rollout_path:str, 
        args:dict,
        obs_size:tuple= (224,224), 
        render_size:tuple= (640, 360),
        camera_cfg:CameraConfig=None,
        record_raw_action:bool=True
    ) -> MinecraftSim:
    '''
    description: 配置好environment
    return {*}
    '''
    
    seed = task_config['seed']
    init_actions = task_config['init_actions']
    # callbacks
    init_inventory = task_config['callback']['init_inventory'].get("init_inventory",[])
    commands = task_config['callback'].get('commands',[])
    reward_cfg = task_config['rewards']
    callbacks = [
        RecordCallback(record_path=rollout_path, fps=args.fps, frame_type="pov"),
        InitInventoryCallback(init_inventory,
            inventory_distraction_level=task_config['callback']['init_inventory'].get("inventory_distraction_level",[0]),
            equip_distraction_level=task_config['callback']['init_inventory'].get("equip_distraction_level",[0]),
            forbidden_slots = task_config['callback']['init_inventory'].get("forbidden_slots",[]),
        ),
        
        RewardsCallback(reward_cfg),
        CommandsCallback(commands),
    ]
    mobs = task_config['callback'].get("mobs")
    if mobs:
        callbacks.append(SummonMobsCallback(mobs))
    
    env = MinecraftSim(
        action_type="env", 
        obs_size = obs_size, 
        render_size = render_size,
        seed = seed,
        preferred_spawn_biome=None,
        camera_config = camera_cfg,
        callbacks=callbacks,
    )
    obs, info = env.reset()
    flag = False
    
    for action in init_actions:
        time.sleep(0.1)
        obs, reward, terminated, truncated, info = env.step(action)
    
    if record_raw_action:
        raw_action_file_path = os.path.join(rollout_path, "raw_action.jsonl")
        with open(raw_action_file_path, 'a', encoding='utf-8') as f:
            for _ in range(len(init_actions)):
                f.write(json.dumps({"raw_action":""}, ensure_ascii=False) + '\n')
        
    return env