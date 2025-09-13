'''
Date: 2024-11-11 19:31:53
LastEditors: Muyao 2350076251@qq.com
LastEditTime: 2025-04-10 22:27:37
FilePath: /MineStudio/minestudio/simulator/callbacks/commands.py
'''

from minestudio.simulator.callbacks.callback import MinecraftCallback

class CommandsCallback(MinecraftCallback):
    
    def __init__(self, commands):
        super().__init__()
        self.commands = commands
    
    def after_reset(self, sim, obs, info):
        for command in self.commands:
            obs, reward, done, info = sim.env.execute_cmd(command)
        if self.commands:
            for kdx in range(30):
                action = sim.env.noop_action()
                obs, reward, done, info = sim.env.step(action)
            obs, info = sim._wrap_obs_info(obs, info)
        return obs, info