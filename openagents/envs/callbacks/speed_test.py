import time
from minestudio.simulator.callbacks.callback import MinecraftCallback

class SpeedTestCallback(MinecraftCallback):
    def __init__(self, interval: int = 100):
        super().__init__()
        self.interval = interval
        self.num_steps = 0
        self.total_times = 0
    
    def before_step(self, sim, action):
        self.start_time = time.time()
        return action
    
    def after_step(self, sim, obs, reward, terminated, truncated, info):
        end_time = time.time()
        self.num_steps += 1
        self.total_times += end_time - self.start_time
        if self.num_steps % self.interval == 0:
            print(f"[Inference Speed Test Status] Average Time: {self.total_times / self.num_steps :.2f} | Average FPS: {self.num_steps / self.total_times :.2f} | Total Steps: {self.num_steps}")
        return obs, reward, terminated, truncated, info
    