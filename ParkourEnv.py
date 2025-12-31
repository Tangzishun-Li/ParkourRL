import gymnasium as gym
from gymnasium import spaces
import numpy as np
import GameEnv  # C++ 编译模块

class ParkourEnv(gym.Env):
    def __init__(self):
        super(ParkourEnv, self).__init__()
        self.game = GameEnv.GameEnv()
        self.action_space = spaces.Discrete(3)
        
        # [修改] 变成 6 维
        # [HeroY, Dist, ObsY, Jump, Down, Speed]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )

    def get_reward_pass(self): return self.game.get_reward_pass()
    def get_reward_death(self): return self.game.get_reward_death()
    def get_reward_hit(self): return self.game.get_reward_hit() 
    def get_damage_taken(self): return self.game.get_damage_taken()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        
        # [修改] 手动构建 6 维初始状态
        # 初始背景速度通常是 4，归一化后 4/20 = 0.2
        initial_speed = 4.0 / 20.0
        
        # [HeroY, Dist, ObsY, Jump, Down, Speed]
        initial_obs = np.array([0.644, 1.0, 0.0, 0.0, 0.0, initial_speed], dtype=np.float32)
        
        return initial_obs, {}

    def step(self, action):
        result = self.game.step(action)
        
        observation = np.array(result.obs, dtype=np.float32)
        reward = float(result.reward)
        terminated = bool(result.done)
        truncated = False 
        info = {"score": result.score}
        
        return observation, reward, terminated, truncated, info