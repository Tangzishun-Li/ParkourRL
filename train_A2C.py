import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import os
from ParkourEnv import ParkourEnv
from play import play_demo

# --- 硬件检测 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- A2C 默认超参数 ---
DEFAULT_EPISODES = 200               
DEFAULT_GAMMA = 0.99                 
DEFAULT_ACTOR_LR = 1e-3              
DEFAULT_CRITIC_LR = 5e-3             
DEFAULT_ENTROPY_COEF = 0.01          
DEFAULT_UPDATE_STEPS = 50            
DEFAULT_MAX_STEPS = 2000             

# ==========================================
# 网络架构
# ==========================================

class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1) 
        )
    def forward(self, x): return self.fc(x)

class CriticNet(nn.Module):
    def __init__(self, state_dim):
        super(CriticNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1) 
        )
    def forward(self, x): return self.fc(x)

# ==========================================
# 训练生成器
# ==========================================

def train_parkour_a2c_generator(
    episodes: int = DEFAULT_EPISODES,
    gamma: float = DEFAULT_GAMMA,
    actor_lr: float = DEFAULT_ACTOR_LR,
    critic_lr: float = DEFAULT_CRITIC_LR,
    entropy_coef: float = DEFAULT_ENTROPY_COEF,
    update_steps: int = DEFAULT_UPDATE_STEPS,
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: int | None = None,
):
    print(f"系统自检：PyTorch 正在使用 -> {device}")
    print("Step 1: 正在初始化 C++ 物理引擎 (A2C)...")
    
    # [修复] 移除 headless 和 max_episode_steps 参数
    env = ParkourEnv()
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = ActorNet(state_dim, action_dim).to(device)
    critic = CriticNet(state_dim).to(device)
    actor_opt = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=critic_lr)

    if not os.path.exists('models'): os.makedirs('models')
    best_reward = -float('inf')

    # 打印配置
    print("-" * 40)
    try:
        print(f"  [+] 通过障碍物奖励:  +{env.get_reward_pass()}")
        print(f"  [-] 碰撞单次扣血量:  -{env.get_damage_taken()}")
    except AttributeError: pass
    print("-" * 40)

    rewards_window = []

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    print(f"Step 2: 进入训练循环 (共 {episodes} 轮)...")

    for episode in range(int(episodes)):
        state, _ = env.reset()
        total_reward = 0
        done = False
        episode_steps = 0

        # 手动在循环条件中控制步数限制
        while not done and episode_steps < int(max_steps):
            log_probs, values, rewards, masks, entropies = [], [], [], [], []
            
            # --- 采集阶段 ---
            for _ in range(int(update_steps)):
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                probs = actor(state_t)
                dist = Categorical(probs)
                action = dist.sample()
                value = critic(state_t)

                next_state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                
                log_probs.append(dist.log_prob(action))
                values.append(value)
                rewards.append(torch.FloatTensor([reward]).to(device))
                masks.append(torch.FloatTensor([1 - done]).to(device))
                entropies.append(dist.entropy())

                state = next_state
                total_reward += reward
                episode_steps += 1
                
                # 手动截断检查
                if episode_steps >= int(max_steps):
                    done = True

                if done: break

            # --- 计算阶段 ---
            next_value = critic(torch.FloatTensor(state).unsqueeze(0).to(device))
            returns = []
            R = next_value.detach()
            for r, m in zip(reversed(rewards), reversed(masks)):
                R = r + gamma * R * m
                returns.insert(0, R)

            returns = torch.cat(returns).detach().view(-1)      
            log_probs_t = torch.cat(log_probs).view(-1)
            values_t = torch.cat(values).view(-1)
            entropies_t = torch.stack(entropies).view(-1)

            advantage = returns - values_t 
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            # --- 更新阶段 ---
            actor_loss = -(log_probs_t * advantage.detach()).mean() - entropy_coef * entropies_t.mean()
            critic_loss = F.mse_loss(values_t, returns)

            actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()
            critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()

        rewards_window.append(total_reward)
        if len(rewards_window) > 50:
            rewards_window.pop(0)
        avg50 = float(np.mean(rewards_window))

        is_best = False
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(actor.state_dict(), "models/parkour_a2c_best.pth")
            is_best = True

        if (episode + 1) % 50 == 0: torch.cuda.empty_cache()

        yield {
            "episode": episode + 1,
            "steps": episode_steps,
            "reward": total_reward,
            "avg_reward": avg50,
            "is_best": is_best,
            "best_reward": best_reward
        }

if __name__ == "__main__":
    # 配置运行参数
    TRAIN_EPISODES = DEFAULT_EPISODES
    DEMO_EPISODES = 3
    
    trainer = train_parkour_a2c_generator(episodes=TRAIN_EPISODES)
    
    print(f"正在启动 A2C 独立训练进程...")
    try:
        for stats in trainer:
            ep = stats['episode']
            reward = stats['reward']
            avg = stats['avg_reward']
            steps = stats['steps']
            best_tag = " [NEW BEST!]" if stats['is_best'] else ""
            print(f"[A2C ] Ep {ep:3d} | Steps: {steps:4d} | Reward: {reward:7.2f} | Avg50: {avg:7.2f}{best_tag}")
            
    except KeyboardInterrupt:
        print("\n训练被用户手动中断。")

    print("\n训练全部结束。")

    # 自动演示
    print("正在加载历史最佳模型进行演示...")
    best_model_path = "models/parkour_a2c_best.pth"
    
    if os.path.exists(best_model_path):
        # 临时实例化环境以获取维度
        env = ParkourEnv()
        s_dim = env.observation_space.shape[0]
        a_dim = env.action_space.n
        del env
        
        actor = ActorNet(s_dim, a_dim).to(device)
        actor.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"成功加载最佳模型权重！")
        play_demo(actor, device, episodes=int(DEMO_EPISODES))
    else:
        print("警告：未找到最佳模型文件。")