import gymnasium as gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
import time
from ParkourEnv import ParkourEnv
from play import play_demo 

# --- 硬件检测 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 超参数配置 (可在此处直接修改)
# ==========================================
DEFAULT_EPISODES = 100              
DEFAULT_MEMORY_SIZE = 50000         
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_STEPS = 10000           
DEFAULT_GAMMA = 0.99
DEFAULT_EPS_START = 1.0
DEFAULT_EPS_MIN = 0.1
DEFAULT_EPS_DECAY = 0.95          
DEFAULT_LR = 1e-4
DEFAULT_TARGET_UPDATE_FREQ = 10     # Target Network 更新频率

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # 网络结构：保持 256 隐藏层以适应复杂的连续状态
        self.fc1 = nn.Linear(state_size, 256) 
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def train_parkour_generator(
    episodes=DEFAULT_EPISODES,
    memory_size=DEFAULT_MEMORY_SIZE,
    batch_size=DEFAULT_BATCH_SIZE,
    max_steps=DEFAULT_MAX_STEPS,
    gamma=DEFAULT_GAMMA,
    lr=DEFAULT_LR,
    epsilon_decay=DEFAULT_EPS_DECAY,
    target_update_freq=DEFAULT_TARGET_UPDATE_FREQ,
    seed=None
):
    """
    DQN 训练生成器 (带 Target Network)。
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    print(f"系统自检：PyTorch 正在使用 -> {device}")
    print("Step 1: 正在初始化 C++ 物理引擎...")
    env = ParkourEnv()
    
    # 打印环境配置
    print("-" * 40)
    print(f"  [+] 通过障碍物奖励:  +{env.get_reward_pass()}")
    print(f"  [!] 碰到障碍物奖励:   {env.get_reward_hit()}")
    print(f"  [†] 死亡奖励/惩罚:    {env.get_reward_death()}")
    print(f"  [-] 碰撞单次扣血量:  -{env.get_damage_taken()}")
    print("-" * 40)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # --- 初始化两个网络 (Double DQN 基础) ---
    policy_net = DQN(state_size, action_size).to(device)
    target_net = DQN(state_size, action_size).to(device)
    
    # 初始化同步
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval() 

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = deque(maxlen=memory_size)
    
    if not os.path.exists('models'): os.makedirs('models')
    
    epsilon = DEFAULT_EPS_START
    best_reward = -float('inf')
    
    # 用于计算移动平均
    recent_rewards = deque(maxlen=50)
    
    print(f"Step 2: 进入训练循环 (共 {episodes} 轮, Target Update 每 {target_update_freq} 轮)...")

    for episode in range(episodes):
        state, _ = env.reset()
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        total_reward = 0
        step_count = 0

        while True:
            # Epsilon-Greedy + 启发式探索
            if random.random() <= epsilon:
                rnd = random.random()
                if rnd < 0.5: action = 1    
                elif rnd < 0.75: action = 2 
                else: action = 0            
            else:
                with torch.no_grad():
                    action = torch.argmax(policy_net(state_t)).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step_count += 1
            if step_count > max_steps: done = True

            next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            memory.append((state_t.cpu(), action, reward, next_state_t.cpu(), done))
            state_t = next_state_t
            total_reward += reward

            # 经验回放
            if len(memory) > batch_size:
                batch = random.sample(memory, batch_size)
                b_s = torch.cat([s for s, a, r, ns, d in batch]).to(device)
                b_a = torch.tensor([a for s, a, r, ns, d in batch]).to(device)
                b_r = torch.tensor([r for s, a, r, ns, d in batch], dtype=torch.float32).to(device)
                b_ns = torch.cat([ns for s, a, r, ns, d in batch]).to(device)
                b_d = torch.tensor([d for s, a, r, ns, d in batch], dtype=torch.float32).to(device)

                # --- 使用 Target Network 计算目标 Q 值 ---
                with torch.no_grad():
                    next_q_values = target_net(b_ns).max(1)[0]
                    target_q = b_r + gamma * next_q_values * (1 - b_d)
                
                # 计算当前 Q 值
                current_q = policy_net(b_s).gather(1, b_a.unsqueeze(1)).squeeze(1)
                
                loss = nn.functional.mse_loss(current_q, target_q)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            if done: break

        # --- 定期更新 Target Network ---
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 更新 Epsilon
        epsilon = max(DEFAULT_EPS_MIN, epsilon * epsilon_decay)
        
        # 统计数据
        recent_rewards.append(total_reward)
        avg_reward = sum(recent_rewards) / len(recent_rewards)

        # 保存最佳模型
        is_best = False
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(policy_net.state_dict(), "models/parkour_best.pth")
            is_best = True

        yield {
            "episode": episode + 1,
            "steps": step_count,
            "reward": total_reward,
            "avg_reward": avg_reward,
            "epsilon": epsilon,
            "is_best": is_best,
            "best_reward": best_reward
        }

# ==========================================
# 独立运行入口
# ==========================================
if __name__ == "__main__":
    # 配置运行参数
    TRAIN_EPISODES = DEFAULT_EPISODES
    DEMO_EPISODES = 3
    
    # 初始化训练生成器
    trainer = train_parkour_generator(episodes=TRAIN_EPISODES)
    
    print(f"正在启动独立训练进程...")
    
    try:
        # 1. 执行训练循环
        for stats in trainer:
            ep = stats['episode']
            reward = stats['reward']
            avg = stats['avg_reward']
            epsilon = stats['epsilon']
            steps = stats['steps']
            
            # 打印进度条
            best_tag = " [NEW BEST!]" if stats['is_best'] else ""
            print(f"Ep {ep:3d} | Steps: {steps:4d} | Reward: {reward:6.1f} | Avg50: {avg:6.1f} | Eps: {epsilon:.3f}{best_tag}")

    except KeyboardInterrupt:
        print("\n训练被用户手动中断。")

    print("\n训练流程结束。")

    # 2. 自动演示最佳模型
    if os.path.exists("models/parkour_best.pth"):
        print("\n" + "="*40)
        print(f">>> 开始演示最佳模型 (GUI模式, {DEMO_EPISODES}局) <<<")
        print("="*40)
        
        try:
            # 临时实例化环境以获取维度，避免硬编码
            env = ParkourEnv()
            s_dim = env.observation_space.shape[0]
            a_dim = env.action_space.n
            del env 
            
            # 加载模型
            model = DQN(s_dim, a_dim).to(device)
            model.load_state_dict(torch.load("models/parkour_best.pth", map_location=device))
            
            # 调用 play.py 中的演示函数
            play_demo(model, device, episodes=DEMO_EPISODES)
            
        except Exception as e:
            print(f"演示启动失败: {e}")
            print("请检查是否安装了 pygame，或显卡驱动是否正常。")
    else:
        print("未找到最佳模型文件 (models/parkour_best.pth)，跳过演示。")