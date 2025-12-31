import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from ParkourEnv import ParkourEnv
from play import play_demo

# --- 硬件检测 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 默认超参数 ---
DEFAULT_EPISODES = 200                 
DEFAULT_MAX_STEPS_PER_EPISODE = 2000   
DEFAULT_GAMMA = 0.99
DEFAULT_EPSILON_START = 1.0
DEFAULT_EPSILON_MIN = 0.05
DEFAULT_EPSILON_DECAY = 0.995          
DEFAULT_LEARNING_RATE = 1e-3

class QNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
    def forward(self, x):
        return self.fc(x)

def _select_action(qnet: QNet, state_t: torch.Tensor, action_size: int, epsilon: float) -> int:
    if random.random() <= epsilon:
        return random.randrange(action_size)
    with torch.no_grad():
        return torch.argmax(qnet(state_t)).item()

def train_parkour_sarsa_generator(
    episodes: int = DEFAULT_EPISODES,
    max_steps_per_episode: int = DEFAULT_MAX_STEPS_PER_EPISODE,
    gamma: float = DEFAULT_GAMMA,
    epsilon_start: float = DEFAULT_EPSILON_START,
    epsilon_min: float = DEFAULT_EPSILON_MIN,
    epsilon_decay: float = DEFAULT_EPSILON_DECAY,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    seed: int | None = None,
):
    print(f"系统自检：PyTorch 正在使用 -> {device}")
    print("Step 1: 正在初始化 C++ 物理引擎 (SARSA)...")
    
    # [修复] 移除初始化参数
    env = ParkourEnv()

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # 打印配置
    print("-" * 40)
    try:
        print(f"  [+] 通过障碍物奖励:  +{env.get_reward_pass()}")
        print(f"  [-] 碰撞单次扣血量:  -{env.get_damage_taken()}")
    except AttributeError: pass
    print("-" * 40)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    qnet = QNet(state_size, action_size).to(device)
    optimizer = optim.Adam(qnet.parameters(), lr=learning_rate)
    
    if not os.path.exists('models'): os.makedirs('models')
    
    epsilon = float(epsilon_start)
    best_reward = -float('inf') 
    rewards_window = []
    
    print(f"Step 2: 进入训练循环... (算法: Deep SARSA)")

    for episode in range(int(episodes)):
        state, _ = env.reset()
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        step_count = 0

        action = _select_action(qnet, state_t, action_size, epsilon)

        done = False
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            step_count += 1
            total_reward += reward
            
            # [修复] 手动处理最大步数截断
            if step_count >= max_steps_per_episode:
                done = True

            next_state_t = torch.as_tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            if not done:
                next_action = _select_action(qnet, next_state_t, action_size, epsilon)
                with torch.no_grad():
                    target_q = reward + gamma * qnet(next_state_t)[0, next_action].item()
            else:
                next_action = None
                target_q = reward

            current_q = qnet(state_t)[0, action]
            loss = (current_q - torch.tensor(target_q, device=device, dtype=torch.float32)).pow(2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qnet.parameters(), max_norm=10.0)
            optimizer.step()

            state_t = next_state_t
            if next_action is not None:
                action = next_action

        epsilon = max(float(epsilon_min), epsilon * float(epsilon_decay))
        rewards_window.append(total_reward)
        if len(rewards_window) > 50:
            rewards_window.pop(0)
        avg50 = float(np.mean(rewards_window))

        is_best = False
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(qnet.state_dict(), "models/parkour_sarsa_best.pth")
            is_best = True
        
        yield {
            "episode": episode + 1,
            "steps": step_count,
            "reward": total_reward,
            "avg_reward": avg50,
            "epsilon": epsilon,
            "is_best": is_best,
            "best_reward": best_reward
        }

if __name__ == "__main__":
    TRAIN_EPISODES = DEFAULT_EPISODES
    DEMO_EPISODES = 3
    
    trainer = train_parkour_sarsa_generator(episodes=TRAIN_EPISODES)
    
    print(f"正在启动 SARSA 独立训练进程...")
    try:
        for stats in trainer:
            ep = stats['episode']
            reward = stats['reward']
            avg = stats['avg_reward']
            epsilon = stats['epsilon']
            steps = stats['steps']
            best_tag = " [NEW BEST!]" if stats['is_best'] else ""
            print(f"[SARSA] Ep {ep:3d} | Steps: {steps:4d} | Reward: {reward:7.2f} | Avg50: {avg:7.2f} | Eps: {epsilon:.3f}{best_tag}")
            
    except KeyboardInterrupt:
        print("\n训练被用户手动中断。")

    print("\n训练全部完成！正在加载 SARSA 最佳模型进行演示...")
    
    best_model_path = "models/parkour_sarsa_best.pth"
    if os.path.exists(best_model_path):
        env = ParkourEnv()
        s_dim = env.observation_space.shape[0]
        a_dim = env.action_space.n
        del env

        best_model = QNet(s_dim, a_dim).to(device)
        best_model.load_state_dict(torch.load(best_model_path, map_location=device))
        play_demo(best_model, device, episodes=int(DEMO_EPISODES))
    else:
        print("警告：未找到最佳模型文件。")