import torch
import time
import os
import numpy as np
import argparse
import sys

# 引入环境和渲染器
from ParkourEnv import ParkourEnv
from render import Renderer 

def play_demo(model, device, episodes=3):
    """
    通用演示函数：支持 DQN, SARSA 和 A2C
    """
    print("\n" + "="*40)
    print("      AI 跑酷演示启动 (通用模式)")
    print("="*40)

    # 1. 初始化环境 (C++ 纯计算)
    env = ParkourEnv()
    
    # 2. 初始化渲染器 (Python PyGame)
    renderer = Renderer()
    
    # 3. 准备模型
    model.eval() # 切换到评估模式
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        print(f"开始演示第 {episode + 1} 局...")
        
        while not done:
            # --- AI 推理 ---
            # 无论是 DQN/SARSA (输出 Q值) 还是 A2C (输出 概率)，
            # 演示时我们都取数值最大的那个动作 (Argmax)
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(state_t)
                action = torch.argmax(output).item()
            
            # --- 环境步进 ---
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            # --- 渲染画面 ---
            # 获取 C++ 数据并传给 Python 渲染器
            render_data = env.game.get_render_data()
            is_running = renderer.draw(render_data)
            
            if not is_running: # 用户点击了关闭窗口
                print("演示中断")
                return 

            # 强制退出保护
            if steps > 5000: break

        print(f"  --> 结局统计 | 步数: {steps:4d} | 总奖励: {total_reward:6.1f}")
        time.sleep(1.0) # 局间休息

    print("演示结束")
    renderer.close()

# ==========================================
# 自动模型加载逻辑 (适配三种算法)
# ==========================================
def load_best_model(device, algo=None):
    """
    尝试加载最佳模型。
    优先级: 用户指定 > DQN > A2C > SARSA
    """
    # 临时创建环境以获取维度
    try:
        env = ParkourEnv()
        s_dim = env.observation_space.shape[0]
        a_dim = env.action_space.n
        del env
    except:
        s_dim, a_dim = 6, 3 # 备用默认值
        print("Warning: 无法实例化环境获取维度，使用默认值 (6, 3)")

    model = None
    model_name = "Unknown"
    
    # 定义候选模型列表 (算法名, 文件路径, 类名, 来源模块)
    candidates = [
        ("DQN", "models/parkour_best.pth", "DQN", "train_DQN"),
        ("A2C", "models/parkour_a2c_best.pth", "ActorNet", "train_A2C"),
        ("SARSA", "models/parkour_sarsa_best.pth", "QNet", "train_SARSA"),
    ]
    
    # 如果用户指定了算法，优先筛选
    if algo:
        candidates = [c for c in candidates if c[0].lower() == algo.lower()]
        if not candidates:
            print(f"[Error] 不支持的算法类型: {algo}")
            return None

    # 遍历尝试加载
    for name, path, class_name, module_name in candidates:
        if os.path.exists(path):
            try:
                # 动态导入模块和类
                module = __import__(module_name)
                ModelClass = getattr(module, class_name)
                
                # 实例化并加载权重
                temp_model = ModelClass(s_dim, a_dim).to(device)
                temp_model.load_state_dict(torch.load(path, map_location=device))
                
                model = temp_model
                model_name = name
                print(f"[System] 发现并加载模型: {name} ({path})")
                break # 找到一个就停止
            except Exception as e:
                print(f"[Warning] 尝试加载 {name} 失败: {e}")
                continue
    
    if model is None:
        print("[Error] 未找到任何可用的模型文件 (models/*.pth)")
        
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3, help="演示局数")
    parser.add_argument("--algo", type=str, default=None, help="指定算法 (DQN, A2C, SARSA)，默认自动检测")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 自动加载
    model = load_best_model(device, args.algo)
    
    if model is not None:
        play_demo(model, device, episodes=args.episodes)