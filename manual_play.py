import pygame
import time
from ParkourEnv import ParkourEnv
from render import Renderer

def main():
    print("\n" + "="*40)
    print("      手动模式启动 (Keyboard Control)")
    print("="*40)
    print(" [W] / [空格] / [↑]  : 跳跃")
    print(" [S] / [↓]          : 下蹲")
    print(" [其他键]           : 跑步")
    print("="*40)

    # 1. 初始化环境和渲染器
    env = ParkourEnv()
    renderer = Renderer()
    
    # 2. 重置游戏
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    # 3. 游戏主循环
    while not done:
        # --- 获取键盘输入 ---
        # 注意：pygame.key.get_pressed() 获取当前按键状态
        keys = pygame.key.get_pressed()
        
        action = 0 # 默认为 0 (跑步)

        # 判定优先级：跳跃 > 下蹲 > 跑步
        if keys[pygame.K_SPACE] or keys[pygame.K_w] or keys[pygame.K_UP]:
            action = 1 # Jump
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            action = 2 # Crouch
        
        # --- 步进环境 ---
        # 将键盘决定的 action 传给 C++ 物理引擎
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        done = terminated or truncated
        total_reward += reward

        # --- 渲染画面 ---
        # 使用之前的渲染器画图
        render_data = env.game.get_render_data()
        is_running = renderer.draw(render_data)
        
        if not is_running: # 如果点击了窗口关闭按钮
            print("游戏退出")
            break

        # 如果死了，打印分数并稍作停顿，然后自动重开
        if done:
            print(f"Game Over! 本局得分: {render_data.score} | 存活奖励: {total_reward:.1f}")
            time.sleep(1) # 死亡后暂停 1 秒
            
            # 重置环境继续玩
            state, _ = env.reset()
            total_reward = 0
            done = False
            
    renderer.close()

if __name__ == "__main__":
    main()