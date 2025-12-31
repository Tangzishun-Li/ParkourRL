import os
import sys
import shutil
import argparse
import time

def build_game_engine():
    """自动化编译 C++ 扩展"""
    print("="*40)
    print("[Build] 检测到 C++ 核心缺失或需要更新，正在编译...")
    print("="*40)
    
    # 1. 创建 build 目录
    os.makedirs("build", exist_ok=True)
    
    # 2. 运行 CMake (Windows 和 Linux 命令略有不同)
    if os.name == 'nt': # Windows
        cmd_cmake = "cd build && cmake .. && cmake --build . --config Release"
    else: # Linux / Mac
        cmd_cmake = "cd build && cmake .. && make -j4"
    
    ret = os.system(cmd_cmake)
    if ret != 0:
        print("[Error] 编译失败！请检查是否安装了 CMake 和 C++ 编译器。")
        sys.exit(1)
        
    # 3. 将编译好的 .pyd (Windows) 或 .so (Linux) 复制到根目录
    found = False
    for root, dirs, files in os.walk("build"):
        for file in files:
            if file.startswith("GameEnv") and (file.endswith(".pyd") or file.endswith(".so")):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(".", file)
                shutil.copy(src_path, dst_path)
                print(f"[Build] 成功生成模块: {file}")
                found = True
    
    if not found:
        print("[Error] 编译看似成功，但未找到生成的库文件。")
        sys.exit(1)
    
    print("[Build] 编译完成！\n")

def check_env():
    """检查环境是否就绪"""
    try:
        import GameEnv
        # 简单测试一下
        env = GameEnv.GameEnv()
        print("[System] C++ 物理引擎加载成功。")
    except ImportError:
        build_game_engine()

def main():
    parser = argparse.ArgumentParser(description="AI Parkour 统一启动器")
    parser.add_argument('mode', type=str, nargs='?', default='play', 
                        choices=['play', 'train', 'build'], 
                        help="运行模式: play(演示), train(训练), build(仅编译)")
    parser.add_argument('--episodes', type=int, default=3, help="演示局数")
    args = parser.parse_args()

    # 1. 如果是 build 模式，编译完直接退出
    if args.mode == 'build':
        build_game_engine()
        return

    # 2. 检查并自动编译环境
    check_env()

    # 3. 运行对应模式
    if args.mode == 'train':
        print("[Main] 启动训练模式...")
        # 延迟导入，防止在编译前导入报错
        from train_DQN import train_parkour
        train_parkour()
        
    elif args.mode == 'play':
        print("[Main] 启动演示模式...")
        
        # 尝试加载模型
        try:
            from train_DQN import DQN
            import torch
            from play import play_demo
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_path = "models/parkour_best.pth"
            
            if os.path.exists(model_path):
                # 注意：这里维度必须和 ParkourEnv.py 中定义的一致 (5)
                model = DQN(5, 3).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"[Main] 已加载模型: {model_path}")
                play_demo(model, device, episodes=args.episodes)
            else:
                print(f"[Warning] 未找到模型 {model_path}，将使用随机初始化的模型进行演示。")
                model = DQN(5, 3).to(device)
                play_demo(model, device, episodes=args.episodes)
                
        except ImportError as e:
            print(f"[Error] 缺少必要的模块: {e}")
            print("请确保 train_DQN.py 和 play.py 位于同一目录。")

if __name__ == "__main__":
    main()