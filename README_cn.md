![Role|50](https://github.com/Tangzishun-Li/ParkourRL/blob/main/res/d1.png)![ParkourRL](https://github.com/Tangzishun-Li/ParkourRL/blob/main/res/paoku.gif)![Role|50](https://github.com/Tangzishun-Li/ParkourRL/blob/main/res/d1.png)
# ParkourRL: 基于深度强化学习的 2D 跑酷游戏

**ParkourRL** 是一个深度强化学习研究项目，它结合了高性能的 C++ 物理引擎与基于 Python 的深度强化学习 (DRL) 算法。本项目利用 **Pybind11** 将 C++ 编写的游戏环境暴露给 Python，从而能够使用 **DQN**、**A2C** 和 **SARSA** 等算法高效地训练智能体，使其能够在复杂的障碍赛道中自主跑酷。

📽️我们的视频: [点这里](https://youtu.be/mt-hTwFStYk)
---

## 📂 项目结构

```text
ParkourRL/
├── CMakeLists.txt       # C++ 扩展的 CMake 编译配置
├── config.py            # 全局配置参数
├── render.py            # PyGame 渲染引擎 (用于可视化)
├── main.py              # 主管理脚本 (负责构建/运行管理)
├── play.py              # 推理脚本 (用于可视化已训练的模型)
├── manual_play.py       # 人工手操测试脚本
├── train_DQN.py         # DQN 算法训练脚本
├── train_A2C.py         # A2C 算法训练脚本
├── train_SARSA.py       # SARSA 算法训练脚本
├── src_fix.zip          # 固定关卡环境源码 (压缩包)
├── res/                 # 游戏资源 (精灵图, 背景图)
│   ├── bg001.png
│   ├── hero1.png
│   └── ... 
└── src/                 # C++ 核心源码目录 (默认: 随机环境)
    ├── GameEnv.h        # 环境头文件 (数据结构, 碰撞逻辑)
    ├── GameEnv.cpp      # 环境实现 (物理引擎, 奖励系统)
    └── binding.cpp      # Pybind11 绑定定义

```

---

## 🛠️ 环境依赖与安装

在运行项目之前，请确保您的系统满足以下要求。

### 第 0 步: 系统基础环境配置

#### 💻 Windows 用户

**方案 A: Visual Studio (推荐)**
这是 Windows 上最稳定的开发方式。

1. 下载并安装 **[Visual Studio Community 2022](https://visualstudio.microsoft.com/zh-hans/vs/community/)**。
2. 在安装过程中，选择工作负载：**“使用 C++ 的桌面开发” (Desktop development with C++)**。
* *注：这会自动安装 MSVC 编译器和 CMake。*


3. 确保已安装 **[Python 3.8+](https://www.python.org/downloads/)** 并将其添加到系统 PATH 环境变量中。

**方案 B: Chocolatey (命令行)**
如果您更喜欢使用包管理器：

1. 以管理员身份打开 PowerShell。
2. 运行以下命令安装 [scoop](https://scoop.sh/)(Windows 命令行安装器)：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression
```
1. 安装 python, mingw, cmake
```bash
scoop install python mingw cmake 
```



#### 🍎 macOS 用户

macOS 需要 Xcode 命令行工具来进行 C++ 编译。

1. **安装 Xcode 命令行工具**:
```bash
xcode-select --install
```


2. **安装 Homebrew** (如果尚未安装):
```bash
/bin/bash -c "$(curl -fsSL [https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh](https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh))"
```


3. **安装 Python 和 CMake**:
```bash
brew install python cmake
```



#### 🐧 Linux (Ubuntu/Debian) 用户

```bash
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv cmake g++ build-essential
```

---

### 第 1 步: Python 环境初始化

1. **克隆/下载项目** 到本地目录。
2. **创建并激活虚拟环境** (推荐):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```


3. **安装依赖**:
```bash
pip install -r requirements.txt
# 或者手动安装: pip install torch numpy gymnasium pygame pybind11
```



### 第 2 步: 编译 C++ 环境

为了追求高性能，核心游戏逻辑采用 C++ 编写。在运行任何 Python 脚本之前，您必须先编译 C++ 扩展。

```bash
python main.py build
```

* **成功提示:** 终端将显示编译进度，并在根目录下生成一个共享库文件 (`.pyd` 或 `.so`)。
* **重要:** 如果您切换了环境模式 (参见下文 *进阶用法*)，必须重新运行此命令。

---

## 🚀 使用指南

### 1. 手动测试 (人工游玩)

通过人工操作来验证游戏物理引擎和渲染是否正常。

* **操作:** `W` or `up` (跳跃), `S` or `down` (下蹲)

```bash
python manual_play.py
```

### 2. 训练智能体

使用支持的算法开始训练。模型会自动保存到 `models/` 目录。

* **DQN:**
```bash
python train_DQN.py
```


* **A2C:**
```bash
python train_A2C.py
```


* **SARSA:**
```bash
python train_SARSA.py
```



### 3. 可视化 (推理演示)

脚本会自动检测并加载表现最好的模型 (DQN/A2C/SARSA)，并渲染游戏画面。

```bash
python play.py
```

*可选参数:* `python play.py --episodes 5 --algo DQN`

---

## ⚔️ 进阶：环境切换

本项目支持两种环境模式：

1. **随机环境 (默认):** 障碍物永远随机生成。最适合通用训练。
2. **固定环境:** 预先设计好的确定性关卡脚本。最适合复现性和特定挑战测试。

**如何切换到固定环境:**

1. 在项目根目录下找到 **`src_fix.zip`**。
2. 解压该文件，并用解压出的内容替换现有的 **`src/`** 文件夹。
3. **⚠️ 重新编译环境:**
更改 C++ 源码后，必须重新编译才能生效：
```bash
python main.py build
```



*(如需切回随机环境，请使用 git 还原 `src/` 文件夹或使用备份，然后重新编译。)*

---

## ⚙️ 参数配置

### 奖励系统配置

要修改强化学习的奖励结构，请编辑 **`src/GameEnv.cpp`** 中的宏定义：

```cpp
#define REWARD_PASS 10.0f  // 通过奖励
#define REWARD_DEATH -0.0f // 死亡额外惩罚 (建议保持0或很小，让具体操作的惩罚占主导)
#define REWARD_STEP 0.02f  // 步数奖励
#define DAMAGE_TAKEN 10    // 每次扣血
#define INITIAL_BLOOD 100  // 初始血量

#define PENALTY_SEVERE -50.0f   // 重度惩罚：反向操作
#define PENALTY_MODERATE -20.0f // 中度惩罚：不操作
#define PENALTY_SLIGHT -10.0f   // 轻微惩罚：操作正确但时机不对
```

**注意:** 修改 `src/GameEnv.cpp` 后，必须重新运行 `python main.py build`。

### 超参数

要修改训练参数 (如 Batch Size, Learning Rate, Epsilon)，请直接编辑对应的 Python 脚本 (例如 `train_DQN.py`, `train_SARSA.py`)。

---

## 📝 更新日志 (Changelog)

### v1.2 (当前版本)

* **新增算法:** 实现了 **A2C** (`train_A2C.py`) 和 **SARSA** (`train_SARSA.py`)，用于进行全面的性能对比。
* **状态空间增强:** 在状态向量中增加了相对速度量，使智能体能够区分静态和动态障碍物。
* **网络升级:** 将 DQN 隐藏层维度从 128 提升至 256，以增强特征提取能力。
* **奖励重构:** 引入了 **多级惩罚机制 (Multi-level Penalty System)**，根据错误的类型 (严重程度/时机) 提供细粒度的反馈，取代了笼统的死亡惩罚。

### v1.1

* **物理修复:** 调整了下蹲持续时间的帧数逻辑 (8,30 -> 8,40)，解决了长挂钩障碍物的碰撞问题。

---
## 🌸 未来工作
1. 开发docker容器，方便在不同环境中运行项目。
2. 实现更多的强化学习算法，如PPO。
   
---

## 🧑‍💻 项目成员

\#3 Wang Minglei \
\#7 Zhou Boyan \
\#10 Luo Zhengnan \
\#15 Zhang Gaoyuan \
\#17 Jin Qihao \
\#21 [Li Tangzishun](https://github.com/Tangzishun-Li)\
\#26 [Chen Junyu](https://github.com/Chenjunyu1010) \
\#30 Li Nizhang\
*(排名顺序没有先后, 只是根据学号排序)*

---
## 🙏鸣谢

最后我们由衷感谢澳门科技大学的卢教授和助教们，他们的指导和支持让我们能够完成这个项目。

---

## 📄 许可与版权

**版权所有 © 澳门科技大学 (Macau University of Science and Technology)**
*期末项目 (Final Project) - 2025 秋季学期*
