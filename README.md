![Role|50](https://github.com/Tangzishun-Li/ParkourRL/blob/main/res/d1.png)![Role|50](https://github.com/Tangzishun-Li/ParkourRL/blob/main/res/d1.png)![Role|50](https://github.com/Tangzishun-Li/ParkourRL/blob/main/res/d1.png)

[Readme Chinese version](./README_cn.md)
---
# ParkourRL: Deep Reinforcement Learning for 2D Parkour Game

**ParkourRL** is a reinforcement learning research project that integrates a high-performance C++ physics engine with Python-based Deep Reinforcement Learning (DRL) algorithms. The project utilizes **Pybind11** to expose the C++ game environment to Python, enabling efficient training of agents using **DQN**, **A2C**, and **SARSA** algorithms to autonomously navigate complex obstacle courses.

📽️our project video: [Click to watch on YouTube](https://youtu.be/mt-hTwFStYk)\


---
## 📂 Project Structure

```text
ParkourRL/
├── CMakeLists.txt       # CMake build configuration for C++ extensions
├── config.py            # Global configuration parameters
├── render.py            # PyGame rendering engine for visualization
├── main.py              # Main management script (Build/Run management)
├── play.py              # Inference script for visualizing trained models
├── manual_play.py       # Human-in-the-loop testing script
├── train_DQN.py         # Training script using Deep Q-Network
├── train_A2C.py         # Training script using Advantage Actor-Critic
├── train_SARSA.py       # Training script using SARSA algorithm
├── src_fix.zip          # Fixed Level Environment Source Code (Archive)
├── res/                 # Game assets (Sprites, Backgrounds)
│   ├── bg001.png
│   ├── hero1.png
│   └── ... 
└── src/                 # Core C++ Source Code (Default: Random Environment)
    ├── GameEnv.h        # Environment header (Data structures, Collision logic)
    ├── GameEnv.cpp      # Environment implementation (Physics, Reward system)
    └── binding.cpp      # Pybind11 binding definitions

```
---

## 🛠️ Prerequisites & Installation

Before running the project, please ensure your system meets the following requirements.

### Step 0: System Environment Setup

#### 💻 For Windows Users

**Option A: Visual Studio (Recommended)**
This is the most stable method for Windows development.

1. Download and install **[Visual Studio Community 2022](https://visualstudio.microsoft.com/vs/community/)**.
2. During installation, select the workload: **"Desktop development with C++"**.
* *Note: This automatically installs the MSVC compiler and CMake.*


3. Ensure **[Python 3.8+](https://www.python.org/downloads/)** is installed and added to your system PATH.

**Option B: Chocolatey (Command Line)**
If you prefer using a package manager:

1. Open PowerShell as Administrator.
2. Run the following command:
```powershell
choco install python mingw cmake -y
```



#### 🍎 For macOS Users

macOS requires Xcode Command Line Tools for C++ compilation.

1. **Install Xcode Command Line Tools**:
```bash
xcode-select --install
```


2. **Install Homebrew** (if not installed):
```bash
/bin/bash -c "$(curl -fsSL [https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh](https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh))"
```


3. **Install Python & CMake**:
```bash
brew install python cmake
```



#### 🐧 For Linux (Ubuntu/Debian) Users

```bash
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv cmake g++ build-essential
```

---

### Step 1: Python Environment Initialization

1. **Clone the repository** to your local machine.
2. **Create and activate a virtual environment** (Recommended):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```


3. **Install Dependencies**:
```bash
pip install -r requirements.txt
# Or manually: pip install torch numpy gymnasium pygame pybind11
```



### Step 2: Build C++ Environment

The core game logic is written in C++ for performance. You must compile the C++ extension before running any Python scripts.

```bash
python main.py build
```

* **Success Indicator:** The terminal will show the compilation progress, and a shared library file (`.pyd` or `.so`) will be generated.
* **Important:** If you switch environment modes (see *Advanced Usage* below), you must re-run this command.

---

## 🚀 Usage

### 1. Manual Testing (Human Player)

Verify the game physics and rendering by playing manually.

* **Controls:** `W` (Jump), `S` (Crouch)

```bash
python manual_play.py
```

### 2. Train the Agent

Start training using one of the supported algorithms. Models are automatically saved to the `models/` directory.

* **Deep Q-Network (DQN):**
```bash
python train_DQN.py
```


* **Advantage Actor-Critic (A2C):**
```bash
python train_A2C.py
```


* **SARSA (On-Policy):**
```bash
python train_SARSA.py
```



### 3. Visualization (Inference)

The script will automatically detect the best available model (DQN/A2C/SARSA) and render the gameplay.

```bash
python play.py
```

*Optional arguments:* `python play.py --episodes 5 --algo DQN`

---

## ⚔️ Advanced: Environment Switching

The project supports two environment modes:

1. **Random Environment (Default):** Obstacles are generated randomly forever. Best for general training.
2. **Fixed Environment:** A pre-designed deterministic level. Best for reproducibility and specific challenge testing.

**How to switch to the Fixed Environment:**

1. Locate **`src_fix.zip`** in the project root directory.
2. Unzip the file and replace the existing **`src/`** folder with the one inside the zip.
3. **⚠️ Recompile the Environment:**
You must re-compile the C++ code for the changes to take effect:
```bash
python main.py build
```



*(To switch back to Random Environment, revert the `src/` folder changes using git or a backup, and recompile.)*

---

## ⚙️ Configuration

### Reward System Configuration

To modify the reinforcement learning reward structure, edit the macros in **`src/GameEnv.cpp`**:

```cpp
#define REWARD_PASS 10.0f 
#define REWARD_DEATH -0.0f 
#define REWARD_STEP 0.02f 
#define DAMAGE_TAKEN 10  
#define INITIAL_BLOOD 100  

#define PENALTY_SEVERE -50.0f
#define PENALTY_MODERATE -20.0f 
#define PENALTY_SLIGHT -10.0f 
```

**Important:** After modifying `src/GameEnv.cpp`, you must re-run `python main.py build`.

### Hyperparameters

To modify training parameters (Batch Size, Learning Rate, Epsilon), edit the corresponding Python script (e.g., `train_DQN.py`, `train_SARSA.py`).

---

## 📝 Changelog

### v1.2 (Current)

* **New Algorithms:** Added **A2C** (`train_A2C.py`) and **SARSA** (`train_SARSA.py`) implementations for comprehensive performance comparison.
* **State Space Augmentation:** Added relative velocity to the state vector, enabling the agent to distinguish between static and dynamic obstacles.
* **Network Upgrade:** Increased DQN hidden layer dimension (128 -> 256) for better feature extraction.
* **Reward Refactor:** Implemented a **Multi-level Penalty System** to provide granular feedback based on the type of error (Severity/Timing), replacing the generic death penalty.

### v1.1

* **Physics Fix:** Adjusted crouch duration frame logic (8,30 -> 8,40) to resolve collision issues with long hook obstacles.
---
## 🌸 Future work
1. Develop docker containers to facilitate running projects in different environments.
2. Implement more reinforcement learning algorithms, such as PPO.


---
## 🙏 Acknowledgments

Finally, we would like to express our sincere gratitude to Professor Lu and teaching assistants at Macau University of Science and Technology for their guidance and support in enabling us to complete this project.

---

## 📄 License & Copyright

**Copyright © Macau University of Science and Technology**
*Final Project - Fall Semester 2025*