#ifndef GAME_ENV_H
#define GAME_ENV_H

#include <vector>
#include <cmath>

// ==========================================
// 1. 数据传输结构体 (C++ -> Python)
// ==========================================

struct RenderData
{
    int heroX, heroY;
    int heroIndex;
    bool heroDown;
    std::vector<int> bgX; // 背景坐标

    struct ObsInfo
    {
        int type; // 0:乌龟, 1:狮子, 2+:挂钩
        int x, y;
        int imgindex; // 动画帧索引
    };
    std::vector<ObsInfo> obstacles;

    int score;
    int heroBlood;
};

struct StepResult
{
    std::vector<float> obs;
    float reward;
    bool done;
    int score;
};

// ==========================================
// 2. 环境类定义
// ==========================================

class GameEnv
{
public:
    GameEnv();

    void reset();
    StepResult step(int action);
    RenderData get_render_data(); // 获取渲染数据

    // 导出超参数
    float get_reward_pass();
    float get_reward_hit();
    float get_reward_death();
    int get_damage_taken();

private:
    void update_physics();
    void create_obstacle(int type);
    void check_hit();
    std::vector<float> get_obs();
    float currentStepPenalty;

    // --- [关键配置] 碰撞箱尺寸常量 ---
    // 英雄尺寸
    const int HERO_W = 60;
    const int HERO_H = 90;
    const int HERO_DOWN_W = 60;
    const int HERO_DOWN_H = 50;

    // 障碍物尺寸配置:
    // [0]乌龟: 60x50
    // [1]狮子: 70x60
    // [2]挂钩: 保持 63x260
    const int OBS_W[3] = {30, 50, 60};
    const int OBS_H[3] = {30, 50, 290};

    // --- 游戏状态变量 ---
    int heroX, heroY;
    int heroBlood;
    int score;
    int heroIndex;

    bool heroJump;
    bool heroDown;
    int heroJumpOff;
    int jumpHeightMax;

    int frameCount;

    // [修改] 恢复随机频率计数器，移除脚本相关变量
    int enemyFre;

    int bgX[3];
    int bgSpeed[3];

    struct Obstacle
    {
        int type;
        int imgindex;
        int x, y;
        int speed;
        bool exist;
        bool hited;
        bool passed;
    };
    Obstacle obstacles[10];

    // --- 辅助函数 ---
    bool rectIntersect(int x1, int y1, int w1, int h1, int x2, int y2, int w2, int h2);
};

#endif