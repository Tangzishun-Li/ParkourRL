#include "GameEnv.h"
#include <cstdlib>
#include <ctime>
#include <algorithm>

// ==========================================
// RL 超参数配置区 (多级惩罚版)
// ==========================================
#define REWARD_PASS 10.0f  // 通过奖励
#define REWARD_DEATH -0.0f // 死亡额外惩罚 (建议保持0或很小，让具体操作的惩罚占主导)
#define REWARD_STEP 0.02f  // 步数奖励
#define DAMAGE_TAKEN 10    // 每次扣血
#define INITIAL_BLOOD 100  // 初始血量

// [新增] 多级惩罚定义
#define PENALTY_SEVERE -50.0f   // 重度惩罚：反向操作
#define PENALTY_MODERATE -20.0f // 中度惩罚：不操作
#define PENALTY_SLIGHT -10.0f   // 轻微惩罚：操作正确但时机不对

// 接口实现
float GameEnv::get_reward_pass() { return REWARD_PASS; }
float GameEnv::get_reward_hit() { return PENALTY_MODERATE; }
float GameEnv::get_reward_death() { return REWARD_DEATH; }
int GameEnv::get_damage_taken() { return DAMAGE_TAKEN; }

// 构造函数
GameEnv::GameEnv()
{
    bgSpeed[0] = 1;
    bgSpeed[1] = 2;
    bgSpeed[2] = 4;
    srand((unsigned int)time(NULL));
    reset();
}

void GameEnv::reset()
{
    heroX = (int)(1012 * 0.5 - HERO_W * 0.5);
    heroY = 345 - HERO_H;

    heroBlood = INITIAL_BLOOD;
    score = 0;
    heroIndex = 0;
    heroJump = false;
    heroDown = false;
    heroJumpOff = -6;
    jumpHeightMax = 345 - HERO_H - 120;

    frameCount = 0;
    currentStepPenalty = 0.0f;

    for (int i = 0; i < 3; i++)
        bgX[i] = 0;
    for (int i = 0; i < 10; i++)
        obstacles[i].exist = false;

    // [修改] 初始化随机频率
    enemyFre = 60;
}

StepResult GameEnv::step(int action)
{
    int oldScore = score;
    int oldBlood = heroBlood;

    // [新增] 每一步开始前重置当帧惩罚
    currentStepPenalty = 0.0f;

    // 动作处理逻辑
    if (action == 1 && !heroJump && !heroDown) // 跳
    {
        heroJump = true;
        heroJumpOff = -6;
    }
    else if (action == 2 && !heroJump && !heroDown) // 蹲
    {
        heroDown = true;
        heroIndex = 0;
    }

    update_physics(); // 这里面会调用 check_hit 更新 currentStepPenalty

    // --- 奖励判定逻辑 (关键修改：死亡判定放在最后) ---
    float r = REWARD_STEP;

    // 1. 优先判定是否得分
    if (score > oldScore)
    {
        r = REWARD_PASS;
    }

    // 2. 其次判定是否受伤 (受伤的惩罚 覆盖 得分的奖励)
    if (heroBlood < oldBlood)
    {
        // 如果扣血了，使用智能裁判计算出的具体惩罚 (例如 -20)
        // 这样即使死掉，模型也知道是因为"严重错误"导致的
        r = (currentStepPenalty != 0.0f) ? currentStepPenalty : PENALTY_MODERATE;
    }

    // 3. 最后判定死亡 (叠加逻辑)
    if (heroBlood <= 0)
    {
        // 死亡惩罚是"最后"加上的
        // 如果刚才因为撞挂钩拿了 -20，这里加上 -0，总分还是 -20 (保留了死因)
        // 之前逻辑是直接返回 REWARD_DEATH (0)，导致模型不知道为什么死
        r += REWARD_DEATH;
    }

    return {get_obs(), r, heroBlood <= 0, score};
}

void GameEnv::update_physics()
{
    // 1. 背景滚动
    for (int i = 0; i < 3; i++)
    {
        bgX[i] -= bgSpeed[i];
        if (bgX[i] < -1012)
            bgX[i] = 0;
    }

    // 2. 英雄跳跃逻辑
    if (heroJump)
    {
        if (heroY < jumpHeightMax)
            heroJumpOff = 6;
        heroY += heroJumpOff;
        if (heroY > 345 - HERO_H)
        {
            heroJump = false;
            heroJumpOff = -6;
        }
    }
    // 3. 英雄下蹲逻辑
    else if (heroDown)
    {
        static int count = 0;
        int delays[2] = {8, 40};
        count++;
        if (count >= delays[heroIndex])
        {
            count = 0;
            heroIndex++;
            if (heroIndex >= 2)
            {
                heroIndex = 0;
                heroDown = false;
            }
        }
    }
    else
        heroIndex = (heroIndex + 1) % 12;

    // 4. [修改] 障碍物生成 - 恢复随机逻辑
    enemyFre--;
    if (enemyFre <= 0)
    {
        // 随机生成间隔
        enemyFre = 120 + rand() % 60;

        // 随机生成类型 0, 1, 2
        int type = rand() % 3;
        create_obstacle(type);
    }

    // 5. 障碍物移动与计分
    for (int i = 0; i < 10; i++)
    {
        if (obstacles[i].exist)
        {
            obstacles[i].x -= (obstacles[i].speed + bgSpeed[2]);
            int type_idx = (obstacles[i].type >= 2) ? 2 : obstacles[i].type;

            if (obstacles[i].x < -(OBS_W[type_idx] * 2))
                obstacles[i].exist = false;

            obstacles[i].imgindex++;

            if (!obstacles[i].passed && !obstacles[i].hited &&
                (obstacles[i].x + OBS_W[type_idx] < heroX))
            {
                score++;
                obstacles[i].passed = true;
            }
        }
    }
    check_hit();
}

void GameEnv::create_obstacle(int type)
{
    int i;
    for (i = 0; i < 10; i++)
        if (!obstacles[i].exist)
            break;
    if (i >= 10)
        return;

    obstacles[i].exist = true;
    obstacles[i].hited = false;
    obstacles[i].passed = false;
    obstacles[i].imgindex = 0;
    obstacles[i].type = type;
    obstacles[i].x = 1012;

    if (obstacles[i].type >= 2) // 挂钩
    {
        obstacles[i].y = 0;
        obstacles[i].speed = 0;
    }
    else // 地面
    {
        int type_idx = obstacles[i].type;
        obstacles[i].y = 345 + 5 - OBS_H[type_idx];
        obstacles[i].speed = (obstacles[i].type == 1) ? 4 : 0;
    }
}

// ==========================================
// 核心逻辑：智能裁判系统 check_hit (完全保持不变)
// ==========================================
void GameEnv::check_hit()
{
    int off = 10;
    bool intersect = false;
    int hitType = -1;

    for (int i = 0; i < 10; i++)
    {
        if (obstacles[i].exist && !obstacles[i].hited)
        {
            int h_x, h_y, h_w, h_h;

            if (!heroDown)
            {
                h_x = heroX + off;
                h_y = heroY + 10;
                h_w = HERO_W - off * 2;
                h_h = HERO_H - 10;
            }
            else
            {
                h_x = heroX + off;
                h_y = 345 - HERO_DOWN_H + 10;
                h_w = HERO_DOWN_W - off * 2;
                h_h = HERO_DOWN_H - 10;
            }

            int type_idx = (obstacles[i].type >= 2) ? 2 : obstacles[i].type;
            int obs_x = obstacles[i].x + off;
            int obs_y = obstacles[i].y;
            int obs_w = OBS_W[type_idx] - off * 2;
            int obs_h = OBS_H[type_idx] - 10;

            if (rectIntersect(h_x, h_y, h_w, h_h, obs_x, obs_y, obs_w, obs_h))
            {
                heroBlood -= DAMAGE_TAKEN;
                obstacles[i].hited = true;
                intersect = true;
                hitType = obstacles[i].type;
                break;
            }
        }
    }

    if (intersect)
    {
        if (hitType == 0 || hitType == 1)
        {
            if (heroDown)
                currentStepPenalty = PENALTY_SEVERE;
            else if (!heroJump && !heroDown)
                currentStepPenalty = PENALTY_MODERATE;
            else if (heroJump)
                currentStepPenalty = PENALTY_SLIGHT;
        }
        else if (hitType >= 2)
        {
            if (heroJump)
                currentStepPenalty = PENALTY_SEVERE;
            else if (!heroJump && !heroDown)
                currentStepPenalty = PENALTY_MODERATE;
            else if (heroDown)
                currentStepPenalty = PENALTY_SLIGHT;
        }
    }
}

bool GameEnv::rectIntersect(int x1, int y1, int w1, int h1, int x2, int y2, int w2, int h2)
{
    if (x1 + w1 < x2)
        return false;
    if (x1 > x2 + w2)
        return false;
    if (y1 + h1 < y2)
        return false;
    if (y1 > y2 + h2)
        return false;
    return true;
}

std::vector<float> GameEnv::get_obs()
{
    float nearest_dist = 1.0f;
    float nearest_y = 0.0f;
    float relative_speed = 0.0f;

    for (int i = 0; i < 10; i++)
    {
        if (obstacles[i].exist && obstacles[i].x > heroX)
        {
            nearest_dist = (float)(obstacles[i].x - heroX) / 1012.0f;
            nearest_y = (float)obstacles[i].y / 396.0f;
            float closing_speed = (float)(obstacles[i].speed + bgSpeed[2]);
            relative_speed = closing_speed / 10.0f;
            break;
        }
    }

    if (relative_speed == 0.0f)
        relative_speed = (float)bgSpeed[2] / 20.0f;

    return {
        (float)heroY / 396.0f,
        nearest_dist,
        nearest_y,
        (heroJump ? 1.0f : 0.0f),
        (heroDown ? 1.0f : 0.0f),
        relative_speed};
}

RenderData GameEnv::get_render_data()
{
    RenderData data;
    data.heroX = heroX;
    data.heroY = heroY;
    data.heroIndex = heroIndex;
    data.heroDown = heroDown;
    data.score = score;
    data.heroBlood = heroBlood;

    for (int x : bgX)
        data.bgX.push_back(x);

    for (int i = 0; i < 10; i++)
    {
        if (obstacles[i].exist)
        {
            data.obstacles.push_back({obstacles[i].type, obstacles[i].x, obstacles[i].y, obstacles[i].imgindex});
        }
    }
    return data;
}