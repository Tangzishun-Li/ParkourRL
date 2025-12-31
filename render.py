import pygame
import os
from config import Config

class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
        pygame.display.set_caption(Config.CAPTION)
        self.clock = pygame.time.Clock()
        # 移除字体加载，因为我们要用图片显示分数
        # self.font = pygame.font.SysFont("Arial", Config.FONT_SIZE, bold=True)
        
        self.imgs = {}
        self._load_resources()
        self.bg_offsets = [0, 119, 330] 

    def _load_resources(self):
        print("[Renderer] 正在加载图片资源...")
        def load(name):
            path = os.path.join(Config.RES_DIR, name)
            if not os.path.exists(path): return pygame.Surface((30,30))
            return pygame.image.load(path).convert_alpha()

        self.imgs['bg'] = [load(f"bg{i+1:03d}.png") for i in range(3)]
        self.imgs['hero'] = [load(f"hero{i+1}.png") for i in range(12)]
        self.imgs['hero_down'] = [load("d1.png"), load("d2.png")]

        # 障碍物
        raw_t1 = load("t1.png")
        self.imgs['obs_0'] = [pygame.transform.scale(raw_t1, (60, 50))] # 乌龟
        
        self.imgs['obs_1'] = []
        for i in range(6):
            raw_lion = load(f"p{i+1}.png")
            self.imgs['obs_1'].append(pygame.transform.scale(raw_lion, (80, 70))) # 狮子

        self.imgs['obs_2'] = []
        for i in range(4):
            raw_hook = load(f"h{i+1}.png")
            self.imgs['obs_2'].append(pygame.transform.scale(raw_hook, (63, 260))) # 挂钩

        # --- [新增] 加载数字图片 ---
        # 假设图片在 res/sz/0.png - 9.png
        self.imgs['sz'] = []
        for i in range(10):
            # 注意路径兼容性：res/sz/0.png
            self.imgs['sz'].append(load(f"sz/{i}.png"))

    def draw(self, data):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False

        self.screen.fill((255, 255, 255)) 

        # 1. 绘制背景
        for i, x in enumerate(data.bgX):
            y_pos = self.bg_offsets[i]
            self.screen.blit(self.imgs['bg'][i], (x, y_pos))
            if x < 0: self.screen.blit(self.imgs['bg'][i], (x + 1012, y_pos))

        # 2. 绘制英雄
        GROUND_Y = 345 
        VISUAL_OFFSET_Y = 0 
        if data.heroDown:
            img = self.imgs['hero_down'][data.heroIndex % 2]
            draw_y = GROUND_Y - img.get_height() + VISUAL_OFFSET_Y
            self.screen.blit(img, (data.heroX, draw_y))
        else:
            img = self.imgs['hero'][data.heroIndex % 12]
            if data.heroY < (GROUND_Y - 90): 
                cpp_height_assumption = 90
                diff = cpp_height_assumption - img.get_height()
                draw_y = data.heroY + diff + VISUAL_OFFSET_Y
            else:
                draw_y = GROUND_Y - img.get_height() + VISUAL_OFFSET_Y
            self.screen.blit(img, (data.heroX, draw_y))

        # 3. 绘制障碍物 (层级：英雄后面，或者覆盖英雄，根据你上一次的要求是覆盖)
        for obs in data.obstacles:
            if obs.type == 0: frames = self.imgs['obs_0']
            elif obs.type == 1: frames = self.imgs['obs_1']
            else: 
                hook_idx = obs.type - 2
                frames = self.imgs['obs_2']
                idx = hook_idx if 0 <= hook_idx < len(frames) else 0
                self.screen.blit(frames[idx], (obs.x, obs.y))
                continue
            img = frames[obs.imgindex % len(frames)]
            self.screen.blit(img, (obs.x, obs.y))

        # --- [UI 还原] ---
        
        # 4. 绘制经典血条
        # 原逻辑：drawBloodBar(10, 10, 200, 10, 2, BLUE, DARKGRAY, RED, percent)
        bar_x, bar_y = 10, 10
        bar_w, bar_h = 200, 10
        border_w = 2
        
        # (1) 绘制空底 (DarkGray)
        pygame.draw.rect(self.screen, (64, 64, 64), (bar_x, bar_y, bar_w, bar_h))
        
        # (2) 绘制红色血量
        if data.heroBlood > 0:
            fill_width = int(bar_w * (data.heroBlood / 100.0))
            # 稍微内缩一点，避免压住边框
            inner_rect = (bar_x + 1, bar_y + 1, fill_width - 2, bar_h - 2)
            pygame.draw.rect(self.screen, (255, 0, 0), inner_rect)
            
        # (3) 绘制蓝色边框 (Blue)
        pygame.draw.rect(self.screen, (0, 0, 255), (bar_x, bar_y, bar_w, bar_h), border_w)

        # 5. 绘制图片数字分数
        # 原逻辑：sx=20, sy=25, 遍历每一位数字绘制
        score_str = str(data.score)
        sx, sy = 20, 25
        
        for char in score_str:
            digit = int(char)
            if 0 <= digit <= 9:
                num_img = self.imgs['sz'][digit]
                self.screen.blit(num_img, (sx, sy))
                sx += num_img.get_width() + 5 # 间距 5

        pygame.display.flip()
        self.clock.tick(Config.FPS)
        return True