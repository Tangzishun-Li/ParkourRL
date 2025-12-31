import os

class Config:
    # 窗口与渲染参数
    SCREEN_WIDTH = 1012
    SCREEN_HEIGHT = 396
    FPS = 60
    CAPTION = "AI Parkour - Cross Platform"
    
    # 资源路径
    RES_DIR = "res"
    
    # 字体设置 (如果没有字体文件则使用默认)
    FONT_SIZE = 30
    
    # 调试模式
    DEBUG_DRAW = False  # 是否绘制碰撞箱轮廓