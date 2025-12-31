cmake_minimum_required(VERSION 3.15)
project(GameEnv)

set(CMAKE_CXX_STANDARD 17)

# --- 核心修复点 ---
# 原来的写法错误: find_package(PythonComponents ...)
# 正确写法如下:
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# 自动寻找 Pybind11 (通过 Python 询问安装路径)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED CONFIG)

# 定义编译目标
pybind11_add_module(GameEnv src/binding.cpp src/GameEnv.cpp)

# 针对不同系统的优化选项
if(MSVC)
    target_compile_options(GameEnv PRIVATE /O2 /MP)
else()
    target_compile_options(GameEnv PRIVATE -O3 -fPIC)
endif()