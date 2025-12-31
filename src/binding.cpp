#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "GameEnv.h"

namespace py = pybind11;

PYBIND11_MODULE(GameEnv, m)
{
    // 绑定内部结构体 ObsInfo
    py::class_<RenderData::ObsInfo>(m, "ObsInfo")
        .def_readonly("type", &RenderData::ObsInfo::type)
        .def_readonly("x", &RenderData::ObsInfo::x)
        .def_readonly("y", &RenderData::ObsInfo::y)
        .def_readonly("imgindex", &RenderData::ObsInfo::imgindex);

    // 绑定 RenderData
    py::class_<RenderData>(m, "RenderData")
        .def_readonly("heroX", &RenderData::heroX)
        .def_readonly("heroY", &RenderData::heroY)
        .def_readonly("heroIndex", &RenderData::heroIndex)
        .def_readonly("heroDown", &RenderData::heroDown)
        .def_readonly("bgX", &RenderData::bgX)
        .def_readonly("obstacles", &RenderData::obstacles)
        .def_readonly("score", &RenderData::score)
        .def_readonly("heroBlood", &RenderData::heroBlood);

    // 绑定 StepResult
    py::class_<StepResult>(m, "StepResult")
        .def_readonly("obs", &StepResult::obs)
        .def_readonly("reward", &StepResult::reward)
        .def_readonly("done", &StepResult::done)
        .def_readonly("score", &StepResult::score);

    // 绑定 GameEnv 类
    py::class_<GameEnv>(m, "GameEnv")
        .def(py::init<>()) // 无参数构造
        .def("reset", &GameEnv::reset)
        .def("step", &GameEnv::step)
        .def("get_render_data", &GameEnv::get_render_data)
        .def("get_reward_pass", &GameEnv::get_reward_pass)
        .def("get_reward_hit", &GameEnv::get_reward_hit)
        .def("get_reward_death", &GameEnv::get_reward_death)
        .def("get_damage_taken", &GameEnv::get_damage_taken);
}