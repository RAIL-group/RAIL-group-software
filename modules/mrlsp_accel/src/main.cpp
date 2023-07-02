#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <vector>
#include <array>
#include <map>
#include "pouct.hpp"


namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

PYBIND11_MODULE(mrlsp_accel, m) {
    m.doc() = R"pbdoc(
        Pybind11 plugin for demonstrating C++ features
        -----------------------

        .. currentmodule:: pycpp_examples

        .. autosummary::
           :toctree: _generate

    )pbdoc";

    // Question: Why do we need std::shared_ptr<FrontierData> here? Isn't this the place for inheritance?
    py::class_<FrontierDataMR, std::shared_ptr<FrontierDataMR>>(m, "FrontierDataMR")
        .def(py::init<double, double, double, long, bool>(),
             py::arg("prob_feasible"),
             py::arg("delta_success_cost"),
             py::arg("exploration_cost"),
             py::arg("hash_id"),
             py::arg("is_from_last_chosen") = 0)
        .def("__hash__", &FrontierDataMR::get_hash);

    // add , std::shared_ptr<State> below if error pops off
    py::class_<State, std::shared_ptr<State>>(m, "State_cpp")
        .def(py::init<
                 int,
                 std::vector<FrontierDataMRPtr> &,
                 std::map<std::pair<int, long>, double>,
                 std::map<long, double> &,
                 std::map<std::pair<long, long>, double> &>(),
             py::arg("num_robots"),
             py::arg("unexplored_frontiers"),
             py::arg("robot_distances"),
             py::arg("goal_distances"),
             py::arg("frontier_distances"))
        .def_readonly("num_robots", &State::num_robots)
        .def_readonly("unexplored_frontiers", &State::unexplored_frontiers)
        .def_readonly("robot_distances", &State::robot_distances)
        .def_readonly("goal_frontiers", &State::goal_frontiers)
        .def_readonly("q_t", &State::q_t)
        .def_readonly("unexplored_frontiers_hash", &State::unexplored_frontiers_hash)
        .def("remove_frontier_from_unexplored_frontiers", &State::remove_frontier_from_unexplored_frontiers)
        .def("add_frontier_to_goal_frontiers", &State::add_frontier_to_goal_frontiers)
        .def("change_rd_for_robot", &State::change_rd_for_robot)
        .def("copy_state", &State::copy_state)
        .def("get_actions", &State::get_actions);

    m.def("find_best_joint_action_accel", &find_best_joint_action_accel);
    m.def("find_progress_and_frontier_for_robot_cpp", &find_progress_and_frontier_for_robot);
    m.def("get_frontier_of_knowledge_and_time_cpp", &get_frontier_of_knowledge_and_time);
    m.def("get_frontier_time_by_triangle_formation_cpp", &get_frontier_time_by_triangle_formation);
    m.def("find_q_t_for_action_cpp", &find_q_t_for_action);
    m.def("get_time_from_qt_cpp", &get_time_from_qt);
    m.def("move_robots_cpp", &move_robots);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
