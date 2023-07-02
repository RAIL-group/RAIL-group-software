
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <cgal_ex.hpp>
#include <pose.hpp>
#include <noisy.hpp>
#include <vector>
#include <array>
#include <map>
// #include <pair>
#include <vertex_graph.hpp>


typedef std::map<int, NoisyVertexPtr> VertIdMap;
PYBIND11_MAKE_OPAQUE(MapPairWall);
// PYBIND11_MAKE_OPAQUE(VertIdMap);
// PYBIND11_MAKE_OPAQUE(std::vector<int>);
// PYBIND11_MAKE_OPAQUE(std::vector<std::vector<int>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<Pose>>);
PYBIND11_MAKE_OPAQUE(std::vector<NoisyVertexPtr>);
PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<HallucinatedVertexDetection>>);
PYBIND11_MAKE_OPAQUE(std::vector<RealObservation>);
PYBIND11_MAKE_OPAQUE(std::vector<TopologyOperation>);


namespace py = pybind11;


PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
PYBIND11_MODULE(vertexnav_accel, m) {
    m.doc() = R"pbdoc(
        Pybind11 plugin for accelerated GapNav
        -----------------------

        .. currentmodule:: vertexnav_accel

        .. autosummary::
           :toctree: _generate

    )pbdoc";

    py::class_<FastWorld>(m, "FastWorld")
        .def(py::init<const VVD &, const VEV &>())
        .def("getVisPoly", &FastWorld::getVisPoly)
        .def("getVisPolyBounded", &FastWorld::getVisPolyBounded);

    py::class_<Seg>(m, "Seg")
        .def(py::init<double, double, double, double>());

    py::class_<Point2>(m, "Point2")
        .def(py::init<double, double>());

    py::class_<Pose, std::shared_ptr<Pose>>(m, "Pose")
        .def(py::init<double, double, double, int>(),
             py::arg("x"), py::arg("y"), py::arg("yaw") = 0, py::arg("robot_id") = 0)
        .def_readwrite("x", &Pose::x)
        .def_readwrite("y", &Pose::y)
        .def_readwrite("yaw", &Pose::yaw)
        .def_readwrite("index", &Pose::index)
        .def_readwrite("robot_id", &Pose::robot_id)
        .def("__rmul__", &Pose::rmul)
        .def("__mul__", &Pose::mul)
        .def_static("get_odom", &Pose::get_odom,
                    py::arg("p_new"), py::arg("p_old"));

    py::class_<NoisyDetectionType, std::shared_ptr<NoisyDetectionType>>(m, "NoisyDetectionType")
        .def(py::init<const char &>())
        .def(py::init<const Eigen::Vector4d &>())
        .def_readonly("prob_left_gap", &NoisyDetectionType::prob_left_gap)
        .def_readonly("prob_right_gap", &NoisyDetectionType::prob_right_gap)
        .def_readonly("label", &NoisyDetectionType::label)
        .def("__eq__", &NoisyDetectionType::eq);

    py::class_<NoisyVertex, std::shared_ptr<NoisyVertex>>(m, "NoisyVertex")
        .def(py::init<const Eigen::Vector2d &, const Eigen::Matrix2d &>())
        .def(py::init<const Eigen::Vector2d &,
             const Eigen::Matrix2d &, const Pose &>())
        .def_property_readonly("position", &NoisyVertex::get_position,
                               py::return_value_policy::copy)
        .def_property_readonly("cov", &NoisyVertex::get_cov,
                               py::return_value_policy::copy)
        .def("set_position", &NoisyVertex::set_position)
        .def("set_cov", &NoisyVertex::set_cov)
        .def_readonly("id", &NoisyVertex::id)
        .def_readonly("num_detections", &NoisyVertex::num_detections)
        .def_readonly("last_updated", &NoisyVertex::last_updated)
        .def_readwrite("is_active", &NoisyVertex::is_active)
        .def_readwrite("is_locked", &NoisyVertex::is_locked)
        .def("add_detection", &NoisyVertex::add_detection)
        .def("__eq__", &NoisyVertex::eq)
        .def("__hash__", &NoisyVertex::hash);

    py::class_<HallucinatedVertexDetection, std::shared_ptr<HallucinatedVertexDetection>>(m, "HallucinatedVertexDetection")
        .def(py::init<double, const Eigen::Vector2d &>(),
             py::arg("angle_rad"), py::arg("position"))
        .def(py::init<double, double, const Pose &>(),
             py::arg("angle_rad"), py::arg("range"), py::arg("r_pose"))
        .def_readonly("angle_rad", &HallucinatedVertexDetection::angle_rad)
        .def_readonly("position", &HallucinatedVertexDetection::position)
        .def_readonly("id", &HallucinatedVertexDetection::id)
        .def_property("detection_type",
                      &HallucinatedVertexDetection::get_detection_type,
                      &HallucinatedVertexDetection::set_detection_type)
        .def("__eq__", &HallucinatedVertexDetection::eq)
        .def("__hash__", &HallucinatedVertexDetection::hash);

    py::class_<NoisyVertexDetection, std::shared_ptr<NoisyVertexDetection>>(m, "NoisyVertexDetection")
        .def(py::init<double, double,
             const std::shared_ptr<NoisyDetectionType> &, const Eigen::Matrix2d &>(),
            py::arg("angle_rad"),
            py::arg("range"),
            py::arg("detection_type"),
            py::arg("cov_rt"))
        .def_readonly("position", &NoisyVertexDetection::position,
                      py::return_value_policy::copy)
        .def_readonly("cov", &NoisyVertexDetection::cov,
                      py::return_value_policy::copy)
        .def_readonly("cov_rt", &NoisyVertexDetection::cov_rt)
        .def_readonly("angle_rad", &NoisyVertexDetection::angle_rad)
        .def_readwrite("range", &NoisyVertexDetection::range)
        .def_readwrite("associated_vertex", &NoisyVertexDetection::associated_vertex_ptr)
        .def_property_readonly("detection_type",
                               &NoisyVertexDetection::get_detection_type)
        .def("add_association", &NoisyVertexDetection::add_association)
        .def("update_props", &NoisyVertexDetection::update_props);

    py::class_<NoisyWall, std::shared_ptr<NoisyWall>>(m, "NoisyWall")
        .def(py::init<
             const std::shared_ptr<NoisyVertex> &,
             const std::shared_ptr<NoisyVertex> &>(),
             py::arg("left_vertex"), py::arg("right_vertex"))
        .def_readwrite("is_active", &NoisyWall::is_active)
        .def_readwrite("prob_list", &NoisyWall::prob_list)
        .def_readonly("left_vertex", &NoisyWall::left_vertex_ptr)
        .def_readonly("right_vertex", &NoisyWall::right_vertex_ptr)
        .def_property_readonly("prob_exists", &NoisyWall::prob_exists)
        .def("_update", &NoisyWall::update)
        .def("__eq__", &NoisyWall::eq)
        .def("add_detection", &NoisyWall::add_detection,
             py::arg("left_detection"), py::arg("right_detection"),
             py::arg("false_positive_factor"));

    py::bind_map<MapPairWall>(m, "MapPairWall");
    // py::bind_map<VertIdMap>(m, "VertIdMap");
    // py::bind_vector<std::vector<int>>(m, "VectorInt");
    py::bind_vector<std::vector<std::shared_ptr<Pose>>>(m, "VectorPose");
    // py::bind_vector<std::vector<std::vector<int>>>(m, "VectorVectorInt");
    py::bind_vector<std::vector<std::shared_ptr<NoisyVertex>>>(m, "VectorVertices");
    py::bind_vector<std::vector<std::shared_ptr<HallucinatedVertexDetection>>>(m, "VectorHalDet");
    py::bind_vector<std::vector<RealObservation>>(m, "VectorRealObservation");
    py::bind_vector<std::vector<TopologyOperation>>(m, "VectorTopologyOperation");

    py::class_<ProposedWorldCore, std::shared_ptr<ProposedWorldCore>>(m, "ProposedWorldCore")
        .def_readonly("vertices", &ProposedWorldCore::vertices)
        .def_property_readonly("topology", &ProposedWorldCore::get_topology_py)
        .def_readonly("vertex_remapping", &ProposedWorldCore::vertex_remapping)
        .def_readonly("remapped_wall_dict", &ProposedWorldCore::remapped_wall_dict)
        .def("get_vertices_for_pose", &ProposedWorldCore::get_vertices_for_pose);


    py::class_<NoisyVertexInfo, std::shared_ptr<NoisyVertexInfo>>(m, "NoisyVertexInfo");

    py::class_<SamplingState, std::shared_ptr<SamplingState>>(m, "SamplingState")
        .def(py::init<const std::vector<NoisyVertexPtr>,
             const std::vector<std::vector<int>> &,
             float, const std::vector<std::shared_ptr<Pose>> &>(),
             py::arg("vertices"), py::arg("topology"), py::arg("log_prob"), py::arg("poses"))
        .def_readonly("log_prob", &SamplingState::log_prob)
        .def_property_readonly("topology", &SamplingState::get_topology_py);


    py::class_<ProbVertexGraph>(m, "ProbVertexGraph")
        .def(py::init<>())
        .def_readwrite("vertices", &ProbVertexGraph::vertices)
        //.def_readwrite("topology", &ProbVertexGraph::topology)
        .def_property("topology",
                      &ProbVertexGraph::get_topology_py,
                      &ProbVertexGraph::set_topology_py)
        .def_readwrite("walls", &ProbVertexGraph::walls)
        .def_readwrite("r_poses", &ProbVertexGraph::poses)
        .def_readwrite("odoms", &ProbVertexGraph::odoms)
        .def_readwrite("observations", &ProbVertexGraph::observations)
        .def_readwrite("false_positive_rate", &ProbVertexGraph::false_positive_rate)
        .def_readwrite("false_negative_rate", &ProbVertexGraph::false_negative_rate)
        .def_readwrite("disallowed_topology_operations", &ProbVertexGraph::disallowed_topology_operations)
        .def_readwrite("TOPOLOGY_LOCK_LOG_THRESH", &ProbVertexGraph::TOPOLOGY_LOCK_LOG_THRESH)
        .def_readwrite("DO_SLAM", &ProbVertexGraph::DO_SLAM)
        .def_readwrite("DO_SAMPLE_TOPOLOGY", &ProbVertexGraph::DO_SAMPLE_TOPOLOGY)
        .def_readwrite("PRIOR_NOISE", &ProbVertexGraph::PRIOR_NOISE)
        .def_readwrite("ODOMETRY_NOISE", &ProbVertexGraph::ODOMETRY_NOISE)
        .def_readwrite("CLUSTERING_NOISE", &ProbVertexGraph::CLUSTERING_NOISE)
        .def("get_state", &ProbVertexGraph::get_state)
        .def("set_state", &ProbVertexGraph::set_state)
        .def("add_observation_pose", &ProbVertexGraph::add_observation_pose)
        .def("add_observation_odom", &ProbVertexGraph::add_observation_odom)
        .def("_associate_detections", &ProbVertexGraph::associate_detections, py::arg("observation"), py::arg("r_pose"), py::arg("association_window") = -1)
        .def("compute_world_log_prob", &ProbVertexGraph::compute_world_log_prob)
        .def("compute_world_log_prob_full", &ProbVertexGraph::compute_world_log_prob_full)
        .def("perform_slam_update", &ProbVertexGraph::perform_slam_update)
        .def("get_proposed_world_fast", &ProbVertexGraph::get_proposed_world_py, py::arg("topology"));

    m.def("eval_mat_mul", &eval_mat_mul);
    m.def("get_vertex_remapping", &get_vertex_remapping);
    m.def("get_remapped_wall_dict", &get_remapped_wall_dict);
    m.def("compute_conservative_space_hallucinated",
          &compute_conservative_space_hallucinated);
    m.def("compute_conservative_space_observed",
          &compute_conservative_space_observed);
    m.def("compute_hypothetical_observation",
          &compute_hypothetical_observation);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
