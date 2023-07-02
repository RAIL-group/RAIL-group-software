cmake_minimum_required(VERSION 2.8.12)
project(vertexnav_accel)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Release")

# Download pybind11
find_package(Git QUIET)
if(GIT_FOUND AND NOT EXISTS "${PROJECT_SOURCE_DIR}/pybind11")
# Update submodules as needed
    option(GIT_CLONE "Clone during build" ON)
    if(GIT_CLONE)
        message(STATUS "Clone update")
        execute_process(COMMAND ${GIT_EXECUTABLE} clone --branch v2.2.0 https://github.com/pybind/pybind11.git
                        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_CLONE_RESULT)
        if(NOT GIT_CLONE_RESULT EQUAL "0")
            message(FATAL_ERROR "git clone failed with ${GIT_CLONE_RESULT}.")
        endif()
    endif()
endif()

# Why do I need this? pybind11?
link_directories(/usr/local/lib)

# Locate the CGAL package
find_package(CGAL REQUIRED COMPONENTS Core)
include_directories(${CGAL_INCLUDE_DIRS})

# Use GTSAM
find_package(GTSAM REQUIRED)
include_directories(${GTSAM_INCLUDE_DIR})
include_directories(/gtsam/gtsam/3rdparty/Eigen)
include_directories(/home/gjstein/research/external/gtsam/gtsam/3rdparty/Eigen)

# Make the library that pybind will link against
include_directories(src)

# Build the CGAL library
add_library(cgal_ex_lib STATIC src/cgal_ex.cpp)
set_target_properties(cgal_ex_lib PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
target_link_libraries(cgal_ex_lib PRIVATE CGAL::CGAL)

# Build separate library for vertexnav_accel
add_library(vertexnav_accel_noisy STATIC src/pose.cpp src/noisy.cpp src/Hungarian/Hungarian.cpp src/vertex_graph.cpp)
set_target_properties(vertexnav_accel_noisy PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
target_link_libraries(vertexnav_accel_noisy gtsam)


# Build the python library
add_subdirectory(pybind11)

pybind11_add_module(vertexnav_accel NO_EXTRAS src/main.cpp)
target_link_libraries(vertexnav_accel PRIVATE cgal_ex_lib vertexnav_accel_noisy)
