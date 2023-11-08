cmake_minimum_required(VERSION 3.18)

project(fmfusion LANGUAGES C CXX)

set(Open3D_ROOT "/home/lch/Code_ws/Open3D/install")

find_package(jsoncpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Open3D REQUIRED PATHS ${Open3D_ROOT})
find_package(OpenCV REQUIRED)

##
include_directories(${OpenCV_INCLUDE_DIRS})
##

add_library(libfmfusion SHARED
    Detection.cpp
    Instance.cpp
    SceneGraph.cpp
    Utility.cpp
)
target_link_libraries(libfmfusion PRIVATE Open3D::Open3D ${OpenCV_LIBS} jsoncpp_lib)

add_executable(IntegrateRGBD)
target_sources(IntegrateRGBD PRIVATE IntegrateRGBD.cpp)
target_link_libraries(IntegrateRGBD PRIVATE Open3D::Open3D ${OpenCV_LIBS} libfmfusion)

add_executable(IntegrateInstanceMap)
target_sources(IntegrateInstanceMap PRIVATE IntegrateInstanceMap.cpp)
target_link_libraries(IntegrateInstanceMap PRIVATE Open3D::Open3D ${OpenCV_LIBS} libfmfusion)

add_executable(GraphMatchNode)
target_sources(GraphMatchNode PRIVATE GraphMatch.cpp)
target_link_libraries(GraphMatchNode PRIVATE Open3D::Open3D ${OpenCV_LIBS} libfmfusion)
