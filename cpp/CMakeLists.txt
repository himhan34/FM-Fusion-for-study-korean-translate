cmake_minimum_required(VERSION 3.18)

project(fmfusion LANGUAGES C CXX)

set(Open3D_ROOT "/home/lch/Code_ws/Open3D/install")
set(Libtorch_ROOT "/home/cliuci/tools/libtorch")
set(CMAKE_CXX_STANDARD 17)

find_package(jsoncpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Open3D REQUIRED PATHS ${Open3D_ROOT})
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED PATHS ${Libtorch_ROOT})


##
include_directories(${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
##

add_library(libfmfusion SHARED
    Detection.cpp
    Instance.cpp
    SceneGraph.cpp
    Utility.cpp
    cluster/PoseGraph.cpp
    # loop_detect/LoopDetector.cpp
)
target_link_libraries(libfmfusion PRIVATE Open3D::Open3D jsoncpp_lib ${OpenCV_LIBS})
# set_target_properties(libfmfusion PROPERTIES CXX_STANDARD 17)

# Executables
add_executable(IntegrateRGBD)
target_sources(IntegrateRGBD PRIVATE IntegrateRGBD.cpp)
target_link_libraries(IntegrateRGBD PRIVATE Open3D::Open3D ${OpenCV_LIBS} libfmfusion)

add_executable(IntegrateInstanceMap)
target_sources(IntegrateInstanceMap PRIVATE IntegrateInstanceMap.cpp)
target_link_libraries(IntegrateInstanceMap PRIVATE Open3D::Open3D ${OpenCV_LIBS} libfmfusion)

add_executable(GraphMatchNode)
target_sources(GraphMatchNode PRIVATE GraphMatch.cpp)
target_link_libraries(GraphMatchNode PRIVATE Open3D::Open3D ${OpenCV_LIBS} libfmfusion)

# add_executable(MapMerger)
# target_sources(MapMerger PRIVATE MapMerger.cpp)
# target_link_libraries(MapMerger PRIVATE Open3D::Open3D ${OpenCV_LIBS} libfmfusion)

add_executable(TestLoop)
target_sources(TestLoop PRIVATE TestLoop.cpp)
target_link_libraries(TestLoop PRIVATE Open3D::Open3D libfmfusion ${TORCH_LIBRARIES})
# set_target_properties(TestLoop PROPERTIES CXX_STANDARD 17)
