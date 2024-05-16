cmake_minimum_required(VERSION 3.18)

project(fmfusion LANGUAGES C CXX)

set(Open3D_DIR "/home/cliuci/code_ws/Open3D_CPU/install")
set(Libtorch_ROOT "/home/cliuci/tools/libtorch")
option(ENABLE_TORCH ON)

find_package(jsoncpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Open3D REQUIRED PATHS ${Open3D_DIR})
find_package(OpenCV REQUIRED)

# print open3d directories
message(STATUS "Open3D_INCLUDE_DIRS: ${Open3D_INCLUDE_DIRS}")
message(STATUS "Open3D_LIBRARIES: ${Open3D_LIBRARIES}")

# FMFusion
if (ENABLE_TORCH)
    add_subdirectory(tokenizer)
    find_package(Torch REQUIRED PATHS ${Libtorch_ROOT})

    add_library(libfmfusion SHARED
        mapping/Detection.cpp
        mapping/Instance.cpp
        mapping/Edge.cpp
        cluster/PoseGraph.cpp
        SceneGraph.cpp
        Utility.cpp
        sgloop/LoopDetector.cpp
    )
    target_link_libraries(libfmfusion PRIVATE Open3D::Open3D jsoncpp_lib ${OpenCV_LIBS} ${TORCH_LIBRARIES} tokenizer)
    target_include_directories(libfmfusion PRIVATE ${TORCH_INCLUDE_DIRS})
    set_target_properties(libfmfusion PROPERTIES CXX_STANDARD 17)
    message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")


else()
    add_library(libfmfusion SHARED
        mapping/Detection.cpp
        mapping/Instance.cpp
        cluster/PoseGraph.cpp
        SceneGraph.cpp
        Utility.cpp
    )
    target_link_libraries(libfmfusion PRIVATE Open3D::Open3D jsoncpp_lib ${OpenCV_LIBS})
endif()

##
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
##


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
set_target_properties(TestLoop PROPERTIES CXX_STANDARD 17)
