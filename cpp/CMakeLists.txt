cmake_minimum_required(VERSION 3.18)

project(fmfusion LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
# set(Open3D_DIR "/home/cliuci/code_ws/Open3d_CPU/install")
# set(Torch_DIR "/home/cliuci/tools/libtorch")
option(ENABLE_TORCH ON)
option(LOOP_DETECTION ON)
add_definitions(-D _GLIBCXX_USE_CXX11_ABI=1)
find_package(jsoncpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Open3D REQUIRED)
find_package(OpenCV REQUIRED)
add_subdirectory(thirdparty/G3Reg)

##
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${Open3D_INCLUDE_DIRS})
include_directories(/usr/include/eigen3)
##
# print open3d directories
message(STATUS "Open3D_INCLUDE_DIRS: ${Open3D_INCLUDE_DIRS}")
message(STATUS "Open3D_LIBRARIES: ${Open3D_LIBRARIES}")
message(STATUS "Eigen library: ${EIGEN_LIBRARIES}")

# FMFusion
if (ENABLE_TORCH)
    # set(SPDLOG_ROOT "/home/cliuci/tools/spdlog/install")
    set(LIB_UTF8PROC "/usr/local/lib/libutf8proc.so")

    add_subdirectory(tokenizer)
    add_library(libfmfusion SHARED
            mapping/Detection.cpp
            mapping/Instance.cpp
            tools/Visualization.cpp
            cluster/PoseGraph.cpp
            SceneGraph.cpp
            Utility.cpp
            sgloop/Graph.cpp
            sgloop/LoopDetector.cpp
    )
    target_link_libraries(libfmfusion PRIVATE jsoncpp_lib ${OpenCV_LIBS} tokenizer Open3D::Open3D G3REG::g3reg)

    find_package(Torch REQUIRED)
    message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
    message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
    target_link_libraries(libfmfusion PRIVATE ${TORCH_LIBRARIES})
    target_include_directories(libfmfusion PRIVATE ${TORCH_INCLUDE_DIRS})

else ()
    add_library(libfmfusion SHARED
            mapping/Detection.cpp
            mapping/Instance.cpp
            sgloop/Graph.cpp
            tools/Visualization.cpp
            cluster/PoseGraph.cpp
            SceneGraph.cpp
            Utility.cpp
    )
    target_link_libraries(libfmfusion PRIVATE Open3D::Open3D jsoncpp_lib ${OpenCV_LIBS})
endif ()


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

add_executable(MapMerger)
target_sources(MapMerger PRIVATE MapMerger.cpp)
target_link_libraries(MapMerger PRIVATE Open3D::Open3D ${OpenCV_LIBS} libfmfusion)

if (LOOP_DETECTION)
    add_executable(TestLoop)
    target_sources(TestLoop PRIVATE TestLoop.cpp)
    target_link_libraries(TestLoop PRIVATE Open3D::Open3D libfmfusion ${TORCH_LIBRARIES} G3REG::g3reg)
endif ()
