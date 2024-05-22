cmake_minimum_required(VERSION 3.18)

project(fmfusion LANGUAGES CXX VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# set(Open3D_DIR "/home/cliuci/code_ws/Open3d_CPU/install")
set(CMAKE_PREFIX_PATH "/home/cliuci/tools/libtorch")
option(ENABLE_TORCH ON)
option(LOOP_DETECTION ON)
option(INSTALL_FMFUSION ON)

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

set(FMFUSION_SRC 
    mapping/Detection.cpp
    mapping/Instance.cpp
    cluster/PoseGraph.cpp
    tools/Visualization.cpp
    SceneGraph.cpp
    Utility.cpp
)    

# FMFusion
if (ENABLE_TORCH)
    set(LIB_UTF8PROC "/usr/local/lib/libutf8proc.so")
    add_subdirectory(tokenizer)
    find_package(Torch REQUIRED)

    list(APPEND FMFUSION_SRC
        tools/IO.cpp
        sgloop/Graph.cpp
        sgloop/LoopDetector.cpp
        sgloop/ShapeEncoder.cpp)
    add_library(fmfusion SHARED ${FMFUSION_SRC})

    target_link_libraries(fmfusion PRIVATE Open3D::Open3D jsoncpp_lib ${OpenCV_LIBS} ${TORCH_LIBRARIES} tokenizer)
    target_include_directories(fmfusion PRIVATE ${TORCH_INCLUDE_DIRS} thirdparty/extensions)
    message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
    message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
else()

    add_library(fmfusion SHARED ${FMFUSION_SRC})
    target_link_libraries(fmfusion PRIVATE Open3D::Open3D jsoncpp_lib ${OpenCV_LIBS})
endif()

##
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
##

# Executables
add_executable(IntegrateRGBD)
target_sources(IntegrateRGBD PRIVATE IntegrateRGBD.cpp)
target_link_libraries(IntegrateRGBD PRIVATE Open3D::Open3D ${OpenCV_LIBS} fmfusion)

add_executable(IntegrateInstanceMap)
target_sources(IntegrateInstanceMap PRIVATE IntegrateInstanceMap.cpp)
target_link_libraries(IntegrateInstanceMap PRIVATE Open3D::Open3D ${OpenCV_LIBS} fmfusion)

add_executable(GraphMatchNode)
target_sources(GraphMatchNode PRIVATE GraphMatch.cpp)
target_link_libraries(GraphMatchNode PRIVATE Open3D::Open3D ${OpenCV_LIBS} fmfusion)

add_executable(MapMerger)
target_sources(MapMerger PRIVATE MapMerger.cpp)
target_link_libraries(MapMerger PRIVATE Open3D::Open3D ${OpenCV_LIBS} fmfusion)

if (LOOP_DETECTION)
    add_executable(TestLoop)
    target_sources(TestLoop PRIVATE TestLoop.cpp)
    target_link_libraries(TestLoop PRIVATE Open3D::Open3D fmfusion ${TORCH_LIBRARIES})
endif()

# Install fmfusion library
if(INSTALL_FMFUSION)
    message(STATUS "Installing fmfusion library")

    install(TARGETS fmfusion
        # EXPORT fmfusionTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
    )
    install(FILES 
        SceneGraph.h
        Common.h
        DESTINATION include/fmfusion
    )

    # include(CMakePackageConfigHelpers)
    # write_basic_package_version_file(
    #     "${CMAKE_CURRENT_BINARY_DIR}/fmfusion/fmfusionConfigVersion.cmake"
    #     VERSION 1.0
    #     COMPATIBILITY AnyNewerVersion
    # )

    # install(EXPORT fmfusionTargets
    #     FILE fmfusionTargets.cmake
    #     NAMESPACE fmfusion::
    #     DESTINATION lib/cmake/fmfusion
    # )

    # configure_file(cmake/fmfusionConfig.cmake
    #     "${CMAKE_CURRENT_BINARY_DIR}/fmfusion/fmfusionConfig.cmake"
    #     COPYONLY
    # )

endif()
