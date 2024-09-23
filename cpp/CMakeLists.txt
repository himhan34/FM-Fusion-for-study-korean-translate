cmake_minimum_required(VERSION 3.18)

project(fmfusion LANGUAGES CXX VERSION 1.0)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_PREFIX_PATH "/home/cliuci/tools/libtorch")
# option(ENABLE_TORCH OFF)
option(LOOP_DETECTION OFF)
option(INSTALL_FMFUSION OFF)
option(RUN_HYDRA OFF)

find_package(jsoncpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Open3D REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GLOG REQUIRED)
find_package(GTest REQUIRED)
add_subdirectory(thirdparty/G3Reg)
add_subdirectory(thirdparty/Kimera-RPGO)
include_directories("thirdparty/Kimera-RPGO/include")

##
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${Open3D_INCLUDE_DIRS})
include_directories(${GTSAM_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${GLOG_INCLUDE_DIRS})

##
# print open3d directories
message(STATUS "Open3D_INCLUDE_DIRS: ${Open3D_INCLUDE_DIRS}")
message(STATUS "Open3D_LIBRARIES: ${Open3D_LIBRARIES}")
message(STATUS "Eigen library: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "CMake install prefix: ${CMAKE_INSTALL_PREFIX}")

set(FMFUSION_HEADER
        mapping/SubVolume.h
        mapping/Detection.h
        mapping/Instance.h
        mapping/SemanticMapping.h
        cluster/PoseGraph.h
        tools/Tools.h
        tools/Utility.h
        tools/IO.h
        tools/g3reg_api.h
        tools/TicToc.h
        Common.h
)

set(FMFUSION_SRC
        mapping/SubVolume.cpp
        mapping/Detection.cpp
        mapping/Instance.cpp
        mapping/SemanticMapping.cpp
        cluster/PoseGraph.cpp
        tools/Visualization.cpp
        tools/Utility.cpp
        tools/IO.cpp
)

set(ALL_TARGET_LIBRARIES
        Open3D::Open3D
        jsoncpp_lib
        ${OpenCV_LIBS}
        gtsam
        G3REG::g3reg
        KimeraRPGO
        ${GLOG_LIBRARIES}
)

message(STATUS "GTSAM VERSION: ${GTSAM_VERSION} ${GTSAM_INCLUDE_DIR} ${gtsam}")

# FMFusion
if (LOOP_DETECTION)
    set(LIB_UTF8PROC "/usr/local/lib/libutf8proc.so")
    add_subdirectory(tokenizer)
    add_subdirectory(thirdparty/extensions)
    find_package(Torch REQUIRED)

    list(APPEND FMFUSION_HEADER
            sgloop/Graph.h
            sgloog/BertBow.h
            sgloop/SGNet.h
            sgloop/ShapeEncoder.h
            sgloop/LoopDetector.h
    )

    list(APPEND FMFUSION_SRC
            tools/IO.cpp
            sgloop/Graph.cpp
            sgloop/BertBow.cpp
            sgloop/SGNet.cpp
            sgloop/ShapeEncoder.cpp
            sgloop/LoopDetector.cpp
    )
    add_library(fmfusion SHARED ${FMFUSION_SRC})

    list(APPEND ALL_TARGET_LIBRARIES
            ${TORCH_LIBRARIES}
            tokenizer
            extensions)

    target_link_libraries(fmfusion PRIVATE ${ALL_TARGET_LIBRARIES})
    target_include_directories(fmfusion PRIVATE ${TORCH_INCLUDE_DIRS} thirdparty/extensions)
    message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
    message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
else ()
    add_library(fmfusion SHARED ${FMFUSION_SRC})
    target_link_libraries(fmfusion PRIVATE ${ALL_TARGET_LIBRARIES})
endif ()

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

# add_executable(GraphMatchNode)
# target_sources(GraphMatchNode PRIVATE GraphMatch.cpp)
# target_link_libraries(GraphMatchNode PRIVATE Open3D::Open3D ${OpenCV_LIBS} fmfusion)

add_executable(MapMerger)
target_sources(MapMerger PRIVATE MapMerger.cpp)
target_link_libraries(MapMerger PRIVATE Open3D::Open3D ${OpenCV_LIBS} fmfusion)

if (LOOP_DETECTION)
    add_executable( "ibraries(TestLoop PRIVATE ${ALL_TARGET_LIBRARIES} fmfusion)

    add_executable(TestRegister)
    target_sources(TestRegister PRIVATE TestRegister.cpp)
    target_link_libraries(TestRegister PRIVATE ${ALL_TARGET_LIBRARIES} fmfusion)

    add_executable(Test3RScanRegister)
    target_sources(Test3RScanRegister PRIVATE Test3RscanRegister.cpp)
    target_link_libraries(Test3RScanRegister PRIVATE ${ALL_TARGET_LIBRARIES} fmfusion)
        
    add_executable(robustPoseAvg)
    target_sources(robustPoseAvg PRIVATE robustPoseAvg.cpp)
    target_link_libraries(robustPoseAvg PRIVATE Open3D::Open3D ${OpenCV_LIBS} fmfusion gtsam -ltbb)

    add_executable(testTwoAgentsPGO)
    target_sources(testTwoAgentsPGO PRIVATE testTwoAgentsPGO.cpp)
    target_link_libraries(testTwoAgentsPGO PRIVATE Open3D::Open3D ${OpenCV_LIBS} fmfusion gtsam KimeraRPGO -ltbb)

endif ()

if (RUN_HYDRA)
    find_package(DBoW2 REQUIRED)
    add_library(DBoW2::DBoW2 INTERFACE IMPORTED)
    set_target_properties(DBoW2::DBoW2 PROPERTIES
                        INTERFACE_LINK_LIBRARIES "${DBoW2_LIBRARIES}"
                        INTERFACE_INCLUDE_DIRECTORIES "${DBoW2_INCLUDE_DIRS}")

    add_executable(TestHydra)
    target_sources(TestHydra PRIVATE TestHydra.cpp)
    target_link_libraries(TestHydra PRIVATE ${ALL_TARGET_LIBRARIES} 
                                        fmfusion 
                                        DBoW2::DBoW2
                                        GTest::gtest_main)
endif()

# Install fmfusion library
if (INSTALL_FMFUSION)
    message(STATUS "Installing fmfusion library. So it can be used in ros warpper.")

    install(TARGETS fmfusion
            # EXPORT fmfusionTargets
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION bin
            INCLUDES DESTINATION include
    )
    install(FILES
            Common.h
            DESTINATION include/fmfusion
    )
    install(FILES
            cluster/PoseGraph.h
            DESTINATION include/fmfusion/cluster
    )
    install(FILES
            mapping/SemanticDict.h
            mapping/SubVolume.h
            mapping/Detection.h
            mapping/Instance.h
            mapping/SemanticMapping.h
            DESTINATION include/fmfusion/mapping
    )
    install(FILES
            sgloop/Graph.h
            sgloop/SGNet.h
            sgloop/ShapeEncoder.h
            sgloop/LoopDetector.h
            sgloop/BertBow.h
            sgloop/Initialization.h
            DESTINATION include/fmfusion/sgloop
    )
    install(FILES
            tools/Tools.h
            tools/Eval.h
            tools/Utility.h
            tools/Color.h
            tools/IO.h
            tools/g3reg_api.h
            tools/TicToc.h
            DESTINATION include/fmfusion/tools
    )
    install(FILES
            tokenizer/logging.h
            tokenizer/basic_string_util.h
            tokenizer/text_tokenizer.h
            tokenizer/bert_tokenizer.h
            DESTINATION include/fmfusion/tokenizer
    )

    install(FILES
            thirdparty/extensions/cpu/torch_helper.h
            thirdparty/extensions/cpu/grid_subsampling.h
            thirdparty/extensions/cpu/radius_neighbors.h
            DESTINATION include/fmfusion/thirdparty/extensions/cpu
    )

    # include(CMakePackageConfigHelpers)
    # write_basic_package_version_file(
    #     "${CMAKE_CURRENT_BINARY_DIR}/fmfusion/fmfusionConfigVersion.cmake"
    #     VERSION 1.0
    #     COMPATIBILITY AnyNewerVersion
    # )

    # install(EXPORT fmfusionTargets
    #     FILE fmfusionTargets.cmake
    #     NAMESPACE fmfusion::s
    #     DESTINATION lib/cmake/fmfusion
    # )

    # configure_file(cmake/fmfusionConfig.cmake
    #     "${CMAKE_CURRENT_BINARY_DIR}/fmfusion/fmfusionConfig.cmake"
    #     COPYONLY
    # )

endif ()
