cmake_minimum_required(VERSION 3.18)

project(fmfusion LANGUAGES CXX VERSION 1.0)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Dependencies
add_definitions(-D _GLIBCXX_USE_CXX11_ABI=1)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# FMFusion::Mapping
set(FMFUSION_HEADER
        mapping/SubVolume.h
        mapping/Detection.h
        mapping/Instance.h
        mapping/SemanticMapping.h
        cluster/PoseGraph.h
        tools/Tools.h
        tools/Utility.h
        tools/IO.h
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

# FMFusion::LoopDetection
if (LOOP_DETECTION)
    include(${PROJECT_ROOT_DIR}/cmake/gtsam.cmake)
    include_directories("thirdparty/Kimera-RPGO/include")
    include_directories("thirdparty/extensions")
    set(CMAKE_PREFIX_PATH "/home/cliuci/tools/libtorch")
    set(LIB_UTF8PROC "/usr/local/lib/libutf8proc.so")

    add_subdirectory(tokenizer)
    add_subdirectory(thirdparty/extensions)
    add_subdirectory(thirdparty/G3Reg)
    add_subdirectory(thirdparty/Kimera-RPGO)
    find_package(Torch REQUIRED)

    list(APPEND FMFUSION_HEADER
            tools/g3reg_api.h
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

    list(APPEND ALL_TARGET_LIBRARIES
            ${TORCH_LIBRARIES}
            G3REG::g3reg
            KimeraRPGO
            tokenizer
            extensions
        )
    message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
    message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
endif()

add_library(fmfusion SHARED ${FMFUSION_SRC})
target_link_libraries(fmfusion PRIVATE ${ALL_TARGET_LIBRARIES})

# Executables
add_executable(IntegrateRGBD)
target_sources(IntegrateRGBD PRIVATE IntegrateRGBD.cpp)
target_link_libraries(IntegrateRGBD PRIVATE ${ALL_TARGET_LIBRARIES} fmfusion)

add_executable(IntegrateInstanceMap)
target_sources(IntegrateInstanceMap PRIVATE IntegrateInstanceMap.cpp)
target_link_libraries(IntegrateInstanceMap PRIVATE ${ALL_TARGET_LIBRARIES} fmfusion)

# Install loop closure detection(LCD) module
# The LCD is in developing. More complete instruction will be released soon.
if (LOOP_DETECTION)
    add_executable(TestLoop)
    target_sources(TestLoop PRIVATE TestLoop.cpp)
    target_link_libraries(TestLoop PRIVATE ${ALL_TARGET_LIBRARIES} fmfusion)

    add_executable(TestRegister)
    target_sources(TestRegister PRIVATE TestRegister.cpp)
    target_link_libraries(TestRegister PRIVATE ${ALL_TARGET_LIBRARIES} fmfusion)

    add_executable(Test3RScanRegister)
    target_sources(Test3RScanRegister PRIVATE Test3RscanRegister.cpp)
    target_link_libraries(Test3RScanRegister PRIVATE ${ALL_TARGET_LIBRARIES} fmfusion)
        
    add_executable(robustPoseAvg)
    target_sources(robustPoseAvg PRIVATE robustPoseAvg.cpp)
    target_link_libraries(robustPoseAvg PRIVATE ${ALL_TARGET_LIBRARIES} fmfusion -ltbb)

    add_executable(testTwoAgentsPGO)
    target_sources(testTwoAgentsPGO PRIVATE testTwoAgentsPGO.cpp)
    target_link_libraries(testTwoAgentsPGO PRIVATE ${ALL_TARGET_LIBRARIES} fmfusion -ltbb)

endif ()

if (RUN_HYDRA)
    find_package(GTest REQUIRED)
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
            mapping/BayesianLabel.h
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
            tools/TicToc.h
            DESTINATION include/fmfusion/tools
    )

    if(LOOP_DETECTION)
        install(FILES
                tools/g3reg_api.h
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
     endif()
endif ()
