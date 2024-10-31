cmake_minimum_required(VERSION 3.18)

project(FmFusion LANGUAGES C CXX)

#### Options ####
option(LOOP_DETECTION OFF)
option(INSTALL_FMFUSION ON) # Install as static library
option(RUN_HYDRA OFF)
#################

set(ALL_TARGET_LIBRARIES "")
include(cmake/eigen.cmake)
include(cmake/glog.cmake)
include(cmake/opencv.cmake)
include(cmake/open3d.cmake)
include(cmake/jsoncpp.cmake)

set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install)
message(STATUS "Project root directory: ${PROJECT_ROOT_DIR}")
message(STATUS "CMake install prefix: ${CMAKE_INSTALL_PREFIX}")
add_subdirectory(src)
