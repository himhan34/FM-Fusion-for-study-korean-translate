cmake_minimum_required(VERSION 3.18)

project(Open3dExtra LANGUAGES C CXX)

add_subdirectory(cpp)
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install)