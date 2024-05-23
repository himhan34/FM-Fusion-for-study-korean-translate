cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(extensions LANGUAGES CXX)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# add_executable(example-app example-app.cpp)
# target_link_libraries(example-app "${TORCH_LIBRARIES}")
# set_property(TARGET example-app PROPERTY CXX_STANDARD 17)

message(STATES "compiling extensions")

add_library(extensions SHARED 
    extra/cloud/cloud.cpp
    extra/nanoflann/nanoflann.hpp
    cpu/radius_neighbors/radius_neighbors_cpu.cpp
    cpu/grid_subsampling/grid_subsampling_cpu.cpp    
    cpu/radius_neighbors/radius_neighbors.cpp
    cpu/grid_subsampling/grid_subsampling.cpp    
    # cpu/radius_neighbors/radius_neighbors.h
    # common/torch_helper.h
)

target_link_libraries(extensions PRIVATE ${TORCH_LIBRARIES})
target_include_directories(extensions PRIVATE ${TORCH_INCLUDE_DIRS})
