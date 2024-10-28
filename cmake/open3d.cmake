find_package(Open3D REQUIRED)
include_directories(${Open3D_INCLUDE_DIRS})
list(APPEND ALL_TARGET_LIBRARIES Open3D::Open3D)
message(STATUS "Open3D_LIBRARIES: ${Open3D_LIBRARIES}")
