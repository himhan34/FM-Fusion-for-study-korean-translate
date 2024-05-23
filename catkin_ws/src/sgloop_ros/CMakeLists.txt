cmake_minimum_required(VERSION 3.0.2)
project(sgloop_ros)

# add_compile_options(-std=c++11)
set(CMAKE_CXX_STANDARD 17)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
  open3d_conversions
)

set(FMFUSION_LIBRARIES "/home/cliuci/code_ws/OpensetFusion/build/cpp/libfmfusion.so")
set(FMFUSION_INCLUDE_DIR "/home/cliuci/code_ws/OpensetFusion/install/include/fmfusion")
set(CMAKE_PREFIX_PATH "/home/cliuci/tools/libtorch")
find_package(OpenCV REQUIRED)
find_package(Open3D REQUIRED)
find_package(Torch REQUIRED)

# find_library(FMFUSION_LIBRARIES fmfusion PATHS "/home/cliuci/code_ws/OpensetFusion/build/cpp")

# Append a library to torch
# list(APPEND TORCH_LIBRARIES "/usr/local/cuda/lib64/libnvToolsExt.so")

message(STATUS "FMFUSION_LIBRARIES: ${FMFUSION_LIBRARIES} ${FMFUSION_INCLUDE_DIR}")
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

catkin_package(
 INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs open3d_conversions
  DEPENDS Open3D
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${FMFUSION_INCLUDE_DIR}
)

add_library(sgloop_ros SHARED
          src/Visualization.cpp)
target_link_libraries(sgloop_ros ${catkin_LIBRARIES} Open3D::Open3D ${TORCH_LIBRARIES} ${FMFUSION_LIBRARIES} ${OpenCV_LIBS})

add_executable(OfflineLoop src/OfflineLoopNode.cpp)
target_link_libraries(OfflineLoop ${catkin_LIBRARIES} Open3D::Open3D ${TORCH_LIBRARIES} ${FMFUSION_LIBRARIES} ${OpenCV_LIBS} sgloop_ros)


## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/sgloop_ros_node.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_sgloop_ros.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
