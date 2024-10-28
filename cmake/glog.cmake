find_package(glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})
list(APPEND ALL_TARGET_LIBRARIES ${GLOG_LIBRARIES})
