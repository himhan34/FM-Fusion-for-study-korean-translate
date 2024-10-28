find_package(jsoncpp REQUIRED)
include_directories(${jsoncpp_INCLUDE_DIRS})
list(APPEND ALL_TARGET_LIBRARIES jsoncpp_lib)