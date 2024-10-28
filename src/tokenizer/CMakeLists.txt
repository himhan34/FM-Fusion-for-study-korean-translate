cmake_minimum_required(VERSION 3.18)

project(tokenizer VERSION 1.0)

find_package(spdlog REQUIRED) # PATHS ${SPDLOG_ROOT})
find_package(glog REQUIRED)

add_library(tokenizer SHARED 
        text_tokenizer.cc
        bert_tokenizer.cc
        )
target_link_libraries(tokenizer PRIVATE spdlog::spdlog ${LIB_UTF8PROC} glog::glog)
set_target_properties(tokenizer PROPERTIES CXX_STANDARD 17)


