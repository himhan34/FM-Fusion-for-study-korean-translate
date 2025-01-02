/*
 * File: logging.h
 * Project: utils
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-24 10:06:23
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */
#pragma once
#include "glog/logging.h"  // Google Logging을 위한 헤더 파일
#include "spdlog/spdlog.h"  // spdlog 라이브러리의 헤더 파일
#include "spdlog/fmt/ostr.h"  // spdlog의 fmt 스트림 연산자 헤더 파일

// 아래는 SPDLOG의 트레이스 로그를 출력하는 매크로입니다.
// 이 매크로는 로거가 TRACE 레벨에서 로그를 출력할 수 있도록 합니다.
// #define SPDLOG_TRACE(logger, ...) \
//   if (logger->should_log(spdlog::level::trace)) { \
//     logger->trace("{}::{}()#{}: ", __FILE__, __FUNCTION__, __LINE__, fmt::format(__VA_ARGS__)); \
//   }
