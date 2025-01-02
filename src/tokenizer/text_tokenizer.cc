/*
 * File: text_tokenizer.cc
 * Project: utils
 * File Created: Sunday, 20th October 2019 10:33:18 am
 * Author: Koth (yovnchine@163.com)
 * -----
 * Last Modified: Sunday, 20th October 2019 10:34:00 am
 * Modified By: Koth (yovnchine@163.com>)
 * -----
 */

#include "text_tokenizer.h"
#include "logging.h"

namespace radish {

// 텍스트 토크나이저 팩토리에서 사용할 정적 변수 sMethods를 초기화합니다.
// 이는 등록된 토크나이저 생성 함수들을 저장하는 맵입니다.
std::map<std::string, TextTokenizerFactory::TCreateMethod>*
    TextTokenizerFactory::sMethods =
        nullptr;  // 포인터를 사용하여 초기화 문제를 해결합니다.

// Register 함수는 주어진 이름과 생성 함수를 맵에 등록합니다.
bool TextTokenizerFactory::Register(
    const std::string name, TextTokenizerFactory::TCreateMethod funcCreate) {
  
  // sMethods가 nullptr인 경우, 새로운 맵을 할당합니다.
  if (sMethods == nullptr) {
    sMethods = new std::map<std::string, TextTokenizerFactory::TCreateMethod>();
  }
  
  // 주어진 이름에 해당하는 생성 함수가 이미 등록되어 있는지 확인합니다.
  auto it = sMethods->find(name);
  if (it == sMethods->end()) {
    // 이름이 없으면 새로운 생성 함수를 등록합니다.
    sMethods->insert(std::make_pair(name, funcCreate));
    return true;  // 등록 성공
  }
  return false;  // 이미 등록된 이름이 존재하면 등록하지 않습니다.
}

// Create 함수는 주어진 이름에 해당하는 텍스트 토크나이저를 생성합니다.
TextTokenizer* TextTokenizerFactory::Create(const std::string& name) {
  // 주어진 이름에 해당하는 생성 함수가 맵에 존재하는지 확인합니다.
  auto it = sMethods->find(name);
  if (it == sMethods->end()) {
    return nullptr;  // 없으면 nullptr 반환
  }
  // 생성 함수가 있으면 해당 함수로 텍스트 토크나이저를 생성하여 반환합니다.
  return it->second();
}

}  // namespace radish
