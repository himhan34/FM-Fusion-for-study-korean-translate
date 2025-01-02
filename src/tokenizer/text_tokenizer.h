/*
 * File: text_tokenizer.h
 * Project: utils
 * File Created: Sunday, 20th October 2019 9:53:58 am
 * Author: Koth (yovnchine@163.com)
 * -----
 * Last Modified: Sunday, 20th October 2019 10:05:56 am
 * Modified By: Koth (yovnchine@163.com>)
 * -----
 */

#pragma once

#include <cxxabi.h>

#include <map>
#include <string>
#include <vector>

namespace radish {

// TextTokenizer 클래스는 텍스트 토크나이저의 기본 인터페이스를 정의합니다.
class TextTokenizer {
 public:
  TextTokenizer() = default;  // 기본 생성자
  virtual ~TextTokenizer() = default;  // 가상 소멸자

  // 어휘 파일을 초기화하는 가상 함수
  virtual bool Init(std::string vocab) = 0;

  // 텍스트를 인코딩하는 가상 함수
  virtual std::vector<int> Encode(std::string text) = 0;

  // 단어를 ID로 변환하는 가상 함수
  virtual int Word2Id(std::string word) const = 0;

  // ID를 단어로 변환하는 가상 함수
  virtual std::string Id2Word(int id) const = 0;

  // Pad 토큰의 ID를 반환하는 기본 구현
  virtual int PadId() const { return 0; }

  // Mask 토큰의 ID를 반환하는 가상 함수
  virtual int MaskId() const = 0;

  // Sep 토큰의 ID를 반환하는 가상 함수
  virtual int SepId() const = 0;

  // Cls 토큰의 ID를 반환하는 가상 함수
  virtual int ClsId() const = 0;

  // Unk 토큰의 ID를 반환하는 가상 함수
  virtual int UnkId() const = 0;

  // 총 토큰 수를 반환하는 가상 함수
  virtual int TotalSize() const = 0;
};

// TextTokenizerFactory 클래스는 TextTokenizer 객체를 생성하는 팩토리 역할을 합니다.
class TextTokenizerFactory {
 public:
  // TCreateMethod는 TextTokenizer 객체를 생성하는 함수 포인터 타입입니다.
  using TCreateMethod = TextTokenizer* (*)();

 public:
  TextTokenizerFactory() = delete;  // 인스턴스를 생성할 수 없도록 delete 처리

  // 텍스트 토크나이저를 등록하는 함수
  static bool Register(const std::string name, TCreateMethod funcCreate);

  // 등록된 이름에 해당하는 텍스트 토크나이저를 생성하는 함수
  static TextTokenizer* Create(const std::string& name);

 private:
  // 등록된 생성 함수들을 저장하는 맵
  static std::map<std::string, TCreateMethod>* sMethods;
};

// 템플릿 클래스: 텍스트 토크나이저를 등록하는 역할을 하는 클래스
template <typename T>
class TextTokenizerRegisteeStub {
 public:
  // 클래스 이름을 문자열로 반환하는 함수
  static std::string factory_name() {
    int status = 0;
    return std::string(abi::__cxa_demangle(typeid(T).name(), 0, 0, &status));
  }

 protected:
  // 클래스가 등록되었는지를 추적하는 정적 변수
  static bool s_bRegistered;
};

// 템플릿 클래스에서 클래스 등록을 처리하는 부분
template <typename T>
bool TextTokenizerRegisteeStub<T>::s_bRegistered =
    TextTokenizerFactory::Register(TextTokenizerRegisteeStub<T>::factory_name(),
                                   []() {
                                     return dynamic_cast<TextTokenizer*>(new T);
                                   });

}  // namespace radish
