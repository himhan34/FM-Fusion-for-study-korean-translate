/*
 * This is a thirdparty module adopted from: https://github.com/LieluoboAi/radish
 * File: bert_tokenizer.h
 * Project: bert
 * File Created: Saturday, 19th October 2019 10:41:25 am
 * Author: Koth (yovnchine@163.com)
 * -----
 * Last Modified: Saturday, 19th October 2019 10:41:38 am
 * Modified By: Koth (yovnchine@163.com>)
 * -----
 */
#pragma once
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "text_tokenizer.h"

namespace radish {

// UString은 16비트 유니코드 문자로 이루어진 문자열 타입입니다.
using UString = std::basic_string<uint16_t>;

// BertTokenizer 클래스는 TextTokenizer를 상속받아 BERT 모델의 토크나이징 기능을 제공합니다.
class BertTokenizer : public TextTokenizer,
                      TextTokenizerRegisteeStub<BertTokenizer> {
 public:
  // 어휘 파일을 사용하여 초기화하는 함수
  bool Init(std::string vocab) override;

  // 파일 내용으로 초기화하는 함수
  bool InitByFileContent(std::string content);

  // 주어진 텍스트를 인코딩하여 토큰 ID 리스트를 반환하는 함수
  std::vector<int> Encode(std::string text) override;

  // 단어를 ID로 변환하는 함수
  int Word2Id(std::string word) const override;

  // ID를 단어로 변환하는 함수
  std::string Id2Word(int id) const override;

  // Pad 토큰의 ID를 반환하는 함수
  int PadId() const override;

  // Mask 토큰의 ID를 반환하는 함수
  int MaskId() const override;

  // Sep 토큰의 ID를 반환하는 함수
  int SepId() const override;

  // Cls 토큰의 ID를 반환하는 함수
  int ClsId() const override;

  // Unk 토큰의 ID를 반환하는 함수
  int UnkId() const override;

  // 총 토큰 수를 반환하는 함수
  int TotalSize() const override;

 private:
  // 문자열을 최대 길이로 분할하여 결과에 토큰 ID를 추가하는 함수
  void max_seg_(std::string s, std::vector<int>& results);

  // 어휘 파일을 읽어들여 라인별로 저장하는 함수
  void load_vocab_(std::string path, std::vector<std::string>& lines);

  // 라인 목록으로부터 초기화하는 함수
  void init_from_lines(const std::vector<std::string>& lines);

  // 텍스트를 기본적인 방식으로 토큰화하는 함수
  UString _basic_tokenize(UString text);

  // 텍스트를 정리하는 함수
  UString _clean(UString text);

  // 토큰과 그에 대응하는 ID를 저장하는 맵
  std::unordered_map<std::string, int> token_2_id_map_;

  // 토큰 목록을 저장하는 벡터
  std::vector<std::string> tokens_;

  // 다양한 특수 토큰들을 정의합니다.
  static std::string kUnkToken;
  static std::string kMaskToken;
  static std::string kSepToken;
  static std::string kPadToken;
  static std::string kClsToken;
};

}  // namespace radish
