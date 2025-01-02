/*
 * File: bert_tokenizer.cc
 * Project: bert
 * File Created: Saturday, 19th October 2019 11:26:14 am
 * Author: Koth (yovnchine@163.com)
 * -----
 * Last Modified: Saturday, 19th October 2019 11:26:17 am
 * Modified By: Koth (yovnchine@163.com>)
 * -----
 */

#include "bert_tokenizer.h"
#include <cwctype>
#include <fstream>

#include "basic_string_util.h"
#include "logging.h"
#include "source/utf8.h"
#include "utf8proc.h"

namespace radish {

// BERT 토크나이저에서 사용하는 여러 토큰들을 정의합니다.
std::string BertTokenizer::kUnkToken = "[UNK]";
std::string BertTokenizer::kMaskToken = "[MASK]";
std::string BertTokenizer::kSepToken = "[SEP]";
std::string BertTokenizer::kPadToken = "[PAD]";
std::string BertTokenizer::kClsToken = "[CLS]";

// 중국어 구두점을 정의한 집합입니다.
static std::unordered_set<uint16_t> kChinesePunts = {
    12290, 65306, 65311, 8212, 8216, 12304, 12305, 12298, 12299, 65307};

// 단어 당 최대 문자 수를 설정합니다.
static int kMaxCharsPerWords = 100;

// 초기화 함수: 주어진 어휘 파일을 읽어서 토크나이저를 초기화합니다.
bool BertTokenizer::Init(std::string vocab_file) {
  // 어휘 파일을 열고 읽습니다.
  std::ifstream ifs(vocab_file);
  if (!ifs) {  // 파일 열기에 실패하면 false 반환
    return false;
  }
  // 파일의 내용을 읽어서 content에 저장합니다.
  std::string content((std::istreambuf_iterator<char>(ifs)),
                      (std::istreambuf_iterator<char>()));
  // 파일 내용으로부터 초기화합니다.
  return InitByFileContent(content);
}

// 파일 내용으로부터 초기화하는 함수
bool BertTokenizer::InitByFileContent(std::string content) {

  std::vector<std::string> lines;
  // 파일 내용을 줄 단위로 분리합니다.
  BasicStringUtil::SplitString(content.c_str(), content.size(),'\n',&lines);
  
  // 분리된 줄을 바탕으로 초기화합니다.
  init_from_lines(lines);

  // 각 토큰이 어휘 사전에 존재하는지 확인합니다.
  if (token_2_id_map_.find(kPadToken) == token_2_id_map_.end()) {
    return false;  // PadToken이 없으면 초기화 실패
  }
  if (token_2_id_map_.find(kUnkToken) == token_2_id_map_.end()) {
    return false;  // UnkToken이 없으면 초기화 실패
  }
  if (token_2_id_map_.find(kClsToken) == token_2_id_map_.end()) {
    return false;  // ClsToken이 없으면 초기화 실패
  }
  if (token_2_id_map_.find(kSepToken) == token_2_id_map_.end()) {
    return false;  // SepToken이 없으면 초기화 실패
  }
  if (token_2_id_map_.find(kMaskToken) == token_2_id_map_.end()) {
    return false;  // MaskToken이 없으면 초기화 실패
  }

  // PadToken의 id 값이 0인지 확인합니다.
  int v = token_2_id_map_.at(kPadToken);
  if (v != 0) {
    return false;  // PadToken의 id가 0이 아니면 초기화 실패
  }
  return true;  // 모든 조건을 통과하면 초기화 성공
}

std::vector<int> BertTokenizer::Encode(std::string text) {
  (void)s_bRegistered;  // 등록을 강제로 수행하기 위한 코드

  std::vector<int> results;

  // 텍스트에서 ASCII 문자를 제외한 공백을 제거합니다.
  text = BasicStringUtil::StripStringASCIIWhole(text);

  // 텍스트를 NFD 형태로 분해하여 nfkcstr에 저장합니다.
  char* nfkcstr = reinterpret_cast<char*>(
      utf8proc_NFD(reinterpret_cast<const unsigned char*>(text.c_str())));

  // NFD 변환에 실패한 경우 에러 로그를 출력하고 빈 벡터를 반환합니다.
  if (nfkcstr == nullptr) {
    spdlog::info("do NFD error");
    return {};
  }

  // 변환된 문자열을 다시 텍스트에 할당하고 메모리를 해제합니다.
  text.assign(nfkcstr, strlen(nfkcstr));
  free(nfkcstr);

  // 텍스트를 소문자로 변환합니다.
  BasicStringUtil::ToLower(text);

  // UTF-8 텍스트를 UTF-16 문자열로 변환합니다.
  UString unicodes;
  utf8::utf8to16(text.c_str(), text.c_str() + text.size(),
                 std::back_inserter(unicodes));

  // 텍스트를 정리합니다.
  unicodes = _clean(unicodes);

  // 기본적인 토큰화 작업을 수행합니다.
  unicodes = _basic_tokenize(unicodes);

  std::string newtext;
  // UTF-16 문자열을 다시 UTF-8로 변환합니다.
  utf8::utf16to8(
      reinterpret_cast<const uint16_t*>(unicodes.c_str()),
      reinterpret_cast<const uint16_t*>(unicodes.c_str() + unicodes.size()),
      std::back_inserter(newtext));

  // 변환된 텍스트를 공백을 기준으로 분할하여 토큰 리스트를 생성합니다.
  std::vector<std::string> tokens;
  BasicStringUtil::SplitString(newtext.c_str(), newtext.size(), ' ', &tokens);

  // 각 토큰에 대해 처리하여 results 벡터에 토큰 ID를 추가합니다.
  for (auto s : tokens) {
    // 토큰의 길이가 최대 길이를 초과하면 UNK 토큰을 추가합니다.
    if (s.size() > kMaxCharsPerWords) {
      results.push_back(token_2_id_map_.at(kUnkToken));
    } else {
      // 그렇지 않으면 max_seg_ 함수를 통해 세그먼트를 나누어 ID를 추가합니다.
      max_seg_(s, results);
    }
  }

  // 결과로 생성된 토큰 ID 리스트를 반환합니다.
  return results;
}

int BertTokenizer::PadId() const { return token_2_id_map_.at(kPadToken); }  // Pad 토큰의 ID를 반환
int BertTokenizer::MaskId() const { return token_2_id_map_.at(kMaskToken); }  // Mask 토큰의 ID를 반환
int BertTokenizer::SepId() const { return token_2_id_map_.at(kSepToken); }  // Sep 토큰의 ID를 반환
int BertTokenizer::ClsId() const { return token_2_id_map_.at(kClsToken); }  // Cls 토큰의 ID를 반환
int BertTokenizer::UnkId() const { return token_2_id_map_.at(kUnkToken); }  // Unk 토큰의 ID를 반환

int BertTokenizer::TotalSize() const { return tokens_.size(); }  // 총 토큰 수를 반환

// 최대 길이로 분할하여 결과에 토큰 ID를 추가하는 함수
void BertTokenizer::max_seg_(std::string s, std::vector<int>& results) {
  int end = s.size();
  int start = 0;
  bool firstOne = true;
  // 시작 위치에서 끝까지 토큰을 분할하여 처리합니다.
  while (start < end) {
    std::string test(s.c_str() + start, end - start);
    if (!firstOne) {
      test = std::string("##") + test;  // 첫 번째 이후에는 '##'를 추가하여 서브 토큰으로 처리
    }
    auto it = token_2_id_map_.find(test);
    if (it == token_2_id_map_.end()) {
      end -= 1;  // 매칭되지 않으면 끝 위치를 한 칸 앞으로 이동
    } else {
      results.push_back(it->second);  // 매칭된 토큰의 ID를 결과에 추가
      start = end;  // 분할을 종료
      end = s.size();  // 끝 위치를 원래 텍스트의 끝으로 설정
      firstOne = false;
    }
  }
  // 첫 번째 토큰이 매칭되지 않으면 UNK 토큰을 추가합니다.
  if (firstOne) {
    results.push_back(token_2_id_map_.at(kUnkToken));
  }
}

// 단어를 ID로 변환하는 함수
int BertTokenizer::Word2Id(std::string s) const {
  if (s.size() > kMaxCharsPerWords) {  // 단어의 길이가 최대 길이를 초과하면 UNK 토큰을 반환
    return token_2_id_map_.at(kUnkToken);
  }
  auto it = token_2_id_map_.find(s);
  if (it == token_2_id_map_.end()) {  // 토큰이 없으면 UNK 토큰 반환
    return token_2_id_map_.at(kUnkToken);
  } else {  // 토큰이 존재하면 해당 ID 반환
    return it->second;
  }
}

// ID를 단어로 변환하는 함수
std::string BertTokenizer::Id2Word(int id) const {
  if (id >= 0 && id < static_cast<int>(tokens_.size())) {  // 유효한 ID 범위일 때
    return tokens_[id];
  }
  return kUnkToken;  // 유효하지 않으면 UNK 토큰 반환
}

// 어휘 목록에서 토큰을 초기화하는 함수
void BertTokenizer::init_from_lines(const std::vector<std::string>& lines) {
  int idx = 0;
  for (size_t i = 0; i < lines.size(); i++) {
    std::string line = lines[i];
    size_t nn = line.size();
    while (nn > 0 && (line[nn - 1] == '\n' || line[nn - 1] == '\r')) {
      nn -= 1;  // 줄 끝의 공백 문자 제거
    }
    if (nn == 0) {
      continue;  // 비어 있는 줄은 건너뜁니다.
    }
    std::string token = line.substr(0, nn);
    tokens_.push_back(token);  // 토큰 리스트에 추가
    token_2_id_map_[token] = idx;  // 토큰과 ID를 맵에 추가
    idx += 1;
  }
}

// 어휘 파일을 읽어들여 줄 단위로 저장하는 함수
void BertTokenizer::load_vocab_(std::string path,
                                std::vector<std::string>& lines) {
  FILE* fp = fopen(path.c_str(), "r");
  CHECK(fp != NULL) << "open file error:" << path;  // 파일 열기 오류 체크
  char line[4096] = {0};
  int idx = 0;
  while (fgets(line, sizeof(line) - 1, fp)) {
    int nn = strlen(line);
    while (nn && (line[nn - 1] == '\n' || line[nn - 1] == '\r')) {
      nn -= 1;  // 줄 끝의 공백 문자 제거
    }
    if (nn <= 0) {
      continue;  // 비어 있는 줄은 건너뜁니다.
    }
    lines.push_back(std::string(line, nn));  // 줄을 저장
  }
  fclose(fp);  // 파일 닫기
}

// 공백 문자인지 확인하는 함수
static bool _is_whitespace(uint16_t c) {
  if (c == '\t' || c == '\n' || c == '\r' || c == ' ') {
    return true;
  }
  return (UTF8PROC_CATEGORY_ZS == utf8proc_category(c));
}

// 제어 문자인지 확인하는 함수
static bool _is_control(uint16_t c) {
  if (c == '\t' || c == '\n' || c == '\r') {
    return false;
  }
  utf8proc_category_t cat = utf8proc_category(c);
  return (cat == UTF8PROC_CATEGORY_CC || cat == UTF8PROC_CATEGORY_CF);
}

// 중국어 문자인지 확인하는 함수
static bool _is_chinese_char(uint16_t cp) {
  if ((cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
      (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
      (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
      (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F)) {
    return true;  // 중국어 범위에 해당하는 문자
  }
  return false;
}

// 문자가 구두점인지 확인하는 함수
static bool _is_punct_char(uint16_t cp) {
  if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
      (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) {
    return true;  // ASCII 구두점
  }
  if (cp == ' ') {
    return false;  // 공백은 구두점이 아님
  }
  // 중국어 구두점 확인
  if (kChinesePunts.find(cp) != kChinesePunts.end()) {
    return true;
  }
  int cate = static_cast<int>(utf8proc_category(cp));
  return (cate >= 12 && cate <= 18);  // 기타 구두점 범주
}

// 기본적인 토큰화를 수행하는 함수
UString BertTokenizer::_basic_tokenize(UString text) {
  UString ret;
  size_t len = text.size();
  for (size_t i = 0; i < len; i++) {
    uint16_t c = text[i];
    // 중국어 문자나 구두점일 경우 공백으로 구분하여 처리
    if (_is_chinese_char(c) || _is_punct_char(c)) {
      if (!ret.empty() && ret.back() != ' ') {
        ret.append(1, ' ');  // 이전에 공백이 없다면 공백 추가
      }
      ret.append(1, c);  // 문자 추가
      ret.append(1, ' ');  // 공백 추가
    } else if (c == ' ') {
      if (!ret.empty() && ret.back() != ' ') {
        ret.append(1, c);  // 공백이 연속되지 않도록 처리
      }
    } else {
      ret.append(1, c);  // 공백이 아니면 문자 그대로 추가
    }
  }
  if (!ret.empty() && ret.back() == ' ') {
    ret.erase(ret.end() - 1);  // 끝에 공백이 있다면 제거
  }
  return ret;
}

// 텍스트를 정리하는 함수
UString BertTokenizer::_clean(UString text) {
  size_t len = text.size();
  UString ret;
  for (size_t i = 0; i < len; i++) {
    uint16_t c = text[i];
    // 0, 유효하지 않은 문자, 제어 문자, 마크 문자는 제거
    if (c == 0 || c == 0xFFFD || _is_control(c) ||
        utf8proc_category(c) == UTF8PROC_CATEGORY_MN) {
      continue;
    }
    // 공백 문자는 하나의 공백으로 치환
    if (_is_whitespace(c)) {
      ret.append(1, ' ');
    } else {
      ret.append(1, c);  // 그 외의 문자는 그대로 추가
    }
  }
  return ret;
}

}  // namespace radish
 
