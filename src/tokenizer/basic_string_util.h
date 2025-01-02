/**
 * @ Author: yvon
 * @ Create Time: 2019-12-25 17:27:21
 * @ Modified by: yvon
 * @ Modified time: 2019-12-25 17:54:22
 * @ Description:
 */
#ifndef RADISH_UTILS_BASIC_STRING_UTIL_H_  // 헤더 파일이 중복 포함되지 않도록 방지하는 전처리기 지시문입니다.
#define RADISH_UTILS_BASIC_STRING_UTIL_H_

#include <cassert>  // 디버깅과 테스트를 위한 assert 매크로를 제공합니다.
#include <cstdio>   // C 표준 입출력 함수들을 제공합니다.
#include <cstdlib>  // 일반적인 유틸리티 함수들을 제공합니다.
#include <ctype.h>  // 문자 처리 함수들을 제공합니다.
#include <memory>   // 스마트 포인터를 포함한 메모리 관리 기능을 제공합니다.
#include <sstream>  // 문자열 스트림을 위한 클래스들을 제공합니다.
#include <stdint.h> // 고정 크기 정수 타입을 제공합니다.
#include <string.h> // 문자열 처리 함수들을 제공합니다.
#include <string>   // C++ 표준 문자열 클래스를 제공합니다.
#include <time.h>   // 시간 관련 함수들을 제공합니다.
#include <vector>   // 동적 배열을 위한 표준 벡터 클래스를 제공합니다.
#include <algorithm> // 다양한 알고리즘 함수를 제공합니다.

#define eq1(x, y) (tolower(x) == tolower(y))  // 두 문자가 대소문자 구분 없이 같은지 비교하는 매크로입니다.
#define eq2(x, y) ((x) == (y))               // 두 값이 같은지 비교하는 매크로입니다.
#define my_eq(t, x, y) ((t) ? eq2(x, y) : eq1(x, y)) // 조건에 따라 eq1 또는 eq2를 선택해 비교하는 매크로입니다.

typedef uint16_t UnicodeCharT;               // 16비트 정수를 Unicode 문자 타입으로 정의합니다.
typedef std::basic_string<UnicodeCharT> UnicodeStr; // Unicode 문자를 사용한 문자열 클래스를 정의합니다.

#define unlikely(x) __builtin_expect(!!(x), 0) // 컴파일러에게 특정 조건의 발생 가능성이 낮음을 힌트로 제공하는 매크로입니다.

namespace std {
  // UnicodeStr 타입에 대해 해시 함수를 정의
  template <> struct hash<UnicodeStr> {
    std::size_t operator()(const UnicodeStr &k) const {
      std::hash<std::u16string> hash_fn;  // std::u16string에 대한 해시 함수 정의
      std::u16string u16str(reinterpret_cast<const char16_t *>(k.c_str()),
                            k.size());  // 입력된 UnicodeStr을 u16string으로 변환
      // u16string을 이용하여 해시 값 계산
      return hash_fn(u16str);
    }
  };
} // namespace std

class BasicStringUtil {
public:
  // 문자열을 소문자로 변환하는 함수
  static void ToLower(std::string &data) {
    std::transform(data.begin(), data.end(), data.begin(),
                   [](unsigned char c) { return std::tolower(c); });
  }

  // 두 문자열 사이의 레벤슈타인 거리 계산 함수
  static unsigned int LevenshteinDistance(const char *word1, const char *word2,
                                          bool caseSensitive = false) {
    unsigned int len1 = strlen(word1), len2 = strlen(word2);
    unsigned int *v = reinterpret_cast<unsigned int *>(
        calloc(len2 + 1, sizeof(unsigned int)));  // 거리 계산을 위한 배열 할당
    unsigned int i = 0, j = 0, current = 0, next = 0, cost = 0;

    // 공통 접두어 제거
    while (len1 > 0 && len2 > 0 && my_eq(caseSensitive, word1[0], word2[0]))
      word1++, word2++, len1--, len2--;

    // 문자열 길이가 0인 경우 처리
    if (!len1)
      return len2;
    if (!len2)
      return len1;

    // 첫 번째 행을 초기화
    for (j = 0; j < len2 + 1; j++)
      v[j] = j;

    // 레벤슈타인 거리 계산
    for (i = 0; i < len1; i++) {
      current = i + 1;
      for (j = 0; j < len2; j++) {
        // 문자 교체 비용 계산
        cost = !(my_eq(caseSensitive, word1[i], word2[j]) ||
                 (i && j && my_eq(caseSensitive, word1[i - 1], word2[j]) &&
                  my_eq(caseSensitive, word1[i], word2[j - 1])));
        // 삽입, 삭제, 교체 중 최소 비용 계산
        next = std::min(std::min(v[j + 1] + 1, current + 1), v[j] + cost);
        v[j] = current;
        current = next;
      }
      v[len2] = next;
    }
    free(v);  // 할당된 메모리 해제
    return next;  // 최종 레벤슈타인 거리 반환
  }

  // 문자열 양쪽 공백을 제거하는 함수
  static std::string TrimString(const std::string &s) {
    size_t nn = s.size();
    size_t i = 0;
    std::string ret(s);
    for (; i < nn; i++) {
      // 공백을 건너뛰기
      if ((i + 1) < nn && static_cast<unsigned char>(s[i]) == 0xC2 &&
          static_cast<unsigned char>(s[i + 1]) == 0xA0) {
        i += 1;
      } else if (s[i] != ' ' && s[i] != '\t' && s[i] != '\r' && s[i] != '\n') {
        break;
      }
    }
    if (i) {
      ret = s.substr(i);  // 앞쪽 공백 제거
    }
    nn = ret.size();
    for (i = nn; i > 0; i--) {
      // 뒤쪽 공백 제거
      if (i > 1 && static_cast<unsigned char>(ret[i - 2]) == 0xC2 &&
          static_cast<unsigned char>(ret[i - 1]) == 0xA0) {
        i -= 1;
      } else if (ret[i - 1] != ' ' && ret[i - 1] != '\t' &&
                 ret[i - 1] != '\r' && ret[i - 1] != '\n') {
        break;
      }
    }
    if (i != nn) {
      ret = ret.substr(0, i);  // 뒤쪽 공백 제거 후 반환
    }
    return ret;
  }

  // 버퍼의 내용을 16진수로 출력하는 함수
  static void HexPrint(const char *buf, const size_t len) {
    size_t i;
    if (len == 0)
      return;
    printf("==========Hex Dump[%d]=========\n", (int)len);
    size_t nn = ((len - 1) / 16) + 1;
    for (size_t j = 0; j < nn; j++) {
      for (i = (16 * j); (i < len) && (i < (16 * j + 16)); i++) {
        if (i < len) {
          printf("%02X", buf[i] & 0xFF);  // 16진수로 출력
        } else {
          printf("  ");
        }
        if (i % 2) {
          putchar(' ');  // 2자리마다 공백 삽입
        }
      }
      putchar('\n');
    }
  }

  // 콜론으로 구분된 문자열을 역순으로 분할하는 함수
  static int SplitAsColonBackward(const char *str, int len,
                                   std::vector<std::pair<std::string, std::string>> *pOut) {
    int i = len - 1;
    int start = i;
    std::string fname;
    std::string fval;
    bool gotVal = false;

    // 역순으로 콜론을 찾아 분할
    for (; i >= 0; i--) {
      if (str[i] == ':') {
        if (gotVal) {
          // 값이 있는 경우 공백을 기준으로 분리
          int j = i + 1;
          while (j < len && str[j] == ' ') {
            j++;
          }
          int starti = j;
          bool needContinue = false;
          while (j < len && str[j] != ' ') {
            if (str[j] == ':') {
              std::string newVal(str + starti, j - starti);
              fval = newVal + ":" + fval;
              needContinue = true;
              start = i - 1;
              break;
            }
            j++;
          }
          if (needContinue) {
            i -= 1;
            continue;
          }
          std::string newVal(str + starti, j - starti);
          while (j < len && str[j] == ' ') {
            j++;
          }
          fname = std::string(str + j, start - j + 1);
          pOut->push_back(std::make_pair(fname, fval));
          fval = newVal;
          start = i - 1;
        } else {
          fval = std::string(str + i + 1, start - i);
          start = i - 1;
          gotVal = true;
        }
      }
    }
    if (!gotVal) {
      return 0;
    }
    if (start < 0) {
      return -1;
    }
    fname = std::string(str, start + 1);
    pOut->push_back(std::make_pair(fname, fval));
    return pOut->size();
  }

  // 문자열에서 공백과 제어 문자를 제거하는 함수
  static std::string StripStringASCIIWhole(const std::string &str) {
    size_t nn = str.size();
    while (nn > 0 && (str[nn - 1] == ' ' || str[nn - 1] == '\t' ||
                      str[nn - 1] == '\r' || str[nn - 1] == '\n')) {
      nn -= 1;
    }
    size_t off = 0;
    while (off < nn && (str[off] == ' ' || str[off] == '\t' ||
                        str[off] == '\r' || str[off] == '\n')) {
      off += 1;
    }
    bool seeWhitespace = false;
    std::string ret;
    for (size_t k = off; k < nn; k++) {
      if (str[k] == ' ' || str[k] == '\t' || str[k] == '\r' || str[k] == '\n') {
        if (!seeWhitespace) {
          seeWhitespace = true;
          ret.append(1, ' ');
        }
      } else {
        seeWhitespace = false;
        ret.append(1, str[k]);
      }
    }
    return ret;
  }

  // 왼쪽 공백을 제거하는 함수
  static std::string StripStringASCIINoSpaceLeft(const std::string &str) {
    size_t nn = str.size();
    while (nn > 0 && (str[nn - 1] == ' ' || str[nn - 1] == '\t' ||
                      str[nn - 1] == '\r' || str[nn - 1] == '\n')) {
      nn -= 1;
    }
    size_t off = 0;
    while (off < nn && (str[off] == ' ' || str[off] == '\t' ||
                        str[off] == '\r' || str[off] == '\n')) {
      off += 1;
    }
    std::string ret;
    for (size_t k = off; k < nn; k++) {
      if (str[k] == ' ' || str[k] == '\t' || str[k] == '\r' || str[k] == '\n') {
      } else {
        ret.append(1, str[k]);
      }
    }
    return ret;
  }

  // 문자열에서 공백을 제거하는 함수
  static std::string StripStringASCII(const std::string &str) {
    size_t nn = str.size();
    while (nn > 0 && (str[nn - 1] == ' ' || str[nn - 1] == '\t' ||
                      str[nn - 1] == '\r' || str[nn - 1] == '\n')) {
      nn -= 1;
    }
    size_t off = 0;
    while (off < nn && (str[off] == ' ' || str[off] == '\t' ||
                        str[off] == '\r' || str[off] == '\n')) {
      off += 1;
    }
    return std::string(str.c_str() + off, nn - off);
  }

  // 주어진 날짜 문자열을 time_t로 변환하는 함수
  static time_t StringToTime(const char *strTime, size_t len) {
    if (NULL == strTime) {
      return 0;
    }
    tm tm_;
    int year, month, day;
    sscanf(strTime, "%d-%d-%d", &year, &month, &day);
    tm_.tm_year = year - 1900;
    tm_.tm_mon = month - 1;
    tm_.tm_mday = day;
    tm_.tm_hour = 0;
    tm_.tm_min = 0;
    tm_.tm_sec = 0;
    tm_.tm_isdst = 0;
    time_t t_ = mktime(&tm_);  // 시간 변환
    return t_;                // 변환된 시간 반환
  }

  // 문자열 앞뒤 공백을 제거하는 함수
  static bool TrimSpace(const std::string &src, std::string *dest) {
    size_t firstNonSpace = 0;
    size_t len = src.size();
    size_t lastNonSpace = len;

    // 앞의 공백 제거
    while (firstNonSpace < len &&
           ((src[firstNonSpace] == ' ') || (src[firstNonSpace] == '\t') ||
            (src[firstNonSpace] == '\n') || (src[firstNonSpace] == '\r'))) {
      firstNonSpace += 1;
    }
    // 뒤의 공백 제거
    while (lastNonSpace > 0 &&
           ((src[lastNonSpace - 1] == ' ') || (src[lastNonSpace - 1] == '\t') ||
            (src[lastNonSpace - 1] == '\n') ||
            (src[lastNonSpace - 1] == '\r'))) {
      lastNonSpace -= 1;
    }
    if (firstNonSpace > lastNonSpace) {
      dest->clear();
      return true;
    }
    if (firstNonSpace != 0 || lastNonSpace != len) {
      dest->assign(src.c_str() + firstNonSpace, lastNonSpace - firstNonSpace);
      return true;
    }
    dest->assign(src);
    return false;
  }

  // 유니코드 문자열을 UTF-16으로 변환하는 함수
  static bool u8tou16(const char *src, size_t len, UnicodeStr &dest) {
    if (src == NULL)
      return true;
    UnicodeCharT stackBuf[1024] = {0};
    UnicodeCharT *ptr = stackBuf;
    size_t out_len = len;
    UnicodeCharT *destBuf = NULL;
    if (out_len > 1024) {
      destBuf = new UnicodeCharT[out_len];  // 충분한 크기 확보
    } else {
      destBuf = stackBuf;
    }
    if (destBuf == NULL)
      return false;
    size_t j = 0;
    unsigned char ubuf[2] = {0};
    for (size_t i = 0; i < len && j < out_len;) {
      unsigned char ch = (src[i] & 0xFF);
      if (ch < (unsigned short)0x80) {
        destBuf[j++] = (ch & 0x7F);  // 1바이트 유니코드 처리
        i += 1;
      } else if (ch < (unsigned short)0xC0) {
        destBuf[j++] = 0x3f;  // 잘못된 유니코드 시 '?'로 대체
        i += 1;
      } else if (ch < (unsigned short)0xE0 && i + 1 < len) {
        ubuf[1] = (((ch & 0x1C) >> 2) & 0x7);
        ubuf[0] = ((((ch & 0x3) << 6)) | ((src[i + 1]) & 0x3F)) & 0xFF;
        ptr = static_cast<UnicodeCharT *>(static_cast<void *>(&ubuf[0]));
        destBuf[j++] = *(ptr);  // 2바이트 유니코드 처리
        i += 2;
      } else if (ch < (unsigned short)0xF0 && i + 2 < len) {
        ubuf[1] = ((((ch & 0x0F) << 4) | ((src[i + 1] & 0x3C) >> 2)) & 0xFF);
        ubuf[0] = ((((src[i + 1] & 0x3) << 6)) | ((src[i + 2]) & 0x3F)) & 0xFF;
        ptr = static_cast<UnicodeCharT *>(static_cast<void *>(&ubuf[0]));
        destBuf[j++] = *ptr;  // 3바이트 유니코드 처리
        i += 3;
      } else {
        destBuf[j++] = 0x3f;  // 잘못된 유니코드 시 '?'로 대체
        i += 4;
      }
    }

    dest.assign(destBuf, j);  // 변환된 문자열 할당
    if (destBuf != stackBuf)
      delete[] destBuf;  // 동적 메모리 해제
    return (j > 0);  // 변환된 문자열 길이 반환
  }
  // 유니코드 문자 배열을 UTF-8 문자열로 변환하는 함수
  static bool u16tou8(const UnicodeCharT *src, size_t len, std::string &dest) {
    if (src == NULL)
      return true;  // 입력이 NULL일 경우 true 반환
    char stackBuf[1024] = {0};  // 스택 버퍼 초기화
    size_t out_len = len * 3;  // UTF-8은 최대 3바이트로 표현 가능
    char *destBuf = NULL;
    // 출력 버퍼 크기 결정
    if (out_len > 1024) {
      destBuf = new char[out_len];  // 필요시 동적 메모리 할당
    } else {
      destBuf = stackBuf;  // 스택 버퍼 사용
    }
    if (destBuf == NULL)
      return false;  // 메모리 할당 실패 시 false 반환
    size_t j = 0;
    for (size_t i = 0; i < len && j < out_len; i++) {
      unsigned short uch = src[i];  // 유니코드 문자 읽기
      if (uch < (unsigned short)0x7F) {
        destBuf[j++] = (uch & 0x007F);  // 1바이트 ASCII 문자 처리
      } else if (uch < (unsigned short)0x7FF) {
        // 2바이트 유니코드 문자 처리
        destBuf[j++] = ((((uch & 0x03C0) >> 6) & 0xFF) | (0xC0)) & 0xFF;
        destBuf[j++] = ((uch & 0x3F) | (0x80)) & 0xFF;
      } else {
        // 3바이트 유니코드 문자 처리
        destBuf[j++] = ((((uch & 0xF000) >> 12) & 0xFF) | (0xE0)) & 0xFF;
        destBuf[j++] = ((((uch & 0x0FC0) >> 6) & 0xFF) | (0x80)) & 0xFF;
        destBuf[j++] = ((uch & 0x3F) | (0x80)) & 0xFF;
      }
    }
    dest.assign(destBuf, j);  // 변환된 UTF-8 문자열 저장
    if (destBuf != stackBuf)
      delete[] destBuf;  // 동적 메모리 해제
    return (j > 0);  // 변환된 바이트 수가 0 이상일 경우 true 반환
  }

  // 문자열을 주어진 구분자로 분할하는 함수
  static int SplitString(const char *str, size_t len, char sepChar,
                         std::vector<std::string> *pOut) {
    const char *ptr = str;
    if (ptr == NULL || len == 0) {
      return 0;  // 입력 문자열이 NULL이거나 길이가 0일 경우 0 반환
    }
    size_t start = 0;
    // 앞의 구분자 공백을 건너뜀
    while (start < len && (str[start] == sepChar)) {
      start += 1;
    }
    ptr = str + start;
    len = len - start;
    // 뒤의 구분자 공백을 건너뜀
    while (len > 0 && ptr[len - 1] == sepChar) {
      len -= 1;
    }
    if (len <= 0) {
      return 0;  // 유효한 문자열이 없으면 0 반환
    }
    size_t ps = 0;
    int nret = 0;
    // 구분자 기준으로 문자열을 분할
    for (size_t i = 0; i < len; i++) {
      if (ptr[i] == sepChar) {
        if (ptr[i - 1] != sepChar) {
          std::string ts(ptr + ps, i - ps);  // 분할된 문자열 저장
          pOut->push_back(ts);
          nret += 1;
        }
        ps = i + 1;
      }
    }
    // 마지막 부분 처리
    if (ps < len) {
      pOut->push_back(std::string(ptr + ps, len - ps));
      nret += 1;
    }
    return nret;  // 분할된 개수 반환
  }

  // 파일 내용을 읽어 반환하는 함수
  static std::string ReadFileContent(const std::string &path) {
    FILE *fp = fopen(path.c_str(), "r");  // 파일 열기
    if (fp == nullptr) {
      fprintf(stderr, "read file [%s] error!\n", path.c_str());
      return std::string();  // 파일 열기 실패 시 빈 문자열 반환
    }
    char buffer[4096] = {0};
    int nn = 0;
    std::string ret;
    // 파일 내용을 버퍼에 읽고 문자열로 저장
    do {
      nn = fread(buffer, sizeof(char), sizeof(buffer) / sizeof(char), fp);
      if (nn > 0) {
        ret.append(buffer, nn);
      }
    } while (nn > 0);
    fclose(fp);  // 파일 닫기
    return ret;  // 파일 내용 반환
  }

  // 문자에 대한 바이트 길이를 반환하는 함수
  static inline int CharByteLen(unsigned char ch) {
    if (unlikely((ch & 0xFC) == 0xFC))
      return 6;  // 4바이트 유니코드 처리
    else if (unlikely((ch & 0xF8) == 0xF8))
      return 5;  // 5바이트 유니코드 처리
    else if (unlikely((ch & 0xF0) == 0xF0))
      return 4;  // 4바이트 유니코드 처리
    else if ((ch & 0xE0) == 0xE0)
      return 3;  // 3바이트 유니코드 처리
    else if (unlikely((ch & 0xC0) == 0xC0))
      return 2;  // 2바이트 유니코드 처리
    else if (unlikely(ch == 0))
      return 1;  // 1바이트 유니코드 처리
    return 1;  // 기본 1바이트 처리
  }

}; // class

namespace utils {
  // 숫자를 문자열로 변환하는 템플릿 함수
  template <typename T> std::string NumberToString(T Number) {
    std::ostringstream ss;
    ss << Number;  // 숫자를 스트림에 삽입
    return ss.str();  // 변환된 문자열 반환
  }

} // namespace utils
#endif // RADISH_UTILS_BASIC_STRING_UTIL_H_
