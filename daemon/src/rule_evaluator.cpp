#include "rule_evaluator.h"
#include <random>
#include <regex>
#include <android/log.h>

#define LOG_TAG "RuleEvaluator"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace miniv::ai {

int evaluate_response(const std::string &text) {
  if (text.empty()) return -100;

  // 더미 랜덤 평가기: 0 ~ 100 범위의 무작위 난수 점수 부여 (N=2 후보군 무작위 선택 테스트용)
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, 100);

  int random_score = dist(gen);

  /*
  // [참고] 추후 적용할 룰베이스 평가 규칙:
  int rule_score = 0;

  // 룰 1: 길이 보상 (너무 짧은 단답 방지)
  if (text.length() > 20) rule_score += 5;

  // 룰 2: 특정 키워드 패널티 (예: 불필요한 사과 멘트 억제)
  if (text.find("죄송합니다") != std::string::npos) rule_score -= 10;

  // 룰 3: 정규표현식 매칭 (예: 마크다운 코드 블록이 정상적으로 포함되었는지)
  std::regex code_block("```[\\s\\S]*?```");
  if (std::regex_search(text, code_block)) rule_score += 15;
  */

  LOGI("evaluate_response (dummy random): score=%d, text_len=%zu", random_score, text.size());
  return random_score;
}

}  // namespace miniv::ai
