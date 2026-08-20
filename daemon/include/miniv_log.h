#pragma once
#include <android/log.h>
#include <atomic>
#include <cstdio>
#include <ctime>

namespace miniv::ai {

inline std::atomic<int>& miniv_req_counter() {
  static std::atomic<int> counter{0};
  return counter;
}

// 새 요청 시작 시 1회 호출 (지금은 hal_service.cpp의 inferStream()에서만 사용).
inline int miniv_next_req_id() {
  return miniv_req_counter().fetch_add(1, std::memory_order_relaxed) + 1;
}

inline int& miniv_thread_req_id() {
  static thread_local int id = 0;
  return id;
}
inline void miniv_set_req_id(int id) { miniv_thread_req_id() = id; }

inline void miniv_file_log(const char* level, const char* msg) {
  FILE* f = fopen("/data/vendor/miniv_ai/debug.log", "a");
  if (!f) return;
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  int req = miniv_thread_req_id();
  if (req > 0) {
    fprintf(f, "[%ld.%03ld][req=%d][%s] %s\n", (long)ts.tv_sec,
            ts.tv_nsec / 1000000, req, level, msg);
  } else {
    fprintf(f, "[%ld.%03ld][%s] %s\n", (long)ts.tv_sec,
            ts.tv_nsec / 1000000, level, msg);
  }
  fclose(f);
}

}  // namespace miniv::ai

#ifndef LOG_TAG
#define LOG_TAG "miniv_ai"
#endif

#define LOGI(...)                                                       \
  do {                                                                  \
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__);        \
    char miniv_logbuf[1024];                                            \
    snprintf(miniv_logbuf, sizeof(miniv_logbuf), __VA_ARGS__);          \
    ::miniv::ai::miniv_file_log("I", miniv_logbuf);                     \
  } while (0)

#define LOGE(...)                                                       \
  do {                                                                  \
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__);       \
    char miniv_logbuf[1024];                                            \
    snprintf(miniv_logbuf, sizeof(miniv_logbuf), __VA_ARGS__);          \
    ::miniv::ai::miniv_file_log("E", miniv_logbuf);                     \
  } while (0)