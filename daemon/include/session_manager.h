#pragma once
#include "llama.h"
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

namespace miniv::ai {

struct SessionMeta {
  int id = -1;
  enum class State { HOT, COLD } state = State::COLD;
  llama_context *ctx = nullptr;
  std::vector<llama_token> tokens;
  std::string swapFilePath;
  size_t swapFileBytes = 0;
  int64_t lastUsedMs = 0;
  std::list<int>::iterator lruIt;
};

// TODO(concurrency): 다중 연결 지원 시 이 클래스 public 메서드들을
// std::mutex + std::lock_guard로 감싸면 됨. 지금은 SocketServer가
// 단일 커넥션만 처리하므로 락 불필요.
class SessionManager {
public:
  static constexpr size_t kDefaultHotLimit = 3;
  static constexpr size_t kDefaultColdLimitBytes = 512ull * 1024 * 1024;
  static constexpr size_t kMinKeepTokens = 64;

  SessionManager(llama_model *model, llama_context_params ctxParams,
                 std::string swapDir);

  void setHotLimit(size_t count);
  void setColdLimit(size_t bytes);

  int createSession();
  bool killSession(int id);
  llama_context *activateForInfer(int id);
  void recordTokens(int id, const llama_token *newTokens, size_t n);

private:
  llama_model *mModel;
  llama_context_params mCtxParams;
  std::string mSwapDir;

  std::unordered_map<int, SessionMeta> mSessions;
  int mNextId = 1;

  std::list<int> mHotLru;
  size_t mHotLimit = kDefaultHotLimit;

  std::list<int> mColdLru;
  size_t mColdLimitBytes = kDefaultColdLimitBytes;
  size_t mColdUsedBytes = 0;

  void touchHot(SessionMeta &meta);
  void evictHotIfNeeded(int excludeId);
  void evictColdIfNeeded(int excludeId);
  bool swapOut(SessionMeta &meta);
  void pruneOldestTokens(SessionMeta &meta, size_t nTokensToRemove);
  std::string swapPathFor(int id) const;
};

} // namespace miniv::ai