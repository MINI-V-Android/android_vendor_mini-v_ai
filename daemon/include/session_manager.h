#pragma once
#include "llama.h"
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

namespace miniv::ai {

// TODO 아래 다 하드코딩임
static constexpr size_t kMinKeepTokens = 64;
static constexpr size_t kDefaultHotLimit = 3;
static constexpr size_t kDefaultColdLimitBytes = 512ull * 1024 * 1024;

struct SessionMeta {
  int id = -1;
  enum class State { HOT, COLD } state = State::COLD;

  llama_context *ctx = nullptr;    // HOT일 때만 non-null
  std::vector<llama_token> tokens; // 전체 토큰 히스토리 (save/load에 필요)
  std::string swapFilePath;        // 한 번이라도 스왑아웃됐으면 non-empty
  size_t swapFileBytes = 0;
  int64_t lastUsedMs = 0;

  std::list<int>::iterator lruIt; // mHotLru 또는 mColdLru 상의 위치
};
// TODO(concurrency): 다중 연결 지원 시 이 클래스 public 메서드들을
// std::mutex + std::lock_guard로 감싸면 됨. 지금은 SocketServer가
// 단일 커넥션만 처리하므로 락 불필요.
class SessionManager {
public:
  SessionManager(llama_model *model, llama_context_params ctxParams,
                 std::string swapDir);

  void setHotLimit(size_t count);  // SET_HOT_LIMIT
  void setColdLimit(size_t bytes); // SET_COLD_LIMIT

  int createSession();
  bool killSession(int id);

  // 실패 시 nullptr (알 수 없는 id, 할당/복원 실패 등)
  llama_context *activateForInfer(int id);

  // infer() 진행 중 생성된 토큰을 세션 히스토리에 반영 (디스크 재저장은 안 함 —
  // 스왑아웃 시점에만 저장)
  void recordTokens(int id, const llama_token *newTokens, size_t n);

private:
  llama_model *mModel;
  llama_context_params mCtxParams;
  std::string mSwapDir;

  std::unordered_map<int, SessionMeta> mSessions;
  int mNextId = 1;

  std::list<int> mHotLru; // front = 가장 오래됨
  size_t mHotLimit = 1;

  std::list<int> mColdLru;
  size_t mColdLimitBytes = 0;
  size_t mColdUsedBytes = 0;

  void touchHot(SessionMeta &meta);
  void evictHotIfNeeded(int excludeId);
  void evictColdIfNeeded(int excludeId);
  void pruneOldestTokens(SessionMeta &meta, size_t nTokensToRemove);
  bool swapOut(SessionMeta &meta); // HOT -> COLD
  std::string swapPathFor(int id) const;
};

} // namespace miniv::ai