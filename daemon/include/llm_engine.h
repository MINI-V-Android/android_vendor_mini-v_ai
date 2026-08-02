#pragma once#include "llama.h"
#include "session_manager.h"
#include <atomic>
#include <functional>
#include <memory>
#include <string>

namespace miniv::ai {

class LLMEngine {
public:
  using TokenCallback = std::function<void(const std::string &tokenText)>;

  ~LLMEngine();

  bool load(const std::string &modelPath, int nCtx, int nThreads);
  bool isReady() const { return mModel != nullptr; }

  // cancelFlag: HAL 레이어(hal_service.cpp)가 세션별로 소유/관리하는 취소
  // 플래그의 포인터. UDS 경로(main.cpp)는 아직 취소 명령이 없어 nullptr로
  // 호출하며, 이 경우 취소 체크를 건너뛴다.
  bool infer(int sessionId, const std::string &prompt, int maxTokens,
             TokenCallback onToken, std::atomic<bool> *cancelFlag = nullptr);

  std::string getModelInfo() const;

  SessionManager *sessionManager() { return mSessionManager.get(); }

private:
  llama_model *mModel = nullptr;
  llama_context_params mCtxParams{};
  std::unique_ptr<SessionManager> mSessionManager;
  std::string mModelPath;
  llama_sampler *mSampler = nullptr; // 신규 — top_k/top_p/temp/dist 체인
};

} // namespace miniv::ai