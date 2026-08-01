#pragma once
#include "llama.h"
#include "session_manager.h"
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

  bool infer(int sessionId, const std::string &prompt, int maxTokens,
             TokenCallback onToken);

  void cancel(int sessionId);

  std::string getModelInfo() const;

  SessionManager *sessionManager() { return mSessionManager.get(); }

private:
  std::unordered_map<int, std::atomic<bool> *> mActiveCancelFlags;
  llama_model *mModel = nullptr;
  llama_context_params mCtxParams{};
  std::unique_ptr<SessionManager> mSessionManager;
  std::string mModelPath;
  llama_sampler *mSampler = nullptr; // 신규 — top_k/top_p/temp/dist 체인
};

} // namespace miniv::ai