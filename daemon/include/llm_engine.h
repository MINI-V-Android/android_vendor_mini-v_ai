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
  std::string getModelInfo() const;
  SessionManager *sessionManager() { return mSessionManager.get(); }

private:
  llama_model *mModel = nullptr;
  llama_context_params mCtxParams{};
  std::unique_ptr<SessionManager> mSessionManager;
  std::string mModelPath;
};

} // namespace miniv::ai