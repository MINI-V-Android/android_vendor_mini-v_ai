#pragma once
#include <functional>
#include <string>
#include <unordered_map>

struct llama_model;
struct llama_context;
struct llama_sampler;

namespace miniv::ai {

class NpuLLMEngine {
public:
  using TokenCallback = std::function<void(const std::string &tokenText)>;

  ~NpuLLMEngine();

  bool load(const std::string &modelPath, const std::string &backendLibDir,
            int nCtx, int nThreads);

  bool isReady() const { return mModel != nullptr; }
  
  bool infer(int sessionId, const std::string &prompt, int maxTokens,
             TokenCallback onToken);

  void destroySession(int sessionId);

  std::string getModelInfo() const;

private:
  llama_model *mModel = nullptr;
  llama_context *mCtx = nullptr;
  llama_sampler *mSampler = nullptr;
  std::string mModelPath;
  bool mBackendsLoaded = false;

  struct SessionState {
    std::string transcript;
  };
  std::unordered_map<int, SessionState> mSessions;

  int mActiveSessionId = -1;  
  int mCachePos = 0;          // mActiveSessionId의 KV 캐시에 쌓인 토큰 수
};

}  // namespace miniv::ai