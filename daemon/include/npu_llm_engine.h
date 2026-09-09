#pragma once
#include <functional>
#include <string>
#include <unordered_map>

struct llama_model;
struct llama_context;
struct llama_sampler;

namespace miniv::ai {

enum class DecodeMode {
  AUTO = 0,   // Check Android property / Env / Default (MULTI)
  MULTI = 1,  // Best-of-N (N=2, Delayed Streaming, Rule Evaluator)
  SINGLE = 2, // Single (N=1, Real-time Immediate Streaming)
};

class NpuLLMEngine {
public:
  using TokenCallback = std::function<void(const std::string &tokenText)>;

  ~NpuLLMEngine();

  bool load(const std::string &modelPath, const std::string &backendLibDir,
            int nCtx, int nThreads);

  bool isReady() const { return mModel != nullptr; }
  
  bool infer(int sessionId, const std::string &prompt, int maxTokens,
             TokenCallback onToken, DecodeMode mode = DecodeMode::AUTO);

  void setGlobalDecodeMode(DecodeMode mode) { mGlobalDecodeMode = mode; }
  DecodeMode getGlobalDecodeMode() const { return mGlobalDecodeMode; }

  void destroySession(int sessionId);

  std::string getModelInfo() const;

private:
  struct SessionState {
    std::string transcript;
  };

  DecodeMode resolveDecodeMode(DecodeMode requestedMode) const;

  bool inferSingle(int sessionId, const std::string &prompt, int maxTokens,
                   TokenCallback onToken, SessionState *state,
                   bool sessionless, bool isBrandNewSession);

  bool inferMulti(int sessionId, const std::string &prompt, int maxTokens,
                  TokenCallback onToken, SessionState *state,
                  bool sessionless, bool isBrandNewSession);

  llama_model *mModel = nullptr;
  llama_context *mCtx = nullptr;
  llama_sampler *mSampler = nullptr;
  std::string mModelPath;
  bool mBackendsLoaded = false;
  DecodeMode mGlobalDecodeMode = DecodeMode::AUTO;

  std::unordered_map<int, SessionState> mSessions;

  int mActiveSessionId = -1;  
  int mCachePos = 0;          // mActiveSessionId의 KV 캐시에 쌓인 토큰 수
};

}  // namespace miniv::ai