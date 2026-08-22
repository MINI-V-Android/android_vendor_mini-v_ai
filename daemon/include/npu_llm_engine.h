#pragma once
#include <functional>
#include <string>

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

  bool infer(const std::string &prompt, int maxTokens, TokenCallback onToken);

  std::string getModelInfo() const;

private:
  llama_model *mModel = nullptr;
  llama_context *mCtx = nullptr;
  llama_sampler *mSampler = nullptr;
  std::string mModelPath;
  bool mBackendsLoaded = false;
};

}  // namespace miniv::ai