#pragma once
#include <atomic>
#include <functional>
#include <string>

class LLMEngine {
public:
  using TokenCallback = std::function<void(const std::string &tokenText)>;

  LLMEngine() = default;
  ~LLMEngine();

  // 모델 로드. 실패 시 false.
  bool load(const std::string &modelPath, int nThreads = 4, int nCtx = 2048);

  // 동기 추론. 토큰마다 callback 호출. cancelled가 true가 되면 루프 중단.
  // onComplete: true면 EOS까지 정상 도달, false면 cancel/max_tokens로 조기
  // 종료.
  void infer(const std::string &prompt, int maxTokens,
             const TokenCallback &onToken, std::atomic<bool> &cancelled);

  bool isReady() const { return mModel != nullptr; }
  std::string getModelInfo() const { return mModelPath; }

private:
  struct llama_model *mModel = nullptr;
  struct llama_context *mCtx = nullptr;
  const struct llama_vocab *mVocab = nullptr;
  std::string mModelPath;
  int mNCtx = 2048;
};