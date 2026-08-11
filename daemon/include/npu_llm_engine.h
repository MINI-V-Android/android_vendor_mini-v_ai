#pragma once
#include "llama.h"
#include <functional>
#include <string>

namespace miniv::ai {

// 세션/멀티턴 없음 — 단일 그래프, 단발 질문만 지원 (§1, §3-3)
class NpuLLMEngine {
public:
  using TokenCallback = std::function<void(const std::string &tokenText)>;

  ~NpuLLMEngine();

  // backendLibDir: libggml-htp.so 등이 위치한 디렉토리
  //   (vendor prebuilt 배치 경로, §4 — 아직 미확정, 임시로 인자화)
  bool load(const std::string &modelPath, const std::string &backendLibDir,
            int nCtx, int nThreads);

  bool isReady() const { return mModel != nullptr; }

  // 세션ID/cancelFlag 없음 — 단발 호출 전용. 매 호출 시작 시 KV 캐시를
  // 무조건 clear하므로 이전 호출의 맥락이 절대 안 이어짐 (의도된 설계)
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