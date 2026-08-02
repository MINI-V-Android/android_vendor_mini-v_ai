#pragma once
#include <aidl/vendor/miniv/ai/BnMiniVAiHal.h>
#include "llm_engine.h"

namespace miniv::ai {

class MiniVAiHalService : public aidl::vendor::miniv::ai::BnMiniVAiHal {
public:
  explicit MiniVAiHalService(LLMEngine* engine) : mEngine(engine) {}

  ndk::ScopedAStatus isReady(bool* _aidl_return) override;
  ndk::ScopedAStatus createSession(int32_t sessionId, int32_t* _aidl_return) override;
  ndk::ScopedAStatus destroySession(int32_t sessionId, int32_t* _aidl_return) override;
  ndk::ScopedAStatus inferStream(
      int32_t sessionId, const std::string& prompt, int32_t maxTokens,
      const std::shared_ptr<aidl::vendor::miniv::ai::IMiniVAiStreamCallback>& callback,
      int32_t* _aidl_return) override;
  ndk::ScopedAStatus cancel(int32_t sessionId) override;
  ndk::ScopedAStatus getModelInfo(std::string* _aidl_return) override;

private:
  LLMEngine* mEngine;
};

}  // namespace miniv::ai