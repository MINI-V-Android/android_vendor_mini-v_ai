#include "llm_engine.h"
#include "llama.h"
#include <vector>

#define LOG_TAG "LLMEngine"
#include "miniv_log.h"

namespace miniv::ai {

LLMEngine::~LLMEngine() {
  // mSessionManager 소멸(unique_ptr) -> HOT 세션들의 llama_context 정리가 먼저
  // 실행됨
  if (mSampler)
    llama_sampler_free(mSampler); // 체인 하나만 free하면 내부
                                  // top_k/top_p/temp/dist 전부 정리됨
  if (mModel)
    llama_model_free(mModel);
  llama_backend_free();
}

bool LLMEngine::load(const std::string &modelPath, int nCtx, int nThreads) {
  llama_backend_init();

  llama_model_params mparams = llama_model_default_params();
  mModel = llama_model_load_from_file(modelPath.c_str(), mparams);
  if (!mModel) {
    LOGE("model load failed: %s", modelPath.c_str());
    return false;
  }
  mModelPath = modelPath;

  mCtxParams = llama_context_default_params();
  mCtxParams.n_ctx = nCtx;
  mCtxParams.n_threads = nThreads;

  mSessionManager = std::make_unique<SessionManager>(
      mModel, mCtxParams, "/data/vendor/miniv_ai/sessions");

  mSessionManager->setHotLimit(SessionManager::kDefaultHotLimit);
  mSessionManager->setColdLimit(SessionManager::kDefaultColdLimitBytes);

  // --- 샘플러 체인 구성: top_k -> top_p -> temp -> dist ---
  // 값(k=40, p=0.9, temp=0.7)은 CPU 라운드 문서에 적힌 "그리디에 가까운 단순
  // 샘플러"에 맞춘 잠정치. 실기기 응답 품질 보고 튜닝 필요.
  auto sparams = llama_sampler_chain_default_params();
  mSampler = llama_sampler_chain_init(sparams);
  llama_sampler_chain_add(mSampler, llama_sampler_init_top_k(40));
  llama_sampler_chain_add(mSampler, llama_sampler_init_top_p(0.9f, 1));
  llama_sampler_chain_add(mSampler, llama_sampler_init_temp(0.7f));
  llama_sampler_chain_add(mSampler,
                          llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

  LOGI("model loaded: %s (ctx=%d, threads=%d)", modelPath.c_str(), nCtx,
       nThreads);
  return true;
}

std::string LLMEngine::getModelInfo() const {
  if (!mModel)
    return "not loaded";
  return mModelPath;
}

bool LLMEngine::infer(int sessionId, const std::string &prompt, int maxTokens,
                      TokenCallback onToken, std::atomic<bool> *cancelFlag) {
  llama_context *ctx = mSessionManager->activateForInfer(sessionId);
  if (!ctx){
    LOGE("infer() session=%d activateForInfer FAILED (session not found?)", sessionId);
    return false; // main.cpp가 ERROR SESSION_NOT_FOUND로 응답
  }
  const llama_vocab *vocab = llama_model_get_vocab(mModel);

  // --- prefill ---
  int nPromptTokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                      nullptr, 0, true, true);
  std::vector<llama_token> promptTokens(nPromptTokens);
  llama_tokenize(vocab, prompt.c_str(), prompt.size(), promptTokens.data(),
                 nPromptTokens, true, true);

  LOGI("infer() session=%d maxTokens=%d nPromptTokens=%d",
      sessionId, maxTokens, nPromptTokens);

  llama_batch batch =
      llama_batch_get_one(promptTokens.data(), promptTokens.size());
  if (llama_decode(ctx, batch) != 0){
    LOGE("infer() session=%d prefill llama_decode FAILED", sessionId);
    return false;
  }
  mSessionManager->recordTokens(sessionId, promptTokens.data(),
                                promptTokens.size());

  // --- decode loop ---
  std::vector<llama_token> generated;
  generated.reserve(maxTokens);

  for (int i = 0; i < maxTokens; ++i) {
    if (cancelFlag && cancelFlag->load()) 
    {
      LOGI("infer() session=%d cancelled at i=%d", sessionId, i);
      break;  // Cancel 여부 확인
    }
    llama_token tok = llama_sampler_sample(mSampler, ctx, -1);
    llama_sampler_accept(mSampler, tok);

    if (llama_vocab_is_eog(vocab, tok)){
      LOGI("infer() session=%d EOG at i=%d tok=%d", sessionId, i, tok);
      break;
    }
    char buf[256];
    int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
    onToken(std::string(buf, n));

    generated.push_back(tok);
    llama_batch nextBatch = llama_batch_get_one(&tok, 1);
    if (llama_decode(ctx, nextBatch) != 0) {
      LOGE("infer() session=%d decode loop FAILED at i=%d", sessionId, i);
      break;
    }
  }
  LOGI("infer() session=%d done, generated=%zu tokens", sessionId, generated.size());
  mSessionManager->recordTokens(sessionId, generated.data(), generated.size());
  return true;
}

} // namespace miniv::ai