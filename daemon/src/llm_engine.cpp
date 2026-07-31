#include "llm_engine.h"
#include "llama.h"
#include <android/log.h>
#include <vector>

#define LOG_TAG "LLMEngine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace miniv::ai {
namespace {

// TODO: llama_sampler_chain(top_k+top_p+temp+dist) API로 교체 예정.
// git log 확인 결과 llm_engine.cpp 커밋이 1개뿐이라 예전 구현을 복원할 방법이
// 없음 -> 새로 작성 필요. 지금은 순수 그리디(argmax) 임시 구현.
llama_token sampleNext(llama_context *ctx) {
  const llama_model *model = llama_get_model(ctx);
  const llama_vocab *vocab = llama_model_get_vocab(model);
  int nVocab = llama_vocab_n_tokens(vocab);

  float *logits = llama_get_logits_ith(ctx, -1);
  llama_token best = 0;
  float bestVal = logits[0];
  for (int i = 1; i < nVocab; ++i) {
    if (logits[i] > bestVal) {
      bestVal = logits[i];
      best = i;
    }
  }
  return best;
}

} // namespace

LLMEngine::~LLMEngine() {
  // mSessionManager는 unique_ptr이라 이 소멸자 진입 시 아직 살아있음 ->
  // SessionManager::~SessionManager()가 먼저 자동 호출되어 HOT 세션들의
  // llama_context를 정리함 (그다음 이 소멸자 본문 실행)
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
                      TokenCallback onToken) {
  llama_context *ctx = mSessionManager->activateForInfer(sessionId);
  if (!ctx)
    return false; // main.cpp가 ERROR SESSION_NOT_FOUND로 응답

  const llama_vocab *vocab = llama_model_get_vocab(mModel);

  // --- prefill ---
  int nPromptTokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                      nullptr, 0, true, true);
  std::vector<llama_token> promptTokens(nPromptTokens);
  llama_tokenize(vocab, prompt.c_str(), prompt.size(), promptTokens.data(),
                 nPromptTokens, true, true);

  llama_batch batch =
      llama_batch_get_one(promptTokens.data(), promptTokens.size());
  if (llama_decode(ctx, batch) != 0)
    return false;
  mSessionManager->recordTokens(sessionId, promptTokens.data(),
                                promptTokens.size());

  // --- decode loop ---
  std::vector<llama_token> generated;
  generated.reserve(maxTokens);

  for (int i = 0; i < maxTokens; ++i) {
    llama_token tok = sampleNext(ctx);
    if (llama_vocab_is_eog(vocab, tok))
      break;

    char buf[256];
    int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
    onToken(std::string(buf, n));

    generated.push_back(tok);
    llama_batch nextBatch = llama_batch_get_one(&tok, 1);
    if (llama_decode(ctx, nextBatch) != 0)
      break;
  }

  mSessionManager->recordTokens(sessionId, generated.data(), generated.size());
  return true;
}

} // namespace miniv::ai