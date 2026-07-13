#include "llm_engine.h"
#include "llama.h"
#include <android/log.h>
#include <vector>

#define LOG_TAG "LLMEngine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

LLMEngine::~LLMEngine() {
  if (mCtx)
    llama_free(mCtx);
  if (mModel)
    llama_model_free(mModel);
  llama_backend_free();
}

bool LLMEngine::load(const std::string &modelPath, int nThreads, int nCtx) {
  llama_backend_init();
  mNCtx = nCtx;

  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = 0; // CPU 전용 (NPU 백엔드는 다음 라운드)

  mModel = llama_model_load_from_file(modelPath.c_str(), mparams);
  if (!mModel) {
    LOGE("model load failed: %s", modelPath.c_str());
    return false;
  }
  mVocab = llama_model_get_vocab(mModel);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = nCtx;
  cparams.n_threads = nThreads;
  cparams.n_threads_batch = nThreads;

  mCtx = llama_init_from_model(mModel, cparams);
  if (!mCtx) {
    LOGE("context init failed");
    return false;
  }

  mModelPath = modelPath;
  LOGI("model loaded: %s (ctx=%d, threads=%d)", modelPath.c_str(), nCtx,
       nThreads);
  return true;
}

void LLMEngine::infer(const std::string &prompt, int maxTokens,
                      const TokenCallback &onToken,
                      std::atomic<bool> &cancelled) {
  if (!mModel || !mCtx) {
    LOGE("infer() called before load()");
    return;
  }

  // 1. 토크나이즈
  std::vector<llama_token> tokens(prompt.size() + 8);
  int nTok = llama_tokenize(mVocab, prompt.c_str(), (int)prompt.size(),
                            tokens.data(), (int)tokens.size(), true, true);
  if (nTok < 0) {
    tokens.resize(-nTok);
    nTok = llama_tokenize(mVocab, prompt.c_str(), (int)prompt.size(),
                          tokens.data(), (int)tokens.size(), true, true);
  }
  tokens.resize(nTok);

  // 2. prefill
  llama_batch batch = llama_batch_get_one(tokens.data(), (int)tokens.size());
  if (llama_decode(mCtx, batch) != 0) {
    LOGE("prefill decode failed");
    return;
  }

  // 3. 샘플러 체인 (탐욕적 샘플링 — 검증용이니 단순하게)
  llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
  llama_sampler *sampler = llama_sampler_chain_init(sparams);
  llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
  llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
  llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234));

  // 4. decode loop
  int nGenerated = 0;
  llama_token newToken;
  char pieceBuf[256];

  while (nGenerated < maxTokens && !cancelled.load()) {
    newToken = llama_sampler_sample(sampler, mCtx, -1);

    if (llama_vocab_is_eog(mVocab, newToken)) {
      LOGI("EOS reached at token %d", nGenerated);
      break;
    }

    int n = llama_token_to_piece(mVocab, newToken, pieceBuf, sizeof(pieceBuf),
                                 0, true);
    if (n > 0) {
      onToken(std::string(pieceBuf, n));
    }

    llama_batch nextBatch = llama_batch_get_one(&newToken, 1);
    if (llama_decode(mCtx, nextBatch) != 0) {
      LOGE("decode failed at token %d", nGenerated);
      break;
    }
    nGenerated++;
  }

  llama_sampler_free(sampler);
}