#include "npu_llm_engine.h"
#include "llama.h"

#include <android/log.h>
#include <vector>

#define LOG_TAG "NpuLLMEngine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace miniv::ai {

NpuLLMEngine::~NpuLLMEngine() {
    if (mSampler) llama_sampler_free(mSampler);
    if (mCtx) llama_free(mCtx);
    if (mModel) llama_free_model(mModel);
    llama_backend_free();
}

bool NpuLLMEngine::load(const std::string &modelPath,
                         const std::string &backendLibDir, int nCtx,
                         int nThreads) {
    llama_backend_init();

    // HTP 백엔드를 스캔 -> dlopen
    ggml_backend_load_all_from_path(backendLibDir.c_str());
    mBackendsLoaded = true;
    
    {
      size_t devCount = ggml_backend_dev_count();
      LOGI("registered backend devices: %zu", devCount);
      for (size_t d = 0; d < devCount; ++d) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(d);
        LOGI("  device[%zu]: name=%s desc=%s type=%d", d,
             ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
             (int)ggml_backend_dev_type(dev));
      }
    }

    llama_model_params mparams = llama_model_default_params();
    // 백엔드(HTP)가 있으면 그쪽으로 최대한 레이어를 오프로드
    mparams.n_gpu_layers = 0;

    mModel = llama_load_model_from_file(modelPath.c_str(), mparams);
    if (!mModel) {
        LOGE("NPU model load failed: %s", modelPath.c_str());
        return false;
    }
    mModelPath = modelPath;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = nCtx;
    cparams.n_threads = nThreads;
    cparams.n_threads_batch = nThreads;
    cparams.flash_attn = true; 

    mCtx = llama_new_context_with_model(mModel, cparams);
    if (!mCtx) {
        LOGE("NPU context init failed");
        return false;
    }

    // 샘플러는 CPU 엔진과 동일한 값 사용 (§5-2 문서 기준, 잠정치)
    auto sparams = llama_sampler_chain_default_params();
    mSampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(mSampler, llama_sampler_init_greedy());
    // llama_sampler_chain_add(mSampler, llama_sampler_init_top_k(40));
    // llama_sampler_chain_add(mSampler, llama_sampler_init_top_p(0.9f, 1));
    // llama_sampler_chain_add(mSampler, llama_sampler_init_temp(0.7f));
    // llama_sampler_chain_add(mSampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    LOGI("NPU model loaded: %s (ctx=%d, threads=%d, backendDir=%s)",
             modelPath.c_str(), nCtx, nThreads, backendLibDir.c_str());
    return true;
}

std::string NpuLLMEngine::getModelInfo() const {
    return mModel ? mModelPath : "not loaded";
}

bool NpuLLMEngine::infer(const std::string &prompt, int maxTokens,
                          TokenCallback onToken) {
  if (!mCtx) return false;

  LOGI("infer() called: maxTokens=%d, promptLen=%zu", maxTokens, prompt.size());

  llama_kv_cache_clear(mCtx);

  int nPromptTokens = -llama_tokenize(mModel, prompt.c_str(), prompt.size(),
                                       nullptr, 0, true, true);
  std::vector<llama_token> promptTokens(nPromptTokens);
  llama_tokenize(mModel, prompt.c_str(), prompt.size(), promptTokens.data(),
                 nPromptTokens, true, true);

  LOGI("tokenized: nPromptTokens=%d", nPromptTokens);

  // 이슈 #44 — llama_batch_get_one()의 "logits=NULL이면 마지막 토큰만
  // 암묵적으로 출력"이라는 경로가 이 포크의 ACCEL(HTP) 스케줄링에서
  // 안 타는 것으로 강하게 의심됨(llama_synchronize 추가해도 결과 불변,
  // 즉 "안 끝난 연산을 늦게 읽음"이 아니라 "연산 자체가 그래프에서 빠짐").
  // llama_batch_init()으로 직접 구성 + logits[last]=1을 명시적으로 세팅해
  // 이 암묵 경로 자체를 우회.
  llama_batch batch = llama_batch_init(nPromptTokens, 0, 1);
  for (int i = 0; i < nPromptTokens; ++i) {
    batch.token[i]     = promptTokens[i];
    batch.pos[i]       = i;
    batch.n_seq_id[i]  = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i]    = (i == nPromptTokens - 1) ? 1 : 0;
  }
  batch.n_tokens = nPromptTokens;

  if (llama_decode(mCtx, batch) != 0) {
    LOGE("prefill llama_decode failed");
    llama_batch_free(batch);
    return false;
  }
  llama_synchronize(mCtx);

  {
    float *logits = llama_get_logits(mCtx);
    if (!logits) {
      LOGE("llama_get_logits returned NULL");
    } else {
      LOGI("logits[0..4] = %f %f %f %f %f", logits[0], logits[1], logits[2],
           logits[3], logits[4]);
      float maxVal = logits[0];
      int maxIdx = 0;
      for (int v = 1; v < 151936; ++v) {
        if (logits[v] > maxVal) { maxVal = logits[v]; maxIdx = v; }
      }
      LOGI("manual argmax: idx=%d val=%f", maxIdx, maxVal);
    }
  }

  llama_batch_free(batch);
  int nPast = nPromptTokens;   // llama_batch_init은 위치 자동추적 없음 — 직접 관리

  for (int i = 0; i < maxTokens; ++i) {
    llama_token tok = llama_sampler_sample(mSampler, mCtx, -1);
    llama_sampler_accept(mSampler, tok);

    if (i < 3) {
      LOGI("token[%d]: id=%d, n_vocab=%d", i, tok, llama_n_vocab(mModel));
    }

    if (llama_token_is_eog(mModel, tok)) {
      LOGI("EOG hit at i=%d, tok=%d", i, tok);
      break;
    }

    char buf[256];
    int n = llama_token_to_piece(mModel, tok, buf, sizeof(buf), 0, true);
    onToken(std::string(buf, n));

    llama_batch nextBatch = llama_batch_init(1, 0, 1);
    nextBatch.token[0]     = tok;
    nextBatch.pos[0]       = nPast;
    nextBatch.n_seq_id[0]  = 1;
    nextBatch.seq_id[0][0] = 0;
    nextBatch.logits[0]    = 1;
    nextBatch.n_tokens     = 1;

    bool decodeOk = llama_decode(mCtx, nextBatch) == 0;
    llama_batch_free(nextBatch);
    if (!decodeOk) {
      LOGE("decode loop failed at i=%d", i);
      break;
    }
    llama_synchronize(mCtx);
    nPast++;
  }
  return true;
}

}    // namespace miniv::ai