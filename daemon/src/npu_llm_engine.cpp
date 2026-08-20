#include "npu_llm_engine.h"
#include "llama.h"

#include <android/log.h>
#include <vector>

// Custom file based MINI-V Logger: for Permanant logger  
// avc denial 전무, permissive에서도 logd 관련 시도 자체 없음 확인.
#include <cstdio>
namespace miniv::ai {
static void miniv_file_log(const char *level, const char *msg) {
  FILE *f = fopen("/data/vendor/miniv_ai/debug.log", "a");
  if (f) {
    fprintf(f, "[%s] %s\n", level, msg);
    fclose(f);
  }
}
}

#define LOG_TAG "NpuLLMEngine"
#define LOGI(...) do { \
  __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__); \
  char miniv_logbuf[1024]; \
  snprintf(miniv_logbuf, sizeof(miniv_logbuf), __VA_ARGS__); \
  ::miniv::ai::miniv_file_log("I", miniv_logbuf); \
} while (0)
#define LOGE(...) do { \
  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__); \
  char miniv_logbuf[1024]; \
  snprintf(miniv_logbuf, sizeof(miniv_logbuf), __VA_ARGS__); \
  ::miniv::ai::miniv_file_log("E", miniv_logbuf); \
} while (0)

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
    setenv("DSP_LIBRARY_PATH", "/data/vendor/miniv_ai:/vendor/lib64/rfsa/adsp", 1);
    setenv("GGML_SCHED_DEBUG", "2", 1);
    // 라이브러리 내부 로그 찍기용 
    freopen("/data/vendor/miniv_ai/stderr.log", "a", stderr);
    setvbuf(stderr, nullptr, _IOLBF, 0);
  
    llama_backend_init();

    // HTP 백엔드 스캔 -> dlopen
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
    mparams.n_gpu_layers = 999;

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

    cparams.n_batch = 2048;
    cparams.n_ubatch = 512;
    cparams.embeddings = true;

    mCtx = llama_new_context_with_model(mModel, cparams);
    if (!mCtx) {
        LOGE("NPU context init failed");
        return false;
    } 

    auto sparams = llama_sampler_chain_default_params();
    mSampler = llama_sampler_chain_init(sparams);
    // llama_sampler_chain_add(mSampler, llama_sampler_init_greedy()); // 그냥 greedy 디버깅용
    llama_sampler_chain_add(mSampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(mSampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(mSampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(mSampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    {
      llama_token bosToken = llama_token_bos(mModel);
      if (bosToken == LLAMA_TOKEN_NULL) {
        bosToken = 0;  // BOS가 없는 모델 대비 폴백
      }
      llama_batch warmupBatch = llama_batch_get_one(&bosToken, 1);
      if (llama_decode(mCtx, warmupBatch) != 0) {
        LOGE("warmup decode failed");
      }
      llama_kv_cache_clear(mCtx);
      llama_synchronize(mCtx);
      LOGI("warmup decode done (bos=%d)", bosToken);
    }
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
    {   
    std::string tokStr;
    for (int i = 0; i < nPromptTokens; ++i) {
      tokStr += std::to_string(promptTokens[i]) + " ";
    }
    LOGI("prompt token ids: %s", tokStr.c_str());
  }
  {
    llama_batch batch = llama_batch_get_one(promptTokens.data(), nPromptTokens);
      
    if (llama_decode(mCtx, batch) != 0) {
      LOGE("prefill llama_decode failed");
      return false;
    }
  }
  llama_synchronize(mCtx);

  // llama_get_logits_ith(-1) 사용 
  {
    float *logits = llama_get_logits_ith(mCtx, -1);
    if (!logits) {
      LOGE("llama_get_logits_ith(-1) returned NULL");
    } else {
      LOGI("logits[0..4] = %f %f %f %f %f", logits[0], logits[1], logits[2],
           logits[3], logits[4]);
    }
  }
  {
    float *embd = llama_get_embeddings_ith(mCtx, -1);
    if (!embd) {
      LOGE("llama_get_embeddings_ith(-1) returned NULL");
    } else {
      LOGI("result_norm[0..4] = %f %f %f %f %f", embd[0], embd[1], embd[2],
           embd[3], embd[4]);
    }
  }

  int nPast = nPromptTokens;

  for (int i = 0; i < maxTokens; ++i) {
    llama_token tok = llama_sampler_sample(mSampler, mCtx, -1);
    llama_sampler_accept(mSampler, tok);

    if (i < 3) {
      LOGI("token[%d]: id=%d", i, tok);
    }

    if (llama_token_is_eog(mModel, tok)) {
      LOGI("EOG hit at i=%d, tok=%d", i, tok);
      break;
    }

    char buf[256];
    int n = llama_token_to_piece(mModel, tok, buf, sizeof(buf), 0, true);
    onToken(std::string(buf, n));

    // 다음 토큰도 동일하게 llama_batch_get_one 사용 
    {
      llama_batch batch = llama_batch_get_one(&tok, 1);
      if (batch.pos) batch.pos[0] = nPast;
      if (batch.seq_id) batch.seq_id[0][0] = 0;
      if (batch.logits) batch.logits[0] = 1;
      if (llama_decode(mCtx, batch) != 0) {
        LOGE("decode loop failed at i=%d", i);
        break;
      }
    }
    llama_synchronize(mCtx);
    nPast++;
  }
  return true;
}

}    // namespace miniv::ai
