#include "npu_llm_engine.h"
#include "llama.h"

#include "ggml-htp.h"
#include "ggml-alloc.h"
#include <cstring>
// #include <dlfcn.h>

#include <android/log.h>
#include <vector>

// Custom file based MINI-V Logger: for Permanant logger
// 결론(avc denial 전무, permissive에서도 logd 관련 시도 자체 없음 확인).
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
    setenv("DSP_LIBRARY_PATH", "/vendor/lib64/rfsa/adsp", 1);

    // >>> MINI-V 유지 (라이브러리 내부 fprintf(stderr,...) 로그 확보용)
    freopen("/data/vendor/miniv_ai/stderr.log", "a", stderr);
    setvbuf(stderr, nullptr, _IOLBF, 0);
  
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

    mCtx = llama_new_context_with_model(mModel, cparams);
    if (!mCtx) {
        LOGE("NPU context init failed");
        return false;
    } // 8/19확인용
    {
        LOGI("=== HTP diag: reusing this process's existing HTP context ===");
        ggml_backend_reg_t diagReg = ggml_backend_htp_reg();
        ggml_backend_dev_t diagDev = ggml_backend_reg_dev_get(diagReg, 0);
        ggml_backend_t diagBackend = ggml_backend_dev_init(diagDev, nullptr);
        LOGI("HTP diag: backend=%p is_htp=%d", (void*)diagBackend,
             (int)ggml_backend_is_htp(diagBackend));

        size_t diagCtxSize = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
        ggml_init_params diagIparams = { diagCtxSize, nullptr, true };
        ggml_context* diagCtx = ggml_init(diagIparams);
        ggml_tensor* dw = ggml_new_tensor_2d(diagCtx, GGML_TYPE_F16, 32, 32);
        ggml_tensor* da = ggml_new_tensor_2d(diagCtx, GGML_TYPE_F32, 32, 1);
        ggml_tensor* dd = ggml_mul_mat(diagCtx, dw, da);

        ggml_backend_buffer_t diagBuf = ggml_backend_alloc_ctx_tensors(diagCtx, diagBackend);
        LOGI("HTP diag: buf=%p is_rpcmem=%d", (void*)diagBuf,
             diagBuf ? (int)ggml_backend_buft_is_rpcmem(ggml_backend_buffer_get_type(diagBuf)) : -1);

        if (diagBuf) {
            std::vector<uint16_t> dwData(32 * 32, 0x3C00);
            std::vector<float> daData(32, 1.0f);
            ggml_backend_tensor_set(dw, dwData.data(), 0, dwData.size() * sizeof(uint16_t));
            ggml_backend_tensor_set(da, daData.data(), 0, daData.size() * sizeof(float));

            ggml_cgraph* diagGraph = ggml_new_graph(diagCtx);
            ggml_build_forward_expand(diagGraph, dd);


            // >>> MINI-V 추가 [MOD-02] (같은 세션 내 반복 호출 시 결과가 바뀌는지 —
            //     "메시지 채널 첫 사용 시 stale 플래그" 가설 검증. 새 파일/skel 경로
            //     작업 없이 기존 진단 블록만으로 확인 가능해서 MOD-01보다 먼저 시도)
            for (int i = 0; i < 3; ++i) {
                enum ggml_status diagStatus = ggml_backend_graph_compute(diagBackend, diagGraph);
                std::vector<float> diagResult(32, -1.0f);
                ggml_backend_tensor_get(dd, diagResult.data(), 0, diagResult.size() * sizeof(float));
                LOGI("HTP diag [MOD-02] iter=%d status=%d result[0]=%f (expect 32.0)",
                    i, (int)diagStatus, diagResult[0]);
            }
            // <<< MINI-V 추가 끝 [MOD-02]

            enum ggml_status diagStatus = ggml_backend_graph_compute(diagBackend, diagGraph);

            std::vector<float> diagResult(32, -1.0f);
            ggml_backend_tensor_get(dd, diagResult.data(), 0, diagResult.size() * sizeof(float));
            LOGI("HTP diag: status=%d result[0]=%f (expect 32.0)", (int)diagStatus, diagResult[0]);
            ggml_backend_buffer_free(diagBuf);
        }
        ggml_free(diagCtx);
        LOGI("=== HTP diag: done — 지금 이 로그 직전/중 adsprpc CDSP0: 로그 확인할 것 ===");
    }

    auto sparams = llama_sampler_chain_default_params();
    mSampler = llama_sampler_chain_init(sparams);
    // llama_sampler_chain_add(mSampler, llama_sampler_init_greedy());
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

bool NpuLLMEngine::infer(int sessionId, const std::string &prompt, int maxTokens,
                          TokenCallback onToken) {
  if (!mCtx) return false;

  LOGI("infer() called: sessionId=%d maxTokens=%d promptLen=%zu", sessionId,
       maxTokens, prompt.size());

  bool sessionless = (sessionId < 0);  // UDS 단발 질의 — 세션 추적 자체를 안 함

  SessionState *state = nullptr;
  bool isBrandNewSession = false;
  if (!sessionless) {
    auto it = mSessions.find(sessionId);
    isBrandNewSession = (it == mSessions.end());
    if (isBrandNewSession) {
      state = &mSessions.emplace(sessionId, SessionState{}).first->second;
    } else {
      state = &it->second;
    }
  }

  bool freshStart = sessionless || (sessionId != mActiveSessionId);

  if (freshStart) {
    llama_kv_cache_clear(mCtx);
    mCachePos = 0;
    mActiveSessionId = sessionId;

    if (!sessionless && !isBrandNewSession && !state->transcript.empty()) {
      // 콜드 세션 복원 — 저장해둔 트랜스크립트를 조용히(토큰 생성 없이)
      // 재생해서 KV 캐시를 그 세션이 마지막으로 있던 상태로 되돌립니다.
      LOGI("reviving cold session %d — replaying %zu chars of history",
           sessionId, state->transcript.size());
      const std::string &hist = state->transcript;
      int nHistTokens = -llama_tokenize(mModel, hist.c_str(), hist.size(),
                                         nullptr, 0, true, true);
      std::vector<llama_token> histTokens(nHistTokens);
      llama_tokenize(mModel, hist.c_str(), hist.size(), histTokens.data(),
                     nHistTokens, true, true);
      llama_batch histBatch = llama_batch_get_one(histTokens.data(), nHistTokens);
      if (llama_decode(mCtx, histBatch) != 0) {
        LOGE("history replay decode failed for session %d", sessionId);
        return false;
      }
      llama_synchronize(mCtx);
      mCachePos += nHistTokens;
      LOGI("history replay done — cachePos=%d", mCachePos);
    } else {
      LOGI("session %d starting fresh (no history to replay)", sessionId);
    }
  } else {
    LOGI("session continued (sessionId=%d) — KV cache preserved, cachePos=%d",
         sessionId, mCachePos);
  }

  // BOS 등 특수 토큰(add_special)은 "이 세션의 캐시가 지금 완전히 비어있는
  // 상태에서 시작하는 첫 텍스트"일 때만 켭니다 — 세션 없는 단발 질의,
  // 또는 세션의 진짜 첫 턴(트랜스크립트가 지금 이 호출 전까지 비어있던
  // 경우)입니다. 콜드 복원은 재생 블록에서 이미 add_special=true로
  // 처리했으므로, 그 뒤에 이어지는 이번 delta는 add_special=false여야
  // 합니다.
  bool addSpecial = sessionless || (!sessionless && isBrandNewSession &&
                                     state->transcript.empty());
  int nPromptTokens =
      -llama_tokenize(mModel, prompt.c_str(), prompt.size(), nullptr, 0,
                       addSpecial, true);
  std::vector<llama_token> promptTokens(nPromptTokens);
  llama_tokenize(mModel, prompt.c_str(), prompt.size(), promptTokens.data(),
                 nPromptTokens, addSpecial, true);

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

  {
    float *logits = llama_get_logits_ith(mCtx, -1);
    if (!logits) {
      LOGE("llama_get_logits_ith(-1) returned NULL");
    } else {
      LOGI("logits[0..4] = %f %f %f %f %f", logits[0], logits[1], logits[2],
           logits[3], logits[4]);
    }
  }

  mCachePos += nPromptTokens;
  int nPast = mCachePos;  // ★ 이번 호출 로컬이 아니라 누적값에서 시작

  std::string generatedText;  // 트랜스크립트 기록용 — 생성된 토큰을 모음

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
    std::string piece(buf, n);
    onToken(piece);
    generatedText += piece;

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
    mCachePos++;  // ★ 누적 카운터도 같이 전진 — 다음 infer() 호출이 이어받음
  }

  if (state != nullptr) {
    // 이번 턴(입력 delta + 모델이 생성한 응답)을 트랜스크립트에 누적.
    // 이 세션이 나중에 콜드가 됐다가 돌아오면 이 전체가 재생됩니다.
    state->transcript += prompt;
    state->transcript += generatedText;
  }

  return true;
}

void NpuLLMEngine::destroySession(int sessionId) {
  if (sessionId < 0) return;

  auto it = mSessions.find(sessionId);
  if (it == mSessions.end()) return;

  if (sessionId == mActiveSessionId && mCtx) {
    llama_kv_cache_clear(mCtx);
    mCachePos = 0;
    mActiveSessionId = -1;
    LOGI("destroySession(%d): was hot — KV cache cleared", sessionId);
  } else {
    LOGI("destroySession(%d): was cold — transcript dropped", sessionId);
  }

  mSessions.erase(it);
}

}    // namespace miniv::ai