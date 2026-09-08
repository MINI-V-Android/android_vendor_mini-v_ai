#include "npu_llm_engine.h"
#include "rule_evaluator.h"
#include "llama.h"

#include "ggml-htp.h"
#include "ggml-alloc.h"
#include <cstring>
// #include <dlfcn.h>

#include <android/log.h>
#include <vector>
#include <chrono>
#include <thread>
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

static void common_batch_clear(struct llama_batch & batch) {
  batch.n_tokens = 0;
}

static void common_batch_add(
    struct llama_batch & batch,
    llama_token id,
    llama_pos pos,
    const std::vector<llama_seq_id> & seq_ids,
    bool logits) {
  batch.token   [batch.n_tokens] = id;
  batch.pos     [batch.n_tokens] = pos;
  batch.n_seq_id[batch.n_tokens] = seq_ids.size();
  for (size_t i = 0; i < seq_ids.size(); ++i) {
    batch.seq_id[batch.n_tokens][i] = seq_ids[i];
  }
  batch.logits  [batch.n_tokens] = logits;
  batch.n_tokens++;
}

static std::string common_token_to_piece(const struct llama_model * model, llama_token tok) {
  char buf[256];
  int n = llama_token_to_piece(model, tok, buf, sizeof(buf), 0, true);
  if (n < 0) {
    std::vector<char> big_buf(-n);
    int n2 = llama_token_to_piece(model, tok, big_buf.data(), big_buf.size(), 0, true);
    return std::string(big_buf.data(), n2 > 0 ? n2 : 0);
  }
  return std::string(buf, n);
}


bool NpuLLMEngine::infer(int sessionId, const std::string &prompt, int maxTokens,
                          TokenCallback onToken) {
  if (!mCtx) return false;

  LOGI("infer() called (N=2 Best-of-N mode): sessionId=%d maxTokens=%d promptLen=%zu",
       sessionId, maxTokens, prompt.size());

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

  // 초기화
  if (freshStart) {
    llama_kv_cache_clear(mCtx);
    mCachePos = 0;
    mActiveSessionId = sessionId;

    if (!sessionless && !isBrandNewSession && !state->transcript.empty()) {
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

  int nPast = mCachePos;

  // 배치 할당 (최대 프롬프트 크기 및 병렬 2 시퀀스 수용 가능하도록 초기화)
  llama_batch batch = llama_batch_init(std::max(2048, nPromptTokens + 64), 0, 2);

  // 1. 프롬프트 토큰화 및 Prefill 연산 (seq_id = 0)
  common_batch_clear(batch);
  for (int i = 0; i < nPromptTokens; ++i) {
    bool is_last = (i == nPromptTokens - 1);
    common_batch_add(batch, promptTokens[i], nPast + i, { 0 }, is_last);
  }

  if (llama_decode(mCtx, batch) != 0) {
    LOGE("Prefill decode failed");
    llama_batch_free(batch);
    return false;
  }
  llama_synchronize(mCtx);

  // 2. KV 캐시 복제 (seq_id 0의 상태를 seq_id 1로 복사)
  // 무거운 프롬프트 연산을 1번만 하고 상태를 공유합니다.
  llama_kv_cache_seq_cp(mCtx, 0, 1, -1, -1);

  // 3. 샘플러 독립 구성 (다양성 확보 - 서로 다른 Seed 부여)
  llama_sampler * sampler0 = llama_sampler_chain_init(llama_sampler_chain_default_params());
  llama_sampler_chain_add(sampler0, llama_sampler_init_top_k(40));
  llama_sampler_chain_add(sampler0, llama_sampler_init_top_p(0.9f, 1));
  llama_sampler_chain_add(sampler0, llama_sampler_init_temp(0.7f));
  llama_sampler_chain_add(sampler0, llama_sampler_init_dist(1234)); // Seed A

  llama_sampler * sampler1 = llama_sampler_chain_init(llama_sampler_chain_default_params());
  llama_sampler_chain_add(sampler1, llama_sampler_init_top_k(40));
  llama_sampler_chain_add(sampler1, llama_sampler_init_top_p(0.9f, 1));
  llama_sampler_chain_add(sampler1, llama_sampler_init_temp(0.7f));
  llama_sampler_chain_add(sampler1, llama_sampler_init_dist(5678)); // Seed B

  // 4. 병렬 디코딩 루프 (Batch Decode - onToken 호출 없이 버퍼에 조용히 저장)
  std::string text0, text1;
  std::vector<std::string> tokens0, tokens1;
  bool active0 = true, active1 = true;
  int i_batch0 = 0;
  int i_batch1 = 0;
  int curPos = nPast + nPromptTokens;

  for (int step = 0; step < maxTokens; ++step) {
    common_batch_clear(batch);

    // seq 0 샘플링 및 배치 추가
    if (active0) {
      llama_token id0 = llama_sampler_sample(sampler0, mCtx, i_batch0);
      if (llama_token_is_eog(mModel, id0)) {
        active0 = false;
        i_batch0 = -1;
      } else {
        std::string piece = common_token_to_piece(mModel, id0);
        text0 += piece;
        tokens0.push_back(piece);
        i_batch0 = batch.n_tokens;
        common_batch_add(batch, id0, curPos, { 0 }, true);
      }
    }

    // seq 1 샘플링 및 배치 추가
    if (active1) {
      llama_token id1 = llama_sampler_sample(sampler1, mCtx, i_batch1);
      if (llama_token_is_eog(mModel, id1)) {
        active1 = false;
        i_batch1 = -1;
      } else {
        std::string piece = common_token_to_piece(mModel, id1);
        text1 += piece;
        tokens1.push_back(piece);
        i_batch1 = batch.n_tokens;
        common_batch_add(batch, id1, curPos, { 1 }, true);
      }
    }

    if (!active0 && !active1) break;
    if (batch.n_tokens == 0) break;

    // 최대 2개의 토큰을 NPU로 일괄 오프로드하여 추론
    if (llama_decode(mCtx, batch) != 0) {
      LOGE("decode loop failed at step %d", step);
      break;
    }
    llama_synchronize(mCtx);

    curPos++;
  }

  // 5. 호스트 CPU 기반 평가 (더미 랜덤 / 룰베이스)
  int score0 = miniv::ai::evaluate_response(text0);
  int score1 = miniv::ai::evaluate_response(text1);

  LOGI("N=2 evaluation: seq0 score=%d len=%zu, seq1 score=%d len=%zu",
       score0, text0.size(), score1, text1.size());

  int bestSeq = (score1 > score0) ? 1 : 0;
  const std::string &bestText = (bestSeq == 1) ? text1 : text0;
  const std::vector<std::string> &bestTokens = (bestSeq == 1) ? tokens1 : tokens0;
  int bestTokenCount = bestTokens.size();

  LOGI("Best Response (Seq %d, score %d): %s",
       bestSeq, (bestSeq == 1 ? score1 : score0), bestText.c_str());

  // 승자 시퀀스의 KV 캐시 동기화 및 패자 시퀀스 제거
  if (bestSeq == 0) {
    llama_kv_cache_seq_rm(mCtx, 1, -1, -1);
  } else {
    llama_kv_cache_seq_rm(mCtx, 0, nPast + nPromptTokens, -1);
    llama_kv_cache_seq_cp(mCtx, 1, 0, nPast + nPromptTokens, -1);
    llama_kv_cache_seq_rm(mCtx, 1, -1, -1);
  }

  mCachePos += nPromptTokens + bestTokenCount;

  // 6. 승리한 시퀀스의 토큰만 앱으로 스트리밍 (지연 스트리밍: Delayed Streaming)
  if (onToken) {
    for (const auto &token_piece : bestTokens) {
      onToken(token_piece);
      // 앱 UI 렌더링(타이핑 효과)을 위해 미세한 딜레이가 필요하다면 아래 주석 해제
      // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  // 7. 트랜스크립트(대화 기록) 업데이트
  if (state != nullptr) {
    state->transcript += prompt;
    state->transcript += bestText;
  }

  llama_sampler_free(sampler0);
  llama_sampler_free(sampler1);
  llama_batch_free(batch);

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