#include "npu_llm_engine.h"

#include <android/log.h>
#include <vector>

#define LOG_TAG "NpuLLMEngine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace miniv::ai {

NpuLLMEngine::~NpuLLMEngine() {
  if (mSampler) llama_sampler_free(mSampler);
  if (mCtx) llama_free(mCtx);
  if (mModel) llama_model_free(mModel);
  llama_backend_free();
}

bool NpuLLMEngine::load(const std::string &modelPath,
                         const std::string &backendLibDir, int nCtx,
                         int nThreads) {
  llama_backend_init();

  // HTP 백엔드(libggml-htp.so 등)를 이 디렉토리에서 스캔해 dlopen.
  // RTLD_LOCAL이 ggml_backend_reg.cpp 내부에 하드코딩돼 있어(§6 확인),
  // CPU 엔진의 동일 파일명 libggml.so와 같은 프로세스에 있어도 심볼
  // 충돌 걱정 없음.
  ggml_backend_load_all_from_path(backendLibDir.c_str());
  mBackendsLoaded = true;

  llama_model_params mparams = llama_model_default_params();
  // 표준 llama.cpp 오프로드 메커니즘 — 브랜드 무관, 등록된 비-CPU
  // 백엔드(HTP)가 있으면 그쪽으로 최대한 레이어를 오프로드함.
  // 실제로 몇 레이어까지 붙는지는(GEMM 오프로드 미적용 이슈, §2 참고)
  // 이 값과 무관하게 커널 구현 자체의 한계일 수 있음 — 별도 추적.
  mparams.n_gpu_layers = 999;

  mModel = llama_model_load_from_file(modelPath.c_str(), mparams);
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

  mCtx = llama_init_from_model(mModel, cparams);
  if (!mCtx) {
    LOGE("NPU context init failed");
    return false;
  }

  // 샘플러는 CPU 엔진과 동일한 값 사용 (§5-2 문서 기준, 잠정치)
  auto sparams = llama_sampler_chain_default_params();
  mSampler = llama_sampler_chain_init(sparams);
  llama_sampler_chain_add(mSampler, llama_sampler_init_top_k(40));
  llama_sampler_chain_add(mSampler, llama_sampler_init_top_p(0.9f, 1));
  llama_sampler_chain_add(mSampler, llama_sampler_init_temp(0.7f));
  llama_sampler_chain_add(mSampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

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

  // 세션이 없으므로 매 호출을 독립된 단발 질문으로 취급 — 무조건 클리어.
  // CPU 라운드 §6에서 겪은 "캐시 재사용으로 이전 대화가 이어지는" 버그를
  // NPU 쪽에서는 설계 자체로 원천 차단.
  llama_memory_clear(llama_get_memory(mCtx), true);

  const llama_vocab *vocab = llama_model_get_vocab(mModel);

  int nPromptTokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                       nullptr, 0, true, true);
  std::vector<llama_token> promptTokens(nPromptTokens);
  llama_tokenize(vocab, prompt.c_str(), prompt.size(), promptTokens.data(),
                 nPromptTokens, true, true);

  llama_batch batch =
      llama_batch_get_one(promptTokens.data(), promptTokens.size());
  if (llama_decode(mCtx, batch) != 0) {
    LOGE("NPU prefill decode failed");
    return false;
  }

  for (int i = 0; i < maxTokens; ++i) {
    llama_token tok = llama_sampler_sample(mSampler, mCtx, -1);
    llama_sampler_accept(mSampler, tok);

    if (llama_vocab_is_eog(vocab, tok)) break;

    char buf[256];
    int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
    onToken(std::string(buf, n));

    llama_batch nextBatch = llama_batch_get_one(&tok, 1);
    if (llama_decode(mCtx, nextBatch) != 0) break;
  }

  return true;
}

}  // namespace miniv::ai