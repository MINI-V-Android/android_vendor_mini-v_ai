// 이슈 #48 진단용 — htp_session_test(세션/채널 성공 확인)의 다음 단계.
// 실제 32x32 MUL_MAT 하나를 ggml_backend_htp를 통해 돌려서
//   (a) htp_ops_support_op()가 true를 주는지 → HTP 경로로 실제로 들어가는지
//   (b) 결과값이 수학적으로 맞는지(전부 1.0으로 채운 32차원 내적 = 32.0이어야 함)
// 를 동시에 확인한다. logcat -s adsprpc를 같이 띄워두면 CDSP0: invoke 로그
// 발생 여부도 이 실행 한 번으로 같이 확인 가능.

#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-htp.h"

// IEEE754 half-precision 1.0
static const uint16_t F16_ONE = 0x3C00;

int main() {
    printf("[1] ggml_backend_htp_reg() ...\n");
    ggml_backend_reg_t reg = ggml_backend_htp_reg();
    if (!reg) { printf("[1] FAILED: reg is null\n"); return 1; }
    printf("[1] OK reg=%p\n", (void*)reg);

    printf("[2] ggml_backend_reg_dev_get(reg, 0) ...\n");
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    if (!dev) { printf("[2] FAILED: dev is null\n"); return 1; }
    printf("[2] OK dev=%p\n", (void*)dev);

    // 여기서 ggml_backend_htp_context 생성자가 실행됨 (dlopen/open_session/init_message_channel 전부)
    printf("[3] ggml_backend_dev_init(dev, nullptr) ... (여기서 HTP 세션 초기화 발생, stderr 확인)\n");
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) { printf("[3] FAILED: backend is null\n"); return 1; }
    printf("[3] OK backend=%p is_htp=%d\n", (void*)backend, (int)ggml_backend_is_htp(backend));

    // ── 32x32 MUL_MAT 그래프 구성 ──
    // weight: F16 [k=32, n=32] (htp_ops_support_op의 "FP16 weight" 케이스)
    // act   : F32 [k=32, m=1]
    // dst   : F32 [n=32, m=1]  (mul_mat 결과)
    printf("[4] building 32x32 F16-weight MUL_MAT graph ...\n");
    size_t ctx_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
    ggml_init_params iparams = { ctx_size, nullptr, /*no_alloc=*/true };
    ggml_context* ctx = ggml_init(iparams);
    if (!ctx) { printf("[4] FAILED: ggml_init\n"); return 1; }

    ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 32, 32);
    ggml_tensor* act    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 1);
    ggml_tensor* dst    = ggml_mul_mat(ctx, weight, act);
    printf("[4] dst type=%d (expect F32=%d)\n", (int)dst->type, (int)GGML_TYPE_F32);

    printf("[5] ggml_backend_alloc_ctx_tensors(ctx, backend) ... (RPCMEM 버퍼에 배치되는지 확인)\n");
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { printf("[5] FAILED: buffer alloc\n"); return 1; }
    printf("[5] OK buf=%p is_rpcmem=%d\n", (void*)buf,
           (int)ggml_backend_buft_is_rpcmem(ggml_backend_buffer_get_type(buf)));

    // 데이터 채우기: weight 전부 1.0, activation 전부 1.0 → 기대값: dst 전부 32.0
    std::vector<uint16_t> w_data(32 * 32, F16_ONE);
    std::vector<float>    a_data(32, 1.0f);
    ggml_backend_tensor_set(weight, w_data.data(), 0, w_data.size() * sizeof(uint16_t));
    ggml_backend_tensor_set(act,    a_data.data(), 0, a_data.size() * sizeof(float));
    printf("[6] input tensors filled (weight=all 1.0 F16, act=all 1.0 F32)\n");

    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, dst);

    printf("[7] ggml_backend_graph_compute(backend, graph) ...\n");
    enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    printf("[7] returned %d (GGML_STATUS_SUCCESS=%d)\n", (int)status, (int)GGML_STATUS_SUCCESS);

    std::vector<float> result(32, -1.0f);
    ggml_backend_tensor_get(dst, result.data(), 0, result.size() * sizeof(float));

    printf("[8] result[0..4] = %f %f %f %f %f  (기대값: 전부 32.0)\n",
           result[0], result[1], result[2], result[3], result[4]);

    bool all_correct = true;
    for (float v : result) {
        if (std::fabs(v - 32.0f) > 0.01f) { all_correct = false; break; }
    }
    printf("[9] RESULT: %s\n", all_correct ? "CORRECT (실제 HTP 연산 성공, 또는 CPU 폴백이 정확)"
                                            : "WRONG (연산 오염 — HTP 경로 자체의 버그 가능성)");
    printf("    ※ 이 결과가 CORRECT이어도, 진짜 HTP를 탄 건지 CPU 폴백인지는\n");
    printf("       이 실행과 동시에 띄워둔 'adb logcat -s adsprpc:V *:S'에\n");
    printf("       CDSP0: 로그가 찍혔는지로만 구분 가능합니다.\n");

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return all_correct ? 0 : 1;
}