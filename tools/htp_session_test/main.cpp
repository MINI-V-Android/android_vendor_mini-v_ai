#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <cstddef>

// libhtp_ops.so 4개 함수 + libcdsprpc.so rpcmem 함수들을
// 원본(ggml-htp.cc / dsprpc_interface.cc)과 동일한 방식(bare-name dlopen)으로
// 호출하되, 매 단계마다 반드시 fprintf로 남긴다. ai_daemon/llama.cpp 전체를
// 거치지 않는 순수 격리 테스트 — 이슈 #48 진단용.

int main() {
    setenv("DSP_LIBRARY_PATH", "/vendor/lib64/rfsa/adsp", 1);

    printf("[1] dlopen(\"libhtp_ops.so\") ...\n");
    void* h = dlopen("libhtp_ops.so", RTLD_LAZY | RTLD_LOCAL);
    if (!h) { printf("[1] FAILED: %s\n", dlerror()); return 1; }
    printf("[1] OK handle=%p\n", h);

    using open_session_fn   = int(int, int);
    using init_backend_fn   = void();
    using create_channel_fn = int(int, unsigned int);
    using close_session_fn  = void();

    auto open_session   = reinterpret_cast<open_session_fn*>(dlsym(h, "open_dsp_session"));
    auto init_backend    = reinterpret_cast<init_backend_fn*>(dlsym(h, "init_htp_backend"));
    auto create_channel  = reinterpret_cast<create_channel_fn*>(dlsym(h, "create_htp_message_channel"));
    auto close_session   = reinterpret_cast<close_session_fn*>(dlsym(h, "close_dsp_session"));

    printf("[2] dlsym: open=%p init=%p create=%p close=%p\n",
           (void*)open_session, (void*)init_backend, (void*)create_channel, (void*)close_session);
    if (!open_session || !init_backend || !create_channel || !close_session) {
        printf("[2] FAILED: missing symbol\n"); return 1;
    }

    printf("[3] open_dsp_session(3, 1) ...\n");
    int err = open_session(3 /*CDSP_DOMAIN_ID*/, 1 /*unsigned_pd*/);
    printf("[3] returned 0x%x (%d) — %s\n", err, err, err == 0 ? "OK" : "FAILED");
    if (err != 0) return 1;

    printf("[4] init_htp_backend() ...\n");
    init_backend();
    printf("[4] returned (void)\n");

    // ---- message channel: ggml-htp.cc::init_message_channel()과 동일 절차 ----
    printf("[5] dlopen(\"libcdsprpc.so\") ...\n");
    void* cdsp = dlopen("libcdsprpc.so", RTLD_LAZY | RTLD_LOCAL);
    if (!cdsp) { printf("[5] FAILED: %s\n", dlerror()); return 1; }

    using rpcmem_alloc_fn = void*(int, unsigned int, int);
    using rpcmem_to_fd_fn = int(void*);
    using fastrpc_mmap_fn = int(int, int, void*, int, size_t, int);

    auto rpcmem_alloc = reinterpret_cast<rpcmem_alloc_fn*>(dlsym(cdsp, "rpcmem_alloc"));
    auto rpcmem_to_fd = reinterpret_cast<rpcmem_to_fd_fn*>(dlsym(cdsp, "rpcmem_to_fd"));
    auto fastrpc_mmap = reinterpret_cast<fastrpc_mmap_fn*>(dlsym(cdsp, "fastrpc_mmap"));
    if (!rpcmem_alloc || !rpcmem_to_fd || !fastrpc_mmap) {
        printf("[5] FAILED: missing libcdsprpc symbol\n"); return 1;
    }

    void* msg = rpcmem_alloc(25 /*RPCMEM_HEAP_ID_SYSTEM*/, 0 /*UNCACHED*/, 4096);
    printf("[6] rpcmem_alloc(SYSTEM,UNCACHED,4096) -> %p\n", msg);
    if (!msg) return 1;

    int fd = rpcmem_to_fd(msg);
    printf("[7] rpcmem_to_fd -> %d\n", fd);
    if (fd < 0) return 1;

    int mmap_err = fastrpc_mmap(3 /*CDSP_DOMAIN_ID*/, fd, msg, 0, 4096, 2 /*FASTRPC_MAP_FD*/);
    printf("[8] fastrpc_mmap -> %d — %s\n", mmap_err, mmap_err == 0 ? "OK" : "FAILED");
    if (mmap_err) return 1;

    printf("[9] create_htp_message_channel(fd=%d, 4096) ...\n", fd);
    int ch_err = create_channel(fd, 4096);
    printf("[9] returned %d — %s  <== 이게 진짜 결정적 값\n", ch_err, ch_err == 0 ? "SUCCESS" : "FAILED");

    printf("[10] close_dsp_session() ...\n");
    close_session();
    printf("[10] done\n");

    return ch_err == 0 ? 0 : 1;
}