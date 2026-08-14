#define LOG_TAG "miniv_ai"

#include <errno.h>
#include <signal.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <string>

#include <android-base/logging.h>
#include <cutils/sockets.h>

#include "base64_util.h"
#include "llm_engine.h"
#include "npu_llm_engine.h"
#include "hal_service.h"

#include <android/binder_ibinder.h>
#include <android/binder_manager.h>
#include <android/binder_process.h>
#include <memory>

namespace {

constexpr const char *kSocketName = "miniv_ai";
constexpr int kBacklog = 4;

std::atomic<bool> g_running{true};
miniv::ai::LLMEngine gEngine;
miniv::ai::NpuLLMEngine gNpuEngine;

void SignalHandler(int signum) {
  LOG(INFO) << "received signal " << signum << ", shutting down";
  g_running.store(false);
}

void HandleClient(int clientFd) {
  FILE *rf = fdopen(clientFd, "r");
  if (!rf) {
    close(clientFd);
    return;
  }
  int dupFd = dup(clientFd);
  FILE *wf = nullptr;
  if (dupFd >= 0) {
    wf = fdopen(dupFd, "w");
  }
  if (!wf) {
    if (dupFd >= 0)
      close(dupFd);
    fclose(rf);
    return;
  }

  char line[4096];
  while (fgets(line, sizeof(line), rf)) {
    std::string cmd(line);

    if (cmd.rfind("CREATE_SESSION ", 0) == 0) {
      int id = atoi(cmd.c_str() + 15);
      bool ok = gEngine.sessionManager()->createSession(id);
      fprintf(wf, ok ? "OK\n" : "ERROR ALREADY_EXISTS\n");
      fflush(wf);

    } else if (cmd.rfind("KILL_SESSION ", 0) == 0) {
      int id = atoi(cmd.c_str() + 13);
      bool ok = gEngine.sessionManager()->killSession(id);
      fprintf(wf, ok ? "OK\n" : "ERROR NOT_FOUND\n");
      fflush(wf);

    } else if (cmd.rfind("SET_HOT_LIMIT ", 0) == 0) {
      size_t count =
          static_cast<size_t>(strtoul(cmd.c_str() + 14, nullptr, 10));
      gEngine.sessionManager()->setHotLimit(count);
      fprintf(wf, "OK\n");
      fflush(wf);

    } else if (cmd.rfind("SET_COLD_LIMIT ", 0) == 0) {
      size_t bytes =
          static_cast<size_t>(strtoull(cmd.c_str() + 15, nullptr, 10));
      gEngine.sessionManager()->setColdLimit(bytes);
      fprintf(wf, "OK\n");
      fflush(wf);

    } else if (cmd.rfind("INFER ", 0) == 0) {
      int sessionId = 0, maxTokens = 0;
      sscanf(cmd.c_str() + 6, "%d %d", &sessionId, &maxTokens);

      std::string prompt;
      char promptLine[4096];
      while (fgets(promptLine, sizeof(promptLine), rf)) {
        if (strncmp(promptLine, "END", 3) == 0)
          break;
        prompt += promptLine;
      }
      if (!prompt.empty() && prompt.back() == '\n')
        prompt.pop_back();

      if (!gEngine.isReady()) {
        fprintf(wf, "ERROR engine not ready\n");
        fflush(wf);
        continue;
      }
      bool ok = gEngine.infer(
          sessionId, prompt, maxTokens, [wf](const std::string &tok) {
            fprintf(wf, "TOKEN %s\n", base64Encode(tok).c_str());
            fflush(wf);
          });

      fprintf(wf, ok ? "DONE\n" : "ERROR SESSION_NOT_FOUND\n");
      fflush(wf);

    } else if (cmd.rfind("NPU_LOAD ", 0) == 0) {
      // NPU_LOAD <modelPath> <backendDir> <nCtx> <nThreads>
      char modelPath[512] = {0};
      char backendDir[512] = {0};
      int nCtx = 0, nThreads = 0;
      sscanf(cmd.c_str() + 9, "%511s %511s %d %d", modelPath, backendDir,
             &nCtx, &nThreads);

      bool ok = gNpuEngine.load(modelPath, backendDir, nCtx, nThreads);
      fprintf(wf, ok ? "OK\n" : "ERROR NPU_LOAD_FAILED\n");
      fflush(wf);

    } else if (cmd.rfind("NPU_INFER ", 0) == 0) {
      // NPU_INFER <maxTokens>\n<prompt>\nEND\n  (세션 없음, 단발 질문)
      int maxTokens = atoi(cmd.c_str() + 10);

      std::string prompt;
      char promptLine[4096];
      while (fgets(promptLine, sizeof(promptLine), rf)) {
        if (strncmp(promptLine, "END", 3) == 0)
          break;
        prompt += promptLine;
      }
      if (!prompt.empty() && prompt.back() == '\n')
        prompt.pop_back();

      if (!gNpuEngine.isReady()) {
        fprintf(wf, "ERROR NPU_NOT_READY\n");
        fflush(wf);
        continue;
      }

      bool ok = gNpuEngine.infer(
          prompt, maxTokens, [wf](const std::string &tok) {
            fprintf(wf, "TOKEN %s\n", base64Encode(tok).c_str());
            fflush(wf);
          });

      fprintf(wf, ok ? "DONE\n" : "ERROR NPU_INFER_FAILED\n");
      fflush(wf);

    } else if (cmd.rfind("HELLO", 0) == 0) {
      fprintf(wf, "HELLO from miniv_ai\n");
      fflush(wf);
    }
  }
  fclose(wf);
  fclose(rf);
}

} // namespace

int main(int argc, char **argv) {
  android::base::InitLogging(argv, &android::base::KernelLogger);
  LOG(INFO) << "ai_daemon starting";

  // 세션 스왑 디렉토리 사전 생성 (SessionManager가 없다고 가정하고 씀)
  mkdir("/data/vendor/miniv_ai", 0700);
  mkdir("/data/vendor/miniv_ai/sessions", 0700);

  if (!gEngine.load("/data/local/tmp/model.gguf", /*nCtx=*/2048,
                    /*nThreads=*/4)) {
    LOG(ERROR) << "model load failed, exiting";
    // return 1;
  }

  // ── HAL 등록 (신규, §11-c) ──────────────────────────────────
  // UDS accept 루프는 계속 메인 스레드 blocking으로 돌고, HAL은
  // libbinder의 별도 스레드풀에서 처리되므로 서로 간섭하지 않음.
  ABinderProcess_setThreadPoolMaxThreadCount(4);

  auto halService =
      ndk::SharedRefBase::make<miniv::ai::MiniVAiHalService>(&gEngine);
  const std::string halInstance =
      std::string(miniv::ai::MiniVAiHalService::descriptor) + "/default";

  binder_status_t halStatus = AServiceManager_addService(
      halService->asBinder().get(), halInstance.c_str());
  if (halStatus != STATUS_OK) {
    // §6-9(service_contexts 미등록) 때문에 지금은 실패가 예상되는 지점.
    // permissive 상태라 addService 자체는 통과하고 denial 로그만 찍힐 수도
    // 있고, enforcing 전환 후엔 진짜로 막힘. 어느 쪽이든 프로세스는 계속
    // 살려서 UDS 디버깅 경로(ai_daemon_cli)는 항상 쓸 수 있게 둠.
    LOG(ERROR) << "AServiceManager_addService failed for " << halInstance
               << ": " << halStatus;
  } else {
    LOG(INFO) << "HAL registered: " << halInstance;
  }

  ABinderProcess_startThreadPool();
  // ─────────────────────────────────────────────────────────────

  signal(SIGTERM, SignalHandler);
  signal(SIGINT, SignalHandler);

  int listen_fd = android_get_control_socket(kSocketName);
  if (listen_fd < 0) {
    LOG(ERROR) << "android_get_control_socket(" << kSocketName << ") failed";
    return 1;
  }
  LOG(INFO) << "got control socket fd=" << listen_fd;

  if (listen(listen_fd, kBacklog) < 0) {
    PLOG(ERROR) << "listen failed";
    return 1;
  }
  LOG(INFO) << "listening on /dev/socket/" << kSocketName;

  while (g_running.load()) {
    int client_fd = accept(listen_fd, nullptr, nullptr);
    if (client_fd < 0) {
      if (errno == EINTR)
        continue;
      PLOG(WARNING) << "accept failed";
      continue;
    }
    LOG(INFO) << "client connected fd=" << client_fd;
    HandleClient(client_fd);
  }

  close(listen_fd);
  LOG(INFO) << "ai_daemon stopped";
  return 0;
}