#define LOG_TAG "hal_test_cli"

#include <aidl/vendor/miniv/ai/BnMiniVAiStreamCallback.h>
#include <aidl/vendor/miniv/ai/IMiniVAiHal.h>
#include <android-base/logging.h>
#include <android/binder_manager.h>
#include <android/binder_process.h>

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

using aidl::vendor::miniv::ai::BnMiniVAiStreamCallback;
using aidl::vendor::miniv::ai::IMiniVAiHal;

namespace {

std::mutex gDoneMutex;
std::condition_variable gDoneCv;
bool gDone = false;

class TestCallback : public BnMiniVAiStreamCallback {
public:
  ndk::ScopedAStatus onToken(int32_t sessionId, const std::string& token) override {
    fprintf(stdout, "%s", token.c_str());
    fflush(stdout);
    return ndk::ScopedAStatus::ok();
  }

  ndk::ScopedAStatus onComplete(int32_t sessionId) override {
    fprintf(stdout, "\n[DONE session=%d]\n", sessionId);
    signalDone();
    return ndk::ScopedAStatus::ok();
  }

  ndk::ScopedAStatus onError(int32_t sessionId, int32_t code,
                              const std::string& message) override {
    fprintf(stdout, "\n[ERROR session=%d code=%d msg=%s]\n", sessionId, code,
            message.c_str());
    signalDone();
    return ndk::ScopedAStatus::ok();
  }

private:
  void signalDone() {
    std::lock_guard<std::mutex> lock(gDoneMutex);
    gDone = true;
    gDoneCv.notify_one();
  }
};

void printUsage() {
  fprintf(stderr,
          "usage:\n"
          "  hal_test_cli is_ready\n"
          "  hal_test_cli get_model_info\n"
          "  hal_test_cli create_session <id>\n"
          "  hal_test_cli infer <id> <prompt> [maxTokens]\n"
          "  hal_test_cli destroy_session <id>\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    printUsage();
    return 1;
  }
  const std::string cmd = argv[1];

  const std::string instance =
      std::string(IMiniVAiHal::descriptor) + "/default";

  fprintf(stdout, "waiting for %s ...\n", instance.c_str());
  ndk::SpAIBinder binder(AServiceManager_waitForService(instance.c_str()));
  if (!binder.get()) {
    fprintf(stderr, "FAILED: service not found\n");
    return 1;
  }
  auto hal = IMiniVAiHal::fromBinder(binder);
  if (!hal) {
    fprintf(stderr, "FAILED: fromBinder returned null\n");
    return 1;
  }

  // 콜백(oneway)을 이 프로세스가 받으려면 binder 스레드풀이 떠 있어야 함.
  ABinderProcess_startThreadPool();

  if (cmd == "is_ready") {
    bool ready = false;
    hal->isReady(&ready);
    fprintf(stdout, "isReady() = %s\n", ready ? "true" : "false");
    return ready ? 0 : 1;
  }

  if (cmd == "get_model_info") {
    std::string info;
    hal->getModelInfo(&info);
    fprintf(stdout, "%s\n", info.c_str());
    return 0;
  }

  if (cmd == "create_session") {
    if (argc < 3) {
      printUsage();
      return 1;
    }
    int32_t sessionId = atoi(argv[2]);
    int32_t rc = 0;
    hal->createSession(sessionId, &rc);
    fprintf(stdout, "createSession(%d) = %d\n", sessionId, rc);
    return rc == 0 ? 0 : 1;
  }

  if (cmd == "destroy_session") {
    if (argc < 3) {
      printUsage();
      return 1;
    }
    int32_t sessionId = atoi(argv[2]);
    int32_t rc = 0;
    hal->destroySession(sessionId, &rc);
    fprintf(stdout, "destroySession(%d) = %d\n", sessionId, rc);
    return rc == 0 ? 0 : 1;
  }

  if (cmd == "infer") {
    if (argc < 4) {
      printUsage();
      return 1;
    }
    int32_t sessionId = atoi(argv[2]);
    std::string prompt = argv[3];
    int32_t maxTokens = argc > 4 ? atoi(argv[4]) : 64;

    bool ready = false;
    hal->isReady(&ready);
    fprintf(stdout, "isReady() = %s\n", ready ? "true" : "false");
    if (!ready) {
      fprintf(stderr, "engine not ready — abort\n");
      return 1;
    }

    auto callback = ndk::SharedRefBase::make<TestCallback>();
    int32_t rc = 0;
    hal->inferStream(sessionId, prompt, maxTokens, callback, &rc);
    fprintf(stdout, "inferStream() accepted = %d\n", rc);
    if (rc != 0) {
      fprintf(stderr, "inferStream failed, code=%d\n", rc);
      return 1;
    }

    {
      std::unique_lock<std::mutex> lock(gDoneMutex);
      gDoneCv.wait_for(lock, std::chrono::seconds(60), [] { return gDone; });
    }
    if (!gDone) {
      fprintf(stderr, "TIMEOUT waiting for onComplete/onError\n");
      return 1;
    }
    // ★ 이전 버전과 달리 여기서 destroySession을 자동으로 호출하지
    // 않습니다 — 세션을 살려둬야 다음 infer 호출에서 핫 연속/콜드 복원을
    // 테스트할 수 있습니다. 다 끝나면 destroy_session을 따로 불러주세요.
    return 0;
  }

  fprintf(stderr, "unknown command: %s\n", cmd.c_str());
  printUsage();
  return 1;
}