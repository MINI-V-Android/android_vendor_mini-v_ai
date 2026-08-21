// HAL 단독 테스트 클라이언트 — service call로는 IMiniVAiStreamCallback
// 같은 AIDL 콜백 인터페이스를 넘길 수 없어서, 콜백을 직접 구현해
// AServiceManager_waitForService로 HAL을 잡고 실제 스트리밍을 눈으로
// 확인하는 용도. 프레임워크 브릿지가 나중에 할 일을 미리 흉내낸 것.

#define LOG_TAG "hal_test_cli"

#include <aidl/vendor/miniv/ai/BnMiniVAiStreamCallback.h>
#include <aidl/vendor/miniv/ai/IMiniVAiHal.h>
#include <android-base/logging.h>
#include <android/binder_manager.h>
#include <android/binder_process.h>

#include <condition_variable>
#include <cstdio>
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

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: hal_test_cli <sessionId> <prompt> [maxTokens]\n");
    return 1;
  }
  int32_t sessionId = atoi(argv[1]);
  std::string prompt = argv[2];
  int32_t maxTokens = argc > 3 ? atoi(argv[3]) : 64;

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

  bool ready = false;
  hal->isReady(&ready);
  fprintf(stdout, "isReady() = %s\n", ready ? "true" : "false");
  if (!ready) {
    fprintf(stderr, "engine not ready — abort\n");
    return 1;
  }

  int32_t rc = 0;
  hal->createSession(sessionId, &rc);
  fprintf(stdout, "createSession(%d) = %d\n", sessionId, rc);
  if (rc != 0) {
    fprintf(stderr, "createSession failed, code=%d\n", rc);
    return 1;
  }

  auto callback = ndk::SharedRefBase::make<TestCallback>();
  rc = 0;
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
  }

  int32_t destroyRc = 0;
  hal->destroySession(sessionId, &destroyRc);
  fprintf(stdout, "destroySession(%d) = %d\n", sessionId, destroyRc);

  return gDone ? 0 : 1;
}