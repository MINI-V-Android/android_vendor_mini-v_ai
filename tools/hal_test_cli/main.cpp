#define LOG_TAG "hal_test_cli"

#include <aidl/vendor/miniv/ai/IMiniVAiHal.h>
#include <aidl/vendor/miniv/ai/BnMiniVAiStreamCallback.h>
#include <android/binder_manager.h>
#include <android/binder_process.h>

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

using aidl::vendor::miniv::ai::IMiniVAiHal;
using aidl::vendor::miniv::ai::BnMiniVAiStreamCallback;

namespace {

// inferStream()의 onToken/onComplete/onError를 받아 콘솔에 그대로 흘려주는
// 테스트용 콜백 구현체. 완료/에러 시까지 메인 스레드를 블로킹해서
// ai_daemon_cli의 동기 출력과 비슷한 사용감을 냄.
class TestCallback : public BnMiniVAiStreamCallback {
public:
  ndk::ScopedAStatus onToken(int32_t /*sessionId*/, const std::string& token) override {
    fputs(token.c_str(), stdout);
    fflush(stdout);
    return ndk::ScopedAStatus::ok();
  }
  ndk::ScopedAStatus onComplete(int32_t /*sessionId*/) override {
    printf("\n[DONE]\n");
    finish();
    return ndk::ScopedAStatus::ok();
  }
  ndk::ScopedAStatus onError(int32_t /*sessionId*/, int32_t code, const std::string& message) override {
    printf("\n[ERROR %d] %s\n", code, message.c_str());
    finish();
    return ndk::ScopedAStatus::ok();
  }
  void wait() {
    std::unique_lock<std::mutex> lk(mMutex);
    mCv.wait(lk, [this] { return mDone; });
  }
private:
  void finish() {
    std::lock_guard<std::mutex> lk(mMutex);
    mDone = true;
    mCv.notify_all();
  }
  std::mutex mMutex;
  std::condition_variable mCv;
  bool mDone = false;
};

void Usage(const char* argv0) {
  fprintf(stderr,
    "usage:\n"
    "  %s is_ready\n"
    "  %s get_model_info\n"
    "  %s create_session <id>\n"
    "  %s destroy_session <id>\n"
    "  %s infer <id> <prompt> <maxTokens>\n"
    "  %s cancel <id>\n",
    argv0, argv0, argv0, argv0, argv0, argv0);
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) { Usage(argv[0]); return 1; }

  // onToken 등 콜백은 별도 binder 스레드로 들어오므로 스레드풀 필수.
  ABinderProcess_startThreadPool();

  const std::string instance = std::string(IMiniVAiHal::descriptor) + "/default";
  ndk::SpAIBinder binder(AServiceManager_waitForService(instance.c_str()));
  if (!binder.get()) {
    fprintf(stderr, "HAL not found: %s\n", instance.c_str());
    return 1;
  }
  auto hal = IMiniVAiHal::fromBinder(binder);
  if (!hal) {
    fprintf(stderr, "fromBinder failed\n");
    return 1;
  }

  const std::string cmd = argv[1];

  if (cmd == "is_ready") {
    bool ready = false;
    auto st = hal->isReady(&ready);
    printf("isReady() -> ok=%d ready=%d\n", st.isOk(), ready);

  } else if (cmd == "get_model_info") {
    std::string info;
    auto st = hal->getModelInfo(&info);
    printf("getModelInfo() -> ok=%d info=%s\n", st.isOk(), info.c_str());

  } else if (cmd == "create_session" && argc >= 3) {
    int32_t id = atoi(argv[2]);
    int32_t ret = 0;
    auto st = hal->createSession(id, &ret);
    printf("createSession(%d) -> ok=%d ret=%d\n", id, st.isOk(), ret);

  } else if (cmd == "destroy_session" && argc >= 3) {
    int32_t id = atoi(argv[2]);
    int32_t ret = 0;
    auto st = hal->destroySession(id, &ret);
    printf("destroySession(%d) -> ok=%d ret=%d\n", id, st.isOk(), ret);

  } else if (cmd == "infer" && argc >= 5) {
    int32_t id = atoi(argv[2]);
    std::string prompt = argv[3];
    int32_t maxTokens = atoi(argv[4]);

    auto callback = ndk::SharedRefBase::make<TestCallback>();
    int32_t ret = 0;
    auto st = hal->inferStream(id, prompt, maxTokens, callback, &ret);
    printf("inferStream() 접수 -> ok=%d ret=%d\n", st.isOk(), ret);
    if (st.isOk() && ret == 0) {
      callback->wait();
    }

  } else if (cmd == "cancel" && argc >= 3) {
    int32_t id = atoi(argv[2]);
    auto st = hal->cancel(id);
    printf("cancel(%d) -> ok=%d\n", id, st.isOk());

  } else {
    Usage(argv[0]);
    return 1;
  }

  return 0;
}