#include "hal_service.h"
#include <atomic>
#include <thread>
#include <unordered_map>
#include <mutex>

namespace miniv::ai {

using aidl::vendor::miniv::ai::IMiniVAiStreamCallback;

namespace {
std::mutex gCancelMapMutex;
std::unordered_map<int, std::shared_ptr<std::atomic<bool>>> gCancelFlags;
}  // namespace

ndk::ScopedAStatus MiniVAiHalService::isReady(bool* _aidl_return) {
  *_aidl_return = mEngine->isReady();
  return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus MiniVAiHalService::createSession(int32_t sessionId, bool* _aidl_return) {
  *_aidl_return = mEngine->sessionManager()->createSession(sessionId);
  return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus MiniVAiHalService::destroySession(int32_t sessionId, bool* _aidl_return) {
  *_aidl_return = mEngine->sessionManager()->killSession(sessionId);
  return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus MiniVAiHalService::inferStream(
    int32_t sessionId, const std::string& prompt, int32_t maxTokens,
    const std::shared_ptr<IMiniVAiStreamCallback>& callback, bool* _aidl_return) {

  auto cancelFlag = std::make_shared<std::atomic<bool>>(false);
  {
    std::lock_guard<std::mutex> lock(gCancelMapMutex);
    gCancelFlags[sessionId] = cancelFlag;
  }

  std::thread([this, sessionId, prompt, maxTokens, callback, cancelFlag]() {
    bool ok = mEngine->infer(sessionId, prompt, maxTokens,
        [&](const std::string& tok) { callback->onToken(sessionId, tok); },
        cancelFlag.get());

    {
      std::lock_guard<std::mutex> lock(gCancelMapMutex);
      gCancelFlags.erase(sessionId);
    }

    if (ok) {
      callback->onComplete(sessionId);
    } else {
      callback->onError(sessionId, -1, "SESSION_NOT_FOUND");
    }
  }).detach();

  *_aidl_return = true;
  return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus MiniVAiHalService::cancel(int32_t sessionId) {
  std::lock_guard<std::mutex> lock(gCancelMapMutex);
  auto it = gCancelFlags.find(sessionId);
  if (it != gCancelFlags.end()) {
    it->second->store(true);
  }
  return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus MiniVAiHalService::getModelInfo(std::string* _aidl_return) {
  *_aidl_return = mEngine->getModelInfo();
  return ndk::ScopedAStatus::ok();
}

}  // namespace miniv::ai