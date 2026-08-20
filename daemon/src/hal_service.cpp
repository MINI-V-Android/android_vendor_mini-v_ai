#include "hal_service.h"
#include <atomic>
#include <thread>
#include <unordered_map>
#include <mutex>

namespace miniv::ai {

using aidl::vendor::miniv::ai::IMiniVAiHal;
using aidl::vendor::miniv::ai::IMiniVAiStreamCallback;

namespace {
std::mutex gCancelMapMutex;
std::unordered_map<int, std::shared_ptr<std::atomic<bool>>> gCancelFlags;
}  // namespace

ndk::ScopedAStatus MiniVAiHalService::isReady(bool* _aidl_return) {
  *_aidl_return = mEngine->isReady();
  return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus MiniVAiHalService::createSession(int32_t sessionId, int32_t* _aidl_return) {
  if (!mEngine->isReady()) {
    *_aidl_return = IMiniVAiHal::CREATE_SESSION_ERR_ENGINE_NOT_READY;
    return ndk::ScopedAStatus::ok();
  }
  bool ok = mEngine->sessionManager()->createSession(sessionId);
  *_aidl_return = ok ? 0 : IMiniVAiHal::CREATE_SESSION_ERR_ALREADY_EXISTS;
  return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus MiniVAiHalService::destroySession(int32_t sessionId, int32_t* _aidl_return) {
  if (!mEngine->isReady()) {
    *_aidl_return = IMiniVAiHal::DESTROY_SESSION_ERR_NOT_FOUND;
    return ndk::ScopedAStatus::ok();
  }
  bool ok = mEngine->sessionManager()->killSession(sessionId);
  *_aidl_return = ok ? 0 : IMiniVAiHal::DESTROY_SESSION_ERR_NOT_FOUND;
  return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus MiniVAiHalService::inferStream(
    int32_t sessionId, const std::string& prompt, int32_t maxTokens,
    const std::shared_ptr<IMiniVAiStreamCallback>& callback, int32_t* _aidl_return) {

  if (!mEngine->isReady()) {
    *_aidl_return = IMiniVAiHal::INFER_ERR_ENGINE_NOT_READY;
    return ndk::ScopedAStatus::ok();
  }

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
      // 세션이 없어서 infer()가 실패한 경우. 프레임워크 쪽 관례(unknown
      // session도 SESSION_EVICTED로 보고)를 그대로 따라 이 코드를 씀.
      callback->onError(sessionId, IMiniVAiStreamCallback::ERROR_SESSION_NOT_FOUND,
                         "SESSION_NOT_FOUND");
    }
  }).detach();

  *_aidl_return = 0;
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