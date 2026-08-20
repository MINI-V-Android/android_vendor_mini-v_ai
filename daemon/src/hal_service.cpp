#include "hal_service.h"
#include <atomic>
#include <thread>
#include <unordered_map>
#include <mutex>

#define LOG_TAG "HalService"
#include "miniv_log.h"

namespace miniv::ai {

using aidl::vendor::miniv::ai::IMiniVAiHal;
using aidl::vendor::miniv::ai::IMiniVAiStreamCallback;

namespace {
std::mutex gCancelMapMutex;
std::unordered_map<int, std::shared_ptr<std::atomic<bool>>> gCancelFlags;
}  // namespace

ndk::ScopedAStatus MiniVAiHalService::isReady(bool* _aidl_return) {
  *_aidl_return = mEngine->isReady();
  LOGI("[HAL] isReady() -> %d", *_aidl_return);
  return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus MiniVAiHalService::createSession(int32_t sessionId, int32_t* _aidl_return) {
  LOGI("[HAL] createSession ENTER session=%d", sessionId);
  if (!mEngine->isReady()) {
    *_aidl_return = IMiniVAiHal::CREATE_SESSION_ERR_ENGINE_NOT_READY;
    return ndk::ScopedAStatus::ok();
  }
  bool ok = mEngine->sessionManager()->createSession(sessionId);
  *_aidl_return = ok ? 0 : IMiniVAiHal::CREATE_SESSION_ERR_ALREADY_EXISTS;
  LOGI("[HAL] createSession EXIT session=%d ret=%d (engine not ready)", sessionId, *_aidl_return);
  return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus MiniVAiHalService::destroySession(int32_t sessionId, int32_t* _aidl_return) {
  LOGI("[HAL] destroySession ENTER session=%d", sessionId);
  if (!mEngine->isReady()) {
    *_aidl_return = IMiniVAiHal::DESTROY_SESSION_ERR_NOT_FOUND;
    LOGI("[HAL] destroySession EXIT session=%d ret=%d (engine not ready)", sessionId, *_aidl_return);
      return ndk::ScopedAStatus::ok();
  }
  bool ok = mEngine->sessionManager()->killSession(sessionId);
  *_aidl_return = ok ? 0 : IMiniVAiHal::DESTROY_SESSION_ERR_NOT_FOUND;
  LOGI("[HAL] destroySession EXIT session=%d ret=%d", sessionId, *_aidl_return);
  return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus MiniVAiHalService::inferStream(
    int32_t sessionId, const std::string& prompt, int32_t maxTokens,
    const std::shared_ptr<IMiniVAiStreamCallback>& callback, int32_t* _aidl_return) {

  int reqId = miniv_next_req_id();
  LOGI("[HAL] inferStream ENTER req=%d session=%d maxTokens=%d promptLen=%zu",
       reqId, sessionId, maxTokens, prompt.size());

  if (!mEngine->isReady()) {
    *_aidl_return = IMiniVAiHal::INFER_ERR_ENGINE_NOT_READY;
    LOGI("[HAL] inferStream EXIT req=%d ret=%d (engine not ready)", reqId, *_aidl_return);
      return ndk::ScopedAStatus::ok();
  }

  auto cancelFlag = std::make_shared<std::atomic<bool>>(false);
  {
    std::lock_guard<std::mutex> lock(gCancelMapMutex);
    gCancelFlags[sessionId] = cancelFlag;
  }

  std::thread([this, sessionId, prompt, maxTokens, callback, cancelFlag, reqId]() {
    miniv_set_req_id(reqId);
  
    bool ok = mEngine->infer(sessionId, prompt, maxTokens,
        [&](const std::string& tok) {
          LOGI("[HAL] onToken fire req=%d session=%d len=%zu", reqId, sessionId, tok.size());
                  callback->onToken(sessionId, tok);
        },
        cancelFlag.get());

    {
      std::lock_guard<std::mutex> lock(gCancelMapMutex);
      gCancelFlags.erase(sessionId);
    }

    if (ok) {
        LOGI("[HAL] inferStream THREAD DONE req=%d session=%d -> onComplete", reqId, sessionId);
          callback->onComplete(sessionId);
    } else {
        LOGI("[HAL] inferStream THREAD DONE req=%d session=%d -> onError SESSION_NOT_FOUND", reqId, sessionId);
          callback->onError(sessionId, IMiniVAiStreamCallback::ERROR_SESSION_NOT_FOUND,
                         "SESSION_NOT_FOUND");
    }
  }).detach();

  *_aidl_return = 0;
  LOGI("[HAL] inferStream EXIT req=%d ret=0 (접수만, 실제 처리는 detached thread)", reqId);
  return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus MiniVAiHalService::cancel(int32_t sessionId) {
  LOGI("[HAL] cancel session=%d", sessionId);
  std::lock_guard<std::mutex> lock(gCancelMapMutex);
  auto it = gCancelFlags.find(sessionId);
  if (it != gCancelFlags.end()) {
    it->second->store(true);
  }
  return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus MiniVAiHalService::getModelInfo(std::string* _aidl_return) {
  *_aidl_return = mEngine->getModelInfo();
  LOGI("[HAL] getModelInfo() -> %s", _aidl_return->c_str());
  return ndk::ScopedAStatus::ok();
}

}  // namespace miniv::ai