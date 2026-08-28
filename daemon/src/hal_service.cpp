#include "hal_service.h"
#include <android-base/logging.h>
#include <thread>
#include <mutex>
#include <set>

namespace miniv::ai {

using aidl::vendor::miniv::ai::IMiniVAiHal;
using aidl::vendor::miniv::ai::IMiniVAiStreamCallback;

namespace {
std::mutex gSessionMutex;
std::set<int32_t> gActiveSessions;

std::mutex gNpuInferMutex;
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
  std::lock_guard<std::mutex> lock(gSessionMutex);
  auto [it, inserted] = gActiveSessions.insert(sessionId);
  *_aidl_return = inserted ? 0 : IMiniVAiHal::CREATE_SESSION_ERR_ALREADY_EXISTS;
  return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus MiniVAiHalService::destroySession(int32_t sessionId, int32_t* _aidl_return) {
  std::lock_guard<std::mutex> lock(gSessionMutex);
  size_t erased = gActiveSessions.erase(sessionId);
  if (erased) {
    std::lock_guard<std::mutex> inferLock(gNpuInferMutex);
    mEngine->destroySession(sessionId);
    // <<< MINI-V 추가 끝
  }
  *_aidl_return = erased ? 0 : IMiniVAiHal::DESTROY_SESSION_ERR_NOT_FOUND;
  return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus MiniVAiHalService::inferStream(
    int32_t sessionId, const std::string& prompt, int32_t maxTokens,
    const std::shared_ptr<IMiniVAiStreamCallback>& callback, int32_t* _aidl_return) {

  if (!mEngine->isReady()) {
    *_aidl_return = IMiniVAiHal::INFER_ERR_ENGINE_NOT_READY;
    return ndk::ScopedAStatus::ok();
  }

  {
    std::lock_guard<std::mutex> lock(gSessionMutex);
    if (gActiveSessions.find(sessionId) == gActiveSessions.end()) {
      *_aidl_return = 0;  // 접수는 됨 — 실패는 onError로 비동기 통보(프레임워크 관례)
      std::thread([sessionId, callback]() {
        callback->onError(sessionId, IMiniVAiStreamCallback::ERROR_SESSION_NOT_FOUND,
                           "SESSION_NOT_FOUND");
      }).detach();
      return ndk::ScopedAStatus::ok();
    }
  }

  std::thread([this, sessionId, prompt, maxTokens, callback]() {
    bool ok;
    {
      // >>> MINI-V 추가 [MOD-06] (직렬화 — 위 gNpuInferMutex 설명 참고)
      std::lock_guard<std::mutex> inferLock(gNpuInferMutex);
      ok = mEngine->infer(sessionId, prompt, maxTokens,
          [&](const std::string& tok) { callback->onToken(sessionId, tok); });
      // <<< MINI-V 추가 끝 [MOD-06]
    }

    if (ok) {
      callback->onComplete(sessionId);
    } else {
      callback->onError(sessionId, IMiniVAiStreamCallback::ERROR_GENERIC_FAILURE,
                         "NPU_INFER_FAILED");
    }
  }).detach();

  *_aidl_return = 0;
  return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus MiniVAiHalService::cancel(int32_t sessionId) {
  LOG(WARNING) << "cancel(" << sessionId << ") requested but NpuLLMEngine "
               << "has no cancellation support yet — ignored";
  // <<< MINI-V 추가 끝 [MOD-06]
  return ndk::ScopedAStatus::ok();
}

ndk::ScopedAStatus MiniVAiHalService::getModelInfo(std::string* _aidl_return) {
  *_aidl_return = mEngine->getModelInfo();
  return ndk::ScopedAStatus::ok();
}

}  // namespace miniv::ai