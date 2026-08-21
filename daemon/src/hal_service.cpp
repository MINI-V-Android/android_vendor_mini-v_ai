#include "hal_service.h"
#include <mutex>
#include <set>

namespace miniv::ai {

using aidl::vendor::miniv::ai::IMiniVAiHal;
using aidl::vendor::miniv::ai::IMiniVAiStreamCallback;

namespace {
// >>> MINI-V 추가 [MOD-06] (NpuLLMEngine은 세션 개념이 없는 단일 전역 엔진이라,
//     HAL 계약(세션ID 존재/중복 검증)을 지키기 위한 추적을 이 레이어에서만 함.
//     실제 추론은 sessionId와 무관하게 항상 같은 gNpuEngine을 씀.)
std::mutex gSessionMutex;
std::set<int32_t> gActiveSessions;

// NpuLLMEngine::infer()가 스레드 세이프하지 않음(mCtx 공유) — HAL 스레드풀에서
// 동시 호출이 들어와도 여기서 직렬화해 안전하게 만듦. 성능보다 정합성 우선.
std::mutex gNpuInferMutex;
// <<< MINI-V 추가 끝 [MOD-06]
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
      ok = mEngine->infer(prompt, maxTokens,
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
  // >>> MINI-V 추가 [MOD-06] (알려진 한계) NpuLLMEngine::infer()는 취소
  // 체크 지점이 없어 현재는 진짜로 취소가 안 됨. 조용히 무시하는 대신
  // 로그로 남겨서, 프레임워크 쪽이 "취소했는데 응답이 계속 온다"를
  // 디버깅할 때 이 로그로 원인을 바로 알 수 있게 함. 향후 개선 여지는
  // 아래 §안내 참고.
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