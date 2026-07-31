#include "session_manager.h"
#include <algorithm>
#include <chrono>
#include <cstdio>

namespace miniv::ai {
namespace {
int64_t nowMs() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(steady_clock::now().time_since_epoch())
      .count();
}
} // namespace

SessionManager::SessionManager(llama_model *model,
                               llama_context_params ctxParams,
                               std::string swapDir)
    : mModel(model), mCtxParams(ctxParams), mSwapDir(std::move(swapDir)) {}

void SessionManager::setHotLimit(size_t count) {
  mHotLimit = std::max<size_t>(1, count);
  evictHotIfNeeded(-1);
}

void SessionManager::setColdLimit(size_t bytes) {
  mColdLimitBytes = bytes;
  evictColdIfNeeded(-1);
}

std::string SessionManager::swapPathFor(int id) const {
  return mSwapDir + "/" + std::to_string(id) + ".state";
}

int SessionManager::createSession() {
  int id = mNextId++;
  SessionMeta meta;
  meta.id = id;
  meta.state = SessionMeta::State::COLD; // ctx도 파일도 없는 초기 상태
  meta.lastUsedMs = nowMs();
  mSessions.emplace(id, std::move(meta));
  // 파일이 없으므로 mColdLru에는 아직 안 넣음 (evict 대상이 아님)
  return id;
}

bool SessionManager::killSession(int id) {
  auto it = mSessions.find(id);
  if (it == mSessions.end())
    return false;
  SessionMeta &meta = it->second;

  if (meta.state == SessionMeta::State::HOT) {
    mHotLru.erase(meta.lruIt);
    llama_free(meta.ctx);
  } else if (!meta.swapFilePath.empty()) {
    mColdLru.erase(meta.lruIt);
    mColdUsedBytes -= meta.swapFileBytes;
    std::remove(meta.swapFilePath.c_str());
  }
  mSessions.erase(it);
  return true;
}

llama_context *SessionManager::activateForInfer(int id) {
  auto it = mSessions.find(id);
  if (it == mSessions.end())
    return nullptr;
  SessionMeta &meta = it->second;

  if (meta.state == SessionMeta::State::HOT) {
    touchHot(meta);
    return meta.ctx;
  }

  // COLD -> HOT
  evictHotIfNeeded(id); // id는 아직 HOT이 아니므로 자기 자신이 뽑힐 일은 없음

  llama_context *ctx = llama_init_from_model(mModel, mCtxParams);
  if (!ctx)
    return nullptr;

  if (!meta.swapFilePath.empty()) {
    // 이전에 스왑아웃된 세션 -> 복원
    meta.tokens.resize(
        mCtxParams.n_ctx); // 용량 상한 추정치, 아래서 실제 길이로 축소
    size_t nTokensOut = 0;
    bool ok = llama_state_load_file(ctx, meta.swapFilePath.c_str(),
                                    meta.tokens.data(), meta.tokens.size(),
                                    &nTokensOut);
    if (!ok) {
      llama_free(ctx);
      return nullptr;
    }
    meta.tokens.resize(nTokensOut);

    mColdLru.erase(meta.lruIt);
    mColdUsedBytes -= meta.swapFileBytes;
    meta.swapFileBytes = 0;
    std::remove(meta.swapFilePath.c_str()); // 다음 스왑아웃 시 새로 씀
  }
  // else: 처음 활성화되는 세션 -> 빈 컨텍스트로 시작

  meta.ctx = ctx;
  meta.state = SessionMeta::State::HOT;
  mHotLru.push_back(id);
  meta.lruIt = std::prev(mHotLru.end());
  meta.lastUsedMs = nowMs();
  return ctx;
}

void SessionManager::recordTokens(int id, const llama_token *newTokens,
                                  size_t n) {
  auto it = mSessions.find(id);
  if (it == mSessions.end())
    return;
  SessionMeta &meta = it->second;
  meta.tokens.insert(meta.tokens.end(), newTokens, newTokens + n);
  if (meta.state == SessionMeta::State::HOT)
    touchHot(meta);
}

void SessionManager::touchHot(SessionMeta &meta) {
  mHotLru.erase(meta.lruIt);
  mHotLru.push_back(meta.id);
  meta.lruIt = std::prev(mHotLru.end());
  meta.lastUsedMs = nowMs();
}

void SessionManager::evictHotIfNeeded(int excludeId) {
  while (mHotLru.size() >= mHotLimit && !mHotLru.empty()) {
    int victim = -1;
    for (int candidate : mHotLru) {
      if (candidate != excludeId) {
        victim = candidate;
        break;
      }
    }
    if (victim == -1)
      break; // excludeId 하나만 남음 -> 더 못 비움
    if (!swapOut(mSessions.at(victim)))
      break;
  }
}

bool SessionManager::swapOut(SessionMeta &meta) {
  std::string path = swapPathFor(meta.id);
  size_t bytes = 0;

  for (;;) {
    if (!llama_state_save_file(meta.ctx, path.c_str(), meta.tokens.data(),
                               meta.tokens.size()))
      return false;
    bytes = llama_state_get_size(meta.ctx);

    // 이 세션 혼자 넣었을 때 COLD 총량이 한도를 넘는지 확인
    if (bytes + mColdUsedBytes <= mColdLimitBytes)
      break;
    if (meta.tokens.size() <= kMinKeepTokens)
      break; // 최소치까지 잘랐으면 더는 못 줄임 -> best-effort로 수용

    size_t toRemove = std::max<size_t>(1, meta.tokens.size() / 4);
    toRemove = std::min(toRemove, meta.tokens.size() - kMinKeepTokens);
    pruneOldestTokens(meta, toRemove);
  }

  mHotLru.erase(meta.lruIt);
  llama_free(meta.ctx);
  meta.ctx = nullptr;
  meta.state = SessionMeta::State::COLD;
  meta.swapFilePath = path;
  meta.swapFileBytes = bytes;

  mColdLru.push_back(meta.id);
  meta.lruIt = std::prev(mColdLru.end());
  mColdUsedBytes += bytes;

  evictColdIfNeeded(meta.id);
  return true;
}

void SessionManager::evictColdIfNeeded(int excludeId) {
  while (mColdUsedBytes > mColdLimitBytes && !mColdLru.empty()) {
    int victim = mColdLru.front();
    if (victim == excludeId) {
      if (mColdLru.size() == 1)
        break; // 얘밖에 없으면 못 지움
      victim = *std::next(mColdLru.begin());
    }
    killSession(victim); // COLD보다 더 차가운 단계가 없으므로 완전 소멸
  }
}

void SessionManager::pruneOldestTokens(SessionMeta &meta,
                                       size_t nTokensToRemove) {
  if (nTokensToRemove == 0 || nTokensToRemove >= meta.tokens.size())
    return;

  llama_memory_t mem = llama_get_memory(meta.ctx);
  llama_memory_seq_rm(mem, /*seq_id=*/0, 0, (llama_pos)nTokensToRemove);
  llama_memory_seq_add(mem, /*seq_id=*/0, (llama_pos)nTokensToRemove, -1,
                       -(llama_pos)nTokensToRemove);

  meta.tokens.erase(meta.tokens.begin(), meta.tokens.begin() + nTokensToRemove);
}

} // namespace miniv::ai
