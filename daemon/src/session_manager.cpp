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

SessionManager::~SessionManager() {
  // HOT 세션들의 llama_context 해제. COLD 세션의 스왑 파일은 남겨둠
  // (데몬 재시작 시 이어서 쓸 수 있게 — 필요 없으면 여기서 지워도 됨, 정책 미정)
  for (auto &kv : mSessions) {
    if (kv.second.state == SessionMeta::State::HOT && kv.second.ctx) {
      llama_free(kv.second.ctx);
    }
  }
}

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

bool SessionManager::createSession(int requestedId) {
  if (mSessions.count(requestedId))
    return false; // 중복 방지
  SessionMeta meta;
  meta.id = requestedId;
  meta.state = SessionMeta::State::COLD;
  meta.lastUsedMs = nowMs();
  mSessions.emplace(requestedId, std::move(meta));
  return true;
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

  evictHotIfNeeded(id);

  llama_context *ctx = llama_init_from_model(mModel, mCtxParams);
  if (!ctx)
    return nullptr;

  if (!meta.swapFilePath.empty()) {
    meta.tokens.resize(mCtxParams.n_ctx);
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
    std::remove(meta.swapFilePath.c_str());
  }

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
      break;
    if (!swapOut(mSessions.at(victim)))
      break;
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

bool SessionManager::swapOut(SessionMeta &meta) {
  std::string path = swapPathFor(meta.id);
  size_t bytes = 0;

  for (;;) {
    if (!llama_state_save_file(meta.ctx, path.c_str(), meta.tokens.data(),
                               meta.tokens.size()))
      return false;
    bytes = llama_state_get_size(meta.ctx);

    if (bytes + mColdUsedBytes <= mColdLimitBytes)
      break;
    if (meta.tokens.size() <= kMinKeepTokens)
      break; // best-effort로 수용

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
        break;
      victim = *std::next(mColdLru.begin());
    }
    killSession(victim);
  }
}

} // namespace miniv::ai
