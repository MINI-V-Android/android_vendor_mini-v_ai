package vendor.miniv.ai;

/**
 * onError() 콜백 코드 (-50x). 프레임워크 ILLMStreamCallback의 -30x
 * 네임스페이스와 의미상 1:1 대응되도록 설계 — Phase 3.4 매핑 작업을
 * 단순화하기 위함 (-501↔-301, -502↔-302, -503↔-303).
 */
@VintfStability
oneway interface IMiniVAiStreamCallback {
    const int ERROR_SESSION_NOT_FOUND = -501;    // framework ERROR_SESSION_EVICTED(-301)와 동일 취급
    const int ERROR_CACHE_LIMIT_EXCEEDED = -502;  // 현재 미사용 — 프루닝으로 대체됨(Vendor_Session_Design §3-1), 자리만 예약
    const int ERROR_GENERIC_FAILURE = -503;

    void onToken(int sessionId, String token);
    void onComplete(int sessionId);
    void onError(int sessionId, int code, String message);
}