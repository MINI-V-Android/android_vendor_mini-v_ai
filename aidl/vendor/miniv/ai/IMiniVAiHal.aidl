package vendor.miniv.ai;

import vendor.miniv.ai.IMiniVAiStreamCallback;

/**
 * 벤더 HAL 동기 반환값 에러코드 (-40x).
 * 프레임워크 IMINIVAIService의 -10x/-20x 네임스페이스와 같은 원칙(자릿수
 * 기반, 이름 있는 상수)을 따르되, system/vendor 경계를 넘는 이 인터페이스
 * 전용 자릿수를 씀. Phase 3.4에서 MINIVAIService가 이 코드를 받아
 * LLMEngine.ErrorCode(-100x)로 매핑하는 실제 브릿지를 구현하게 됨.
 */
@VintfStability
interface IMiniVAiHal {
    // createSession() 반환값
    const int CREATE_SESSION_ERR_ENGINE_NOT_READY = -401;
    const int CREATE_SESSION_ERR_ALREADY_EXISTS = -402;

    // destroySession() 반환값
    const int DESTROY_SESSION_ERR_NOT_FOUND = -403;

    // inferStream() 반환값 — "접수 가능 여부"만 동기로 알림.
    // 세션 유효성 등 실제 추론 성패는 IMiniVAiStreamCallback으로 비동기 통보.
    const int INFER_ERR_ENGINE_NOT_READY = -404;

    boolean isReady();

    // 0 = 성공, 그 외 = 위 CREATE_SESSION_ERR_* 상수
    int createSession(int sessionId);

    // 0 = 성공, 그 외 = 위 DESTROY_SESSION_ERR_* 상수
    int destroySession(int sessionId);

    // 0 = 접수됨(스트리밍 시작), 그 외 = 위 INFER_ERR_* 상수
    int inferStream(int sessionId, String prompt, int maxTokens, IMiniVAiStreamCallback callback);

    void cancel(int sessionId);
    String getModelInfo();
}