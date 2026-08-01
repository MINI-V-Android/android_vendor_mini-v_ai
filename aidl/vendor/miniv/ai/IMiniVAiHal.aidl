package vendor.miniv.ai;

import vendor.miniv.ai.IMiniVAiStreamCallback;

@VintfStability
interface IMiniVAiHal {
    boolean isReady();
    boolean createSession(int sessionId);
    boolean destroySession(int sessionId);
    boolean inferStream(int sessionId, String prompt, int maxTokens, IMiniVAiStreamCallback callback);
    void cancel(int sessionId);
    String getModelInfo();
}