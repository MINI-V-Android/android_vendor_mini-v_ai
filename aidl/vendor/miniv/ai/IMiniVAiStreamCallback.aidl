package vendor.miniv.ai;

@VintfStability
oneway interface IMiniVAiStreamCallback {
    void onToken(int sessionId, String token);
    void onComplete(int sessionId);
    void onError(int sessionId, int code, String message);
}