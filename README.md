# MINI-V Vendor AI Daemon & CLI

Android 기기(Qualcomm Snapdragon 8 Gen 2 / SM8550)의 Hexagon v73 HTP NPU를 활용하여 Vendor 레벨에서 온디바이스 LLM을 구동하고, Best-of-N (N=2) 병렬 생성 파이프라인 및 지연 스트리밍을 제공하는 핵심 서비스 모듈입니다.

본 프로젝트의 NPU 가속 백엔드 및 라이브러리는 **Scaling LLM** 연구 소스 및 아래 업스트림 오픈소스를 기반으로 빌드되어 동작합니다:
* **[haozixu/htp-ops-lib](https://github.com/haozixu/htp-ops-lib)**: Qualcomm Hexagon HTP 커스텀 연산자(QNN / HTP Operations) 라이브러리
* **[haozixu/llama.cpp-npu](https://github.com/haozixu/llama.cpp-npu)**: Hexagon v73 HTP NPU 가속 백엔드가 통합된 llama.cpp 구현체
* `prebuilts/` 디렉토리에 포함된 바이너리/라이브러리(`libllama-npu`, `libggml-htp-npu`, `libhtp-ops-npu` 등)는 위 소스를 타겟 기기용으로 빌드한 산출물입니다.

---

## 1. 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                      │
│   • Android Framework Apps (Java/Kotlin via AIDL Service)   │
│   • Debug / Terminal CLI (ai_daemon_cli via UNIX Socket)    │
└──────────────────────────────┬──────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            │                                     │
      [AIDL IPC]                           [UNIX Domain Socket]
 (IMiniVAiHal / Binder)                   (/dev/socket/miniv_ai)
            │                                     │
┌───────────▼─────────────────────────────────────▼───────────┐
│                    ai_daemon (Vendor Service)               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ MiniVAiHalService (AIDL Stub) / Socket HandleClient    │ │
│  └───────────────────────────┬────────────────────────────┘ │
│                              │                              │
│  ┌───────────────────────────▼────────────────────────────┐ │
│  │ NpuLLMEngine (libnpu-engine-miniv.so)                  │ │
│  │   • N=2 Best-of-N Parallel Decode Loop                 │ │
│  │   • O(1) KV Cache Sequence Branching                   │ │
│  │   • Host CPU Rule-based / Random Evaluator             │ │
│  │   • Delayed Streaming (ANR Prevention)                 │ │
│  │   • Multi-turn Session & Cold History Replay           │ │
│  └───────────────────────────┬────────────────────────────┘ │
└──────────────────────────────┼──────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            │  libllama-npu / libggml-htp-npu     │
            │  Qualcomm Hexagon DSP (v73 CDSP)    │
            └─────────────────────────────────────┘
```

---

## 2. Daemon 워크플로우 (N=2 Best-of-N 파이프라인)

Hexagon v73 NPU의 32비트 메모리 주소 한계를 고려하면서 생성 품질을 극대화하기 위해 **N=2 병렬 생성 및 지연 스트리밍(Delayed Streaming)** 파이프라인을 채택했습니다.

### 세부 동작 순서

1. **초기화 및 모델 로드 (`load`)**:
   - `DSP_LIBRARY_PATH`를 설정하고 HTP 백엔드 라이브러리(`libggml-htp-npu-prebuilt.so`)를 동적 로드합니다.
   - GGUF 양자화 모델(예: Qwen2.5-1.5B)을 NPU 컨텍스트에 999 레이어로 오프로드합니다.
   - Warmup 디코딩을 수행하여 NPU 캐시 및 HTP 세션을 활성화합니다.

2. **단일 Prefill 연산 (`seq_id = 0`)**:
   - 입력 프롬프트 전체를 1회만 NPU로 전달하여 연산(`llama_decode`)합니다.
   - 이로써 중복 연산 오버헤드를 완전히 제거합니다.

3. **KV 캐시 복제 (`llama_kv_cache_seq_cp`)**:
   - `llama_kv_cache_seq_cp(mCtx, 0, 1, -1, -1)`를 통해 Sequence 0의 KV 캐시 상태를 Sequence 1로 O(1) 복제합니다.

4. **독립 샘플러 구성 (다양성 확보)**:
   - `sampler0` (Seed 1234)과 `sampler1` (Seed 5678)을 독립적으로 초기화하여 두 시퀀스가 서로 다른 후보 답변을 생성하도록 유도합니다.

5. **N=2 병렬 디코딩 루프 (Batch Decode)**:
   - 매 스텝마다 최대 2개의 토큰을 단일 배치에 묶어 NPU로 전송합니다.
   - EOG(문장 종료) 토큰이 먼저 발생한 시퀀스는 배치에서 즉시 제외하여 NPU 부하를 최적화합니다.
   - 디코딩 중에는 앱으로 토큰을 바로 전송하지 않고 각 시퀀스의 버퍼(`tokens0`, `tokens1`)에 조용히 저장합니다.

6. **호스트 CPU 평가 (`evaluate_response`)**:
   - NPU 생성이 완료된 후, 호스트 CPU에서 `rule_evaluator.cpp`의 평가 함수를 실행하여 두 후보 텍스트 중 더 높은 점수를 받은 승자(`bestTokens`, `bestText`)를 결정합니다.

7. **KV 캐시 동기화 및 패자 시퀀스 제거**:
   - 탈락한 시퀀스의 KV 캐시는 `llama_kv_cache_seq_rm`으로 즉시 해제하고, 승자 시퀀스의 상태를 `seq_id 0`으로 정리하여 다음 대화 턴(Multi-turn)에 대비합니다.

8. **지연 스트리밍 (Delayed Streaming)**:
   - 승리한 시퀀스의 토큰 벡터를 순회하며 `onToken` 콜백(또는 소켓 패킷)을 순차 전송합니다.
   - 앱 프론트엔드는 기존 스트리밍 인터페이스 수정 없이 자연스러운 타이핑 UX를 제공받으며, ANR 우려가 해소됩니다.

---

## 3. CLI (`ai_daemon_cli`) 사용 방법

`ai_daemon_cli`는 개발 및 온디바이스 디버깅을 위해 데몬의 UNIX Domain Socket(`/dev/socket/miniv_ai`)에 직접 접속하여 동작을 검증할 수 있는 유틸리티입니다.

### 빌드 및 타겟 기기 실행

```bash
# 모듈 빌드 (Android 빌드 환경)
m ai_daemon ai_daemon_cli

# 타겟 기기 adb 쉘 접속
adb root
adb shell
```

### 1) NPU 모델 로드 (`npu_load`)
데몬이 기동될 때 기본 모델이 자동 로드되지만, 수동으로 다른 모델이나 파라미터로 변경할 때 사용합니다.

```bash
ai_daemon_cli npu_load <modelPath> <backendDir> <nCtx> <nThreads>

# 예시:
ai_daemon_cli npu_load /data/local/tmp/llama.cpp/qwen2.5-1.5b.iq4_nl+q8_0-hmx.gguf /vendor/lib64/miniv-npu 2048 4
```

### 2) NPU 질의 및 생성 (`npu_infer`)
프롬프트를 전달하여 NPU 추론 결과를 터미널로 실시간 스트리밍 출력합니다. (단일/다중 모드 옵션 지정 가능)

```bash
ai_daemon_cli npu_infer "<prompt>" <maxTokens> [multi|single]

# 예시 1: 기본 모드(Default: Multi Best-of-N)
ai_daemon_cli npu_infer "대한민국의 수도는 어디인가요?" 256

# 예시 2: Single 모드 강제 지정 (N=1, Zero-latency 즉시 스트리밍)
ai_daemon_cli npu_infer "대한민국의 수도는 어디인가요?" 256 single
```

**출력 예시:**
```text
대한민국의 수도는 서울특별시입니다. 서울은 정치, 경제, 문화의 중심지 역할을 하고 있습니다.
[DONE]
```

### 3) 디코딩 모드 동적 전환 (`set_mode` / `setprop`)
N=2 다중 디코딩(Best-of-N)과 N=1 단일 디코딩(Single)을 데몬 재시작 없이 자유롭게 전환할 수 있습니다.

**방법 A. CLI 명령어 사용:**
```bash
# 단일 디코딩(실시간 즉시 스트리밍)으로 전환
ai_daemon_cli set_mode single

# 다중 디코딩(N=2 Best-of-N 선별)으로 전환
ai_daemon_cli set_mode multi

# 시스템 프로퍼티/환경변수 자동 감지 모드로 복귀
ai_daemon_cli set_mode auto
```

**방법 B. Android 시스템 프로퍼티(`setprop`) 사용:**
```bash
# 데몬/앱 실행 중 즉시 Single 모드로 전환
adb shell setprop persist.vendor.miniv.decode_mode single

# Multi 모드로 복귀
adb shell setprop persist.vendor.miniv.decode_mode multi
```

### 4) 헬로 테스트 (`HELLO`)
인자 없이 실행하거나 `HELLO` 명령을 보내 데몬 연결 상태를 확인합니다.

```bash
ai_daemon_cli
# 출력: HELLO from miniv_ai
```

---

## 4. 소스 코드 구성

```
android_vendor_mini-v_ai/
├── Android.bp                  # 빌드 설정 (공통 defaults, prebuilts 연결)
├── README.md                   # 본 문서
├── aidl/                       # AIDL 인터페이스 정의
│   └── vendor/miniv/ai/        # IMiniVAiHal.aidl, IMiniVAiStreamCallback.aidl
├── cli/                        # 터미널 테스트 CLI 도구
│   ├── Android.bp              # ai_daemon_cli 빌드 타겟
│   └── main.cpp                # 소켓 연결, NPU_INFER / TOKEN 디코딩 처리
├── daemon/                     # AI 데몬 및 NPU 엔진 소스
│   ├── Android.bp              # libnpu-engine-miniv.so 및 ai_daemon 바이너리 빌드
│   ├── etc/                    # init.mini-v-ai.rc 데몬 서비스 실행 스크립트
│   ├── include/
│   │   ├── base64_util.h       # 소켓 통신용 Base64 인코더/디코더
│   │   ├── hal_service.h       # AIDL HAL 서비스 클래스 정의
│   │   ├── npu_llm_engine.h    # NPU LLM 엔진 헤더
│   │   └── rule_evaluator.h    # 호스트 CPU 평가기 헤더
│   ├── main.cpp                # 데몬 진입점, Binder 및 소켓 서버 구동
│   └── src/
│       ├── hal_service.cpp     # Binder IPC 스트리밍 핸들러
│       ├── npu_llm_engine.cpp  # N=2 병렬 디코딩, KV 캐시 관리, 지연 스트리밍 핵심 로직
│       └── rule_evaluator.cpp  # 룰베이스 / 랜덤 평가기 구현체
├── prebuilts/                  # llama.cpp-npu 빌드 산출물 및 HTP 헤더
└── sepolicy/                   # SELinux vendor policy 규칙
```
