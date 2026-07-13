#define LOG_TAG "miniv_ai"

#include <errno.h>
#include <signal.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <string>

#include <android-base/logging.h>
#include <cutils/sockets.h>

#include "llm_engine.h"
#include "base64_util.h"

namespace {

constexpr const char* kSocketName = "miniv_ai";
constexpr int kBacklog = 4;

std::atomic<bool> g_running{true};
LLMEngine gEngine;

void SignalHandler(int signum) {
    LOG(INFO) << "received signal " << signum << ", shutting down";
    g_running.store(false);
}

void HandleClient(int clientFd) {
    FILE* rf = fdopen(clientFd, "r");
    if (!rf) {
        close(clientFd);
        return;
    }
    int dupFd = dup(clientFd);
    FILE* wf = nullptr;
    if (dupFd >= 0) {
        wf = fdopen(dupFd, "w");
    }
    if (!wf) {
        if (dupFd >= 0) close(dupFd);
        fclose(rf);
        return;
    }

    char line[4096];
    while (fgets(line, sizeof(line), rf)) {
        std::string cmd(line);

        if (cmd.rfind("INFER ", 0) == 0) {
            int maxTokens = atoi(cmd.c_str() + 6);

            std::string prompt;
            char promptLine[4096];
            while (fgets(promptLine, sizeof(promptLine), rf)) {
                if (strncmp(promptLine, "END", 3) == 0) break;
                prompt += promptLine;
            }
            if (!prompt.empty() && prompt.back() == '\n') prompt.pop_back();

            if (!gEngine.isReady()) {
                fprintf(wf, "ERROR engine not ready\n");
                fflush(wf);
                continue;
            }

            std::atomic<bool> cancelled(false);
            gEngine.infer(prompt, maxTokens,
                [wf](const std::string& tok) {
                    fprintf(wf, "TOKEN %s\n", base64Encode(tok).c_str());
                    fflush(wf);
                },
                cancelled);

            fprintf(wf, "DONE\n");
            fflush(wf);

        } else if (cmd.rfind("HELLO", 0) == 0) {
            fprintf(wf, "HELLO from miniv_ai\n");
            fflush(wf);
        }
    }
    fclose(wf);
    fclose(rf);
}

}  // namespace

int main(int argc, char** argv) {
    android::base::InitLogging(argv, &android::base::KernelLogger);
    LOG(INFO) << "ai_daemon starting";

    if (!gEngine.load("/data/local/tmp/model.gguf", /*nThreads=*/4, /*nCtx=*/2048)) {
        LOG(ERROR) << "model load failed, exiting";
        return 1;
    }

    signal(SIGTERM, SignalHandler);
    signal(SIGINT, SignalHandler);

    int listen_fd = android_get_control_socket(kSocketName);
    if (listen_fd < 0) {
        LOG(ERROR) << "android_get_control_socket(" << kSocketName << ") failed";
        return 1;
    }
    LOG(INFO) << "got control socket fd=" << listen_fd;

    if (listen(listen_fd, kBacklog) < 0) {
        PLOG(ERROR) << "listen failed";
        return 1;
    }
    LOG(INFO) << "listening on /dev/socket/" << kSocketName;

    while (g_running.load()) {
        int client_fd = accept(listen_fd, nullptr, nullptr);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            PLOG(WARNING) << "accept failed";
            continue;
        }
        LOG(INFO) << "client connected fd=" << client_fd;
        HandleClient(client_fd);
    }

    close(listen_fd);
    LOG(INFO) << "ai_daemon stopped";
    return 0;
}