#define LOG_TAG "miniv_ai"

#include <errno.h>
#include <signal.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>

#include <android-base/logging.h>
#include <cutils/sockets.h>

namespace {

constexpr const char* kSocketName = "miniv_ai";
constexpr int kBacklog = 4;

std::atomic<bool> g_running{true};

void SignalHandler(int signum) {
    LOG(INFO) << "received signal " << signum << ", shutting down";
    g_running.store(false);
}

void HandleClient(int client_fd) {
    const char* greeting = "HELLO from miniv_ai\n";
    ssize_t n = write(client_fd, greeting, strlen(greeting));
    if (n < 0) {
        PLOG(WARNING) << "write failed";
    }
    close(client_fd);
}

}  // namespace

int main(int argc, char** argv) {
    android::base::InitLogging(argv, &android::base::KernelLogger);
    LOG(INFO) << "ai_daemon starting";

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