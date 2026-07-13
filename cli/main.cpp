#define LOG_TAG "ai_daemon_cli"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <string>

#include "base64_util.h"

namespace {

constexpr const char* kSocketPath = "/dev/socket/miniv_ai";
constexpr size_t kReadBufSize = 256;

int Connect() {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        fprintf(stderr, "socket: %s\n", strerror(errno));
        return -1;
    }

    struct sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, kSocketPath, sizeof(addr.sun_path) - 1);

    if (connect(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        fprintf(stderr, "connect to %s: %s\n", kSocketPath, strerror(errno));
        close(fd);
        return -1;
    }

    return fd;
}

int ReadAndPrint(int fd) {
    char buf[kReadBufSize];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    if (n < 0) {
        fprintf(stderr, "read: %s\n", strerror(errno));
        return 1;
    }
    if (n == 0) {
        fprintf(stderr, "peer closed connection without sending data\n");
        return 1;
    }
    buf[n] = '\0';
    fputs(buf, stdout);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    int fd = Connect();
    if (fd < 0) return 1;

    if (argc >= 2 && strcmp(argv[1], "infer") == 0) {
        std::string prompt = (argc >= 3) ? argv[2] : "";
        int maxTokens = (argc >= 4) ? atoi(argv[3]) : 64;

        dprintf(fd, "INFER %d\n%s\nEND\n", maxTokens, prompt.c_str());

        FILE* rf = fdopen(fd, "r");
        if (!rf) {
            close(fd);
            return 1;
        }

        char line[4096];
        while (fgets(line, sizeof(line), rf)) {
            if (strncmp(line, "TOKEN ", 6) == 0) {
                std::string tokenLine(line + 6);
                if (!tokenLine.empty() && tokenLine.back() == '\n') {
                    tokenLine.pop_back();
                }
                std::string decoded = base64Decode(tokenLine);
                printf("%s", decoded.c_str());
                fflush(stdout);
            } else if (strncmp(line, "DONE", 4) == 0) {
                printf("\n[DONE]\n");
                break;
            } else if (strncmp(line, "ERROR", 5) == 0) {
                printf("\n[ERROR] %s", line + 6);
                break;
            }
        }
        fclose(rf);
        return 0;
    } else {
        dprintf(fd, "HELLO\n");
        int rc = ReadAndPrint(fd);
        close(fd);
        return rc;
    }
}