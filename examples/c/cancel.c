/*
 * cancel.c — demonstrate cancelling an in-flight completion (aicore C ABI).
 *
 * Usage: ./cancel <model.gguf>
 *
 * Starts a long completion on a worker thread, waits ~1s, calls
 * aicore_chat_cancel, then checks that the response's finish_reason is
 * "cancelled". Exit 0 on success, 1 otherwise.
 */
#define _POSIX_C_SOURCE 200809L

#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "aicore.h"

struct job {
    aicore_chat *chat;
    char *response;
    char *err;
};

static bool on_token(const char *piece, size_t len, void *user_data)
{
    (void)piece;
    (void)len;
    (void)user_data;
    return true;
}

static void *run(void *arg)
{
    struct job *j = (struct job *)arg;
    const char *request =
        "{\"messages\":[{\"role\":\"user\",\"content\":\"Write a 2000 word essay about rivers\"}],"
        "\"max_tokens\":2000}";
    j->response = aicore_chat_complete(j->chat, request, on_token, NULL, &j->err);
    return NULL;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }

    char *err = NULL;

    aicore_model *model = aicore_model_load(argv[1], NULL, &err);
    if (!model) {
        fprintf(stderr, "aicore_model_load: %s\n", err ? err : "(no error message)");
        aicore_string_free(err);
        return 1;
    }

    aicore_chat *chat = aicore_chat_new(model, NULL, &err);
    if (!chat) {
        fprintf(stderr, "aicore_chat_new: %s\n", err ? err : "(no error message)");
        aicore_string_free(err);
        aicore_model_free(model);
        return 1;
    }

    struct job j = { .chat = chat, .response = NULL, .err = NULL };
    pthread_t th;
    if (pthread_create(&th, NULL, run, &j) != 0) {
        fprintf(stderr, "pthread_create failed\n");
        aicore_chat_free(chat);
        aicore_model_free(model);
        return 1;
    }

    sleep(1);
    aicore_chat_cancel(chat);
    pthread_join(th, NULL);

    int ok = 0;
    if (!j.response) {
        fprintf(stderr, "aicore_chat_complete failed: %s\n", j.err ? j.err : "(no error message)");
    } else {
        fprintf(stderr, "response: %s\n", j.response);
        if (strstr(j.response, "cancelled")) {
            printf("cancel observed: finish_reason is \"cancelled\"\n");
            ok = 1;
        } else {
            fprintf(stderr, "expected finish_reason \"cancelled\"\n");
        }
    }

    aicore_string_free(j.response);
    aicore_string_free(j.err);
    aicore_chat_free(chat);
    aicore_model_free(model);
    return ok ? 0 : 1;
}
