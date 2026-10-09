/*
 * cancel.c — demonstrate cancelling an in-flight completion (offgrid C ABI).
 *
 * Usage: ./cancel <model.gguf>
 *
 * Starts a long completion on a worker thread, waits ~1s, calls
 * offgrid_chat_cancel, then checks that the response's finish_reason is
 * "cancelled". Exit 0 on success, 1 otherwise.
 */
#define _POSIX_C_SOURCE 200809L

#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "offgrid.h"

struct job {
    offgrid_chat *chat;
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
    j->response = offgrid_chat_complete(j->chat, request, on_token, NULL, &j->err);
    return NULL;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }

    char *err = NULL;

    offgrid_model *model = offgrid_model_load(argv[1], NULL, &err);
    if (!model) {
        fprintf(stderr, "offgrid_model_load: %s\n", err ? err : "(no error message)");
        offgrid_string_free(err);
        return 1;
    }

    offgrid_chat *chat = offgrid_chat_new(model, NULL, &err);
    if (!chat) {
        fprintf(stderr, "offgrid_chat_new: %s\n", err ? err : "(no error message)");
        offgrid_string_free(err);
        offgrid_model_free(model);
        return 1;
    }

    struct job j = { .chat = chat, .response = NULL, .err = NULL };
    pthread_t th;
    if (pthread_create(&th, NULL, run, &j) != 0) {
        fprintf(stderr, "pthread_create failed\n");
        offgrid_chat_free(chat);
        offgrid_model_free(model);
        return 1;
    }

    sleep(1);
    offgrid_chat_cancel(chat);
    pthread_join(th, NULL);

    int ok = 0;
    if (!j.response) {
        fprintf(stderr, "offgrid_chat_complete failed: %s\n", j.err ? j.err : "(no error message)");
    } else {
        fprintf(stderr, "response: %s\n", j.response);
        if (strstr(j.response, "cancelled")) {
            printf("cancel observed: finish_reason is \"cancelled\"\n");
            ok = 1;
        } else {
            fprintf(stderr, "expected finish_reason \"cancelled\"\n");
        }
    }

    offgrid_string_free(j.response);
    offgrid_string_free(j.err);
    offgrid_chat_free(chat);
    offgrid_model_free(model);
    return ok ? 0 : 1;
}
