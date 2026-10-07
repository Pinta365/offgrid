/*
 * chat.c — minimal streaming chat example for the aicore C ABI.
 *
 * Usage: ./chat <model.gguf> "<question>"
 *
 * Prints version/device JSON to stdout, streams the assistant's reply to
 * stdout as it is generated, and prints the final response JSON to stderr.
 */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aicore.h"

static void fail(const char *what, char *err)
{
    fprintf(stderr, "%s: %s\n", what, err ? err : "(no error message)");
    aicore_string_free(err);
    exit(1);
}

/* Escape `in` into a freshly malloc'd JSON string literal body (no quotes). */
static char *json_escape(const char *in)
{
    size_t cap = strlen(in) * 6 + 1; /* worst case: \u00XX per byte */
    char *out = malloc(cap);
    if (!out) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }

    size_t j = 0;
    for (const unsigned char *p = (const unsigned char *)in; *p; ++p) {
        unsigned char c = *p;
        switch (c) {
        case '"':  out[j++] = '\\'; out[j++] = '"';  break;
        case '\\': out[j++] = '\\'; out[j++] = '\\'; break;
        case '\n': out[j++] = '\\'; out[j++] = 'n';  break;
        case '\r': out[j++] = '\\'; out[j++] = 'r';  break;
        case '\t': out[j++] = '\\'; out[j++] = 't';  break;
        case '\b': out[j++] = '\\'; out[j++] = 'b';  break;
        case '\f': out[j++] = '\\'; out[j++] = 'f';  break;
        default:
            if (c < 0x20) {
                j += (size_t)snprintf(out + j, 7, "\\u%04x", c);
            } else {
                out[j++] = (char)c;
            }
            break;
        }
    }
    out[j] = '\0';
    return out;
}

static bool on_token(const char *piece, size_t len, void *user_data)
{
    (void)user_data;
    fwrite(piece, 1, len, stdout);
    fflush(stdout);
    return true;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <model.gguf> \"<question>\"\n", argv[0]);
        return 2;
    }

    char *version = aicore_version_json();
    if (!version) {
        fprintf(stderr, "aicore_version_json: (no error message)\n");
        return 1;
    }
    printf("version: %s\n", version);
    aicore_string_free(version);

    char *devices = aicore_devices_json();
    if (!devices) {
        fprintf(stderr, "aicore_devices_json: (no error message)\n");
        return 1;
    }
    printf("devices: %s\n", devices);
    aicore_string_free(devices);

    char *err = NULL;

    aicore_model *model = aicore_model_load(argv[1], NULL, &err);
    if (!model) {
        fail("aicore_model_load", err);
    }

    aicore_chat *chat = aicore_chat_new(model, NULL, &err);
    if (!chat) {
        fprintf(stderr, "aicore_chat_new: %s\n", err ? err : "(no error message)");
        aicore_string_free(err);
        aicore_model_free(model);
        return 1;
    }

    char *question = json_escape(argv[2]);
    const char *prefix =
        "{\"messages\":["
        "{\"role\":\"system\",\"content\":\"You are a concise, helpful assistant.\"},"
        "{\"role\":\"user\",\"content\":\"";
    const char *suffix = "\"}],\"max_tokens\":256,\"temperature\":0.7}";

    size_t req_len = strlen(prefix) + strlen(question) + strlen(suffix) + 1;
    char *request = malloc(req_len);
    if (!request) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    snprintf(request, req_len, "%s%s%s", prefix, question, suffix);
    free(question);

    printf("assistant: ");
    fflush(stdout);

    char *response = aicore_chat_complete(chat, request, on_token, NULL, &err);
    if (!response) {
        fprintf(stderr, "aicore_chat_complete: %s\n", err ? err : "(no error message)");
        aicore_string_free(err);
        free(request);
        aicore_chat_free(chat);
        aicore_model_free(model);
        return 1;
    }

    fputc('\n', stdout);
    fprintf(stderr, "response: %s\n", response);

    aicore_string_free(response);
    free(request);
    aicore_chat_free(chat);
    aicore_model_free(model);
    return 0;
}
