/*
 * embed.c — embedding example for the aicore C ABI (float buffer API).
 *
 * Usage: ./embed <embed-model.gguf>
 *
 * Embeds three fixed sentences and prints the pairwise cosine similarity.
 */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aicore.h"

static const char *SENTENCES[3] = {
    "The cat sits on the mat.",
    "A kitten is resting on a rug.",
    "Quarterly tax filing deadlines",
};

static float dot(const float *a, const float *b, size_t dim)
{
    float s = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s <embed-model.gguf>\n", argv[0]);
        return 2;
    }

    char *err = NULL;

    aicore_model *model = aicore_model_load(argv[1], NULL, &err);
    if (!model) {
        fprintf(stderr, "aicore_model_load: %s\n", err ? err : "(no error message)");
        aicore_string_free(err);
        return 1;
    }

    aicore_embedder *emb = aicore_embedder_new(model, NULL, &err);
    if (!emb) {
        fprintf(stderr, "aicore_embedder_new: %s\n", err ? err : "(no error message)");
        aicore_string_free(err);
        aicore_model_free(model);
        return 1;
    }

    int64_t dim = aicore_embedder_dim(emb);
    if (dim <= 0) {
        fprintf(stderr, "aicore_embedder_dim failed (dim=%lld)\n", (long long)dim);
        aicore_embedder_free(emb);
        aicore_model_free(model);
        return 1;
    }
    printf("dim: %lld\n", (long long)dim);

    size_t n = 3;
    float *out = malloc(n * (size_t)dim * sizeof(float));
    if (!out) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }

    int32_t rc = aicore_embed(emb, SENTENCES, n, out, n * (size_t)dim, &err);
    if (rc != 0) {
        fprintf(stderr, "aicore_embed: %s\n", err ? err : "(no error message)");
        aicore_string_free(err);
        free(out);
        aicore_embedder_free(emb);
        aicore_model_free(model);
        return 1;
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            float cos = dot(out + i * (size_t)dim, out + j * (size_t)dim, (size_t)dim);
            printf("cosine[%zu,%zu] = %.4f  (%s | %s)\n",
                   i, j, (double)cos, SENTENCES[i], SENTENCES[j]);
        }
    }

    free(out);
    aicore_embedder_free(emb);
    aicore_model_free(model);
    return 0;
}
