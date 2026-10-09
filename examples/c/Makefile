# C examples for the offgrid C ABI.
CC       ?= cc
CFLAGS   ?= -Wall -Wextra -O2
CPPFLAGS += -I../../crates/offgrid-ffi/include

LIBDIR   = ../../target/release
# Link dynamically against target/release, with an rpath relative to the binary.
LDFLAGS  = -L$(LIBDIR) -Wl,-rpath,'$$ORIGIN/../../target/release'
LDLIBS   = -loffgrid

MODEL       ?= ../../spike/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
EMBED_MODEL ?= ../../spike/models/Qwen3-Embedding-0.6B-Q8_0.gguf

PROGS = chat embed cancel

.PHONY: all clean run static
all: $(PROGS)

chat: chat.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

embed: embed.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

cancel: cancel.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS) -lpthread

# Statically linked chat: liboffgrid.a plus its C++/Vulkan/system dependencies.
chat-static: chat.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LIBDIR)/liboffgrid.a \
		-lstdc++ -lvulkan -lm -lpthread -ldl

static: chat-static

run: all
	./chat "$(MODEL)" "In one sentence, what is a llama?"
	./embed "$(EMBED_MODEL)"
	./cancel "$(MODEL)"

clean:
	rm -f $(PROGS) chat-static
