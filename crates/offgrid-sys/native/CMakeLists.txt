# With the `image` feature: llama.cpp, then stable-diffusion.cpp built against llama.cpp's ggml
# (sd.cpp reuses an existing `ggml` target), so the process has a single ggml.
cmake_minimum_required(VERSION 3.14)
project(offgrid_native C CXX)

# sd.cpp needs longer tensor names; the define changes ggml_tensor's layout, so everything that
# shares this ggml must be built with it.
add_compile_definitions(GGML_MAX_NAME=160)

add_subdirectory(${OFFGRID_LLAMA_DIR} llama.cpp)
add_subdirectory(${OFFGRID_SD_DIR} stable-diffusion.cpp)
