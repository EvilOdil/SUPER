// [px4_oa] Named ranges for Nsight Systems (NVTX). Compiled in when CMake finds the NVTX
// header (Jetson / CUDA toolkit); otherwise ROGMAP_RANGE() is nothing.
#pragma once

#ifdef ROGMAP_NVTX
#include <nvtx3/nvToolsExt.h>

namespace rog_map {
struct ScopedRange {
    explicit ScopedRange(const char *name) { nvtxRangePushA(name); }
    ~ScopedRange() { nvtxRangePop(); }
    ScopedRange(const ScopedRange &) = delete;
    ScopedRange &operator=(const ScopedRange &) = delete;
};
}  // namespace rog_map
#define ROGMAP_RANGE_CAT(a, b) a##b
#define ROGMAP_RANGE_NAME(line) ROGMAP_RANGE_CAT(rogmap_range_, line)
#define ROGMAP_RANGE(name) rog_map::ScopedRange ROGMAP_RANGE_NAME(__LINE__)(name)
#else
#define ROGMAP_RANGE(name)
#endif
