// [px4_oa] CUDA backend for ROG-Map's probabilistic update. See rog_map/cuda/raycast_backend.h
// and docs/cuda_rogmap_plan.md.
//
// Parallel structure
//   kernel 1 (traceRays): one thread per ray. Each cell on the ray gets one atomicAdd on a packed
//     per-cell counter (low 16 bits: touches, high 16: hits). The thread whose atomicAdd finds
//     the counter empty owns the cell's entry in the touched list (the CPU code's queue).
//   kernel 2 (applyCells): one thread per touched cell. Each cell appears once in the list, so
//     each thread owns its cell: no atomics on the map. A cell that changes type adds its change
//     to its counter-map (inflation) cell with one atomicAdd.
//   kernel 3 (gatherDeltas): one thread per counter-map cell; lists the cells with a change for
//     the CPU, which keeps the inflation logic.
//   All run on one stream, in order.
// Shared writes go through atomics only; results are integer sums and per-cell updates, so the
// map is identical to the CPU path's whatever the thread order. Compiled with -fmad=false, and
// the CPU path with -ffp-contract=off, so the floating-point steps round identically too.
#include "rog_map/cuda/raycast_backend.h"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdio>
#include <stdexcept>

namespace rog_map {
namespace cuda {
namespace {

// Touched cells per apply(). A frame of ~18,000 rays touches ~2.5 M cells.
// Past this the touched list is abandoned and applyDense scans every cell instead.
constexpr uint32_t kListCap = 8u << 20;
constexpr int kMaxRays = 1 << 18;
constexpr uint32_t kDeltaCap = 4u << 20;  // counter-map cells changed per apply()

// List lengths. Device memory, not pinned host memory: every first touch of a cell does an
// atomicAdd here, and the GPU does not cache pinned host memory. Copied to the host (8 bytes)
// after each wait.
struct Counts {
    uint32_t touched;
    uint32_t deltas;
};

#define ROGMAP_CUDA_CHECK(call)                                                                  \
    do {                                                                                         \
        const cudaError_t err_ = (call);                                                         \
        if (err_ != cudaSuccess) {                                                               \
            throw std::runtime_error(std::string("rog_map CUDA: ") + #call + ": " +              \
                                     cudaGetErrorString(err_));                                  \
        }                                                                                        \
    } while (0)

__device__ __forceinline__ int cellOf(const double d, const double resolution) {
    return static_cast<int>(floor(d / resolution));  // RayCaster::posToIndex (ORIGIN_AT_CORNER)
}

// SlidingMap::globalIndexToLocalIndex for one axis: the cell's slot in the circular buffer,
// in [-half, half].
__device__ __forceinline__ int localOf(const int id, const int size, const int half) {
    int l = id % size;
    if (l > half) {
        l -= size;
    } else if (l < -half) {
        l += size;
    }
    return l;
}

// SlidingMap::getLocalIndexHash
__device__ __forceinline__ uint32_t hashOf(const int l[3], const Geometry &g) {
    return static_cast<uint32_t>(((l[0] + g.half_map_size[0]) * g.map_size[1] + l[1] + g.half_map_size[1]) *
                                 g.map_size[2] + l[2] + g.half_map_size[2]);
}

__device__ __forceinline__ void touch(uint32_t *counter, uint32_t *touched, Counts *counts, const uint32_t cell,
                                      const uint32_t inc) {
    const uint32_t before = atomicAdd(&counter[cell], inc);
    if ((before & 0xffffu) == 0) {  // first touch since the last apply: this thread lists the cell
        const uint32_t slot = atomicAdd(&counts->touched, 1u);
        if (slot < kListCap) {
            touched[slot] = cell;
        }
    }
}

__global__ void traceRays(const Ray *rays, const int n, const Geometry g, uint32_t *counter, uint32_t *touched,
                          Counts *counts) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const Ray r = rays[i];
    if (r.hit_cell >= 0) {
        touch(counter, touched, counts, static_cast<uint32_t>(r.hit_cell), 0x10001u);
    }

    // RayCaster::setInput + step, cell by cell from the start cell up to (not including) the end cell
    int cur[3], end[3], dir[3], loc[3];
    double t_step[3], t_bound[3], dis[3];
    bool moves = false;
    for (int a = 0; a < 3; ++a) {
        cur[a] = cellOf(r.start[a], g.resolution);
        end[a] = cellOf(r.end[a], g.resolution);
        dir[a] = (end[a] > cur[a]) - (end[a] < cur[a]);
        moves |= dir[a] != 0;
        dis[a] = fabs(r.end[a] - r.start[a]);
    }
    if (!moves) {
        return;  // start and end in one cell: nothing between them
    }
    const double t_max = sqrt(dis[0] * dis[0] + dis[1] * dis[1] + dis[2] * dis[2]);
    for (int a = 0; a < 3; ++a) {
        dis[a] /= t_max;
        loc[a] = localOf(cur[a], g.map_size[a], g.half_map_size[a]);
        t_step[a] = dir[a] == 0 ? DBL_MAX : fabs(g.resolution / dis[a]);
        const double next_bound = (cur[a] + 0.5) * g.resolution + dir[a] * g.resolution * 0.5;
        t_bound[a] = dir[a] == 0 ? DBL_MAX : fabs(next_bound - r.start[a]) / dis[a];
    }
    while (!(cur[0] == end[0] && cur[1] == end[1] && cur[2] == end[2])) {
        // SlidingMap::insideLocalMap
        if (abs(cur[0] - g.origin[0]) > g.half_map_size[0] || abs(cur[1] - g.origin[1]) > g.half_map_size[1] ||
            abs(cur[2] - g.origin[2]) > g.half_map_size[2]) {
            return;
        }
        touch(counter, touched, counts, hashOf(loc, g), 1u);
        const int a = t_bound[0] < t_bound[1] ? (t_bound[0] < t_bound[2] ? 0 : 2) : (t_bound[1] < t_bound[2] ? 1 : 2);
        cur[a] += dir[a];
        t_bound[a] += t_step[a];
        // step the circular-buffer slot with the cell: no division per step
        loc[a] += dir[a];
        if (loc[a] > g.half_map_size[a]) {
            loc[a] -= g.map_size[a];
        } else if (loc[a] < -g.half_map_size[a]) {
            loc[a] += g.map_size[a];
        }
    }
}

enum : uint32_t { kUnknown = 0, kKnownFree = 1, kOccupied = 2 };

__device__ __forceinline__ uint32_t typeOf(const float l, const Thresholds &t) {
    return l >= t.l_occ ? kOccupied : (l < t.l_free ? kKnownFree : kUnknown);
}

// The counter-map cell holding probability-map cell `cell`: SlidingMap::hashIdToPos on the
// probability map, then posToGlobalIndex + getHashIndexFromGlobalIndex on the counter map.
__device__ __forceinline__ uint32_t counterCellOf(const uint32_t cell, const Geometry &m, const Geometry &c) {
    int l[3], cl[3];
    l[0] = static_cast<int>(cell / (m.map_size[1] * m.map_size[2]));
    l[1] = static_cast<int>((cell - l[0] * m.map_size[1] * m.map_size[2]) / m.map_size[2]);
    l[2] = static_cast<int>(cell - l[0] * m.map_size[1] * m.map_size[2] - l[1] * m.map_size[2]);
    for (int a = 0; a < 3; ++a) {
        l[a] -= m.half_map_size[a];
        // SlidingMap::localIndexToGlobalIndex
        const int min_g = -m.half_map_size[a] + m.origin[a];
        const int min_l = localOf(min_g, m.map_size[a], m.half_map_size[a]);
        int d = l[a] - min_l;
        d = d < 0 ? m.map_size[a] + d : d;
        const double pos = (d + min_g + 0.5) * m.resolution;
        cl[a] = localOf(static_cast<int>(floor(pos * c.resolution_inv)), c.map_size[a], c.half_map_size[a]);
    }
    return hashOf(cl, c);
}

// ProbMap::hitPointUpdate / missPointUpdate for one cell; zeroes its counter.
__device__ __forceinline__ void applyCell(const uint32_t cell, const Thresholds &t, const Geometry &m,
                                          const Geometry &c, float *occupancy, uint32_t *counter, int *delta) {
    const uint32_t packed = counter[cell];
    counter[cell] = 0;
    const int touches = static_cast<int>(packed & 0xffffu), hits = static_cast<int>(packed >> 16);
    float l = occupancy[cell];
    const uint32_t from = typeOf(l, t);
    if (hits > 0) {
        l += t.l_hit * hits;
        if (l > t.l_max) {
            l = t.l_max;
        }
    } else {
        l += t.l_miss * (touches - hits);
        if (l < t.l_min) {
            l = t.l_min;
        }
    }
    occupancy[cell] = l;
    const uint32_t to = typeOf(l, t);
    if (from != to) {
        // One int per counter-map cell: occupied change in the low 16 bits, unknown change above
        // (as a plain sum: occupied + 65536 * unknown, both within +-8, the sub-cells of a cell).
        const int d = (static_cast<int>(to == kOccupied) - static_cast<int>(from == kOccupied)) +
                      65536 * (static_cast<int>(to == kUnknown) - static_cast<int>(from == kUnknown));
        atomicAdd(&delta[counterCellOf(cell, m, c)], d);
    }
}

__global__ void applyCells(const uint32_t *touched, const Thresholds t, const Geometry m, const Geometry c,
                           float *occupancy, uint32_t *counter, int *delta, const Counts *counts) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < counts->touched && i < kListCap) {  // counts->touched is final: traceRays has finished
        applyCell(touched[i], t, m, c, occupancy, counter, delta);
    }
}

// Fallback when the touched list overflowed: every cell with a non-empty counter.
__global__ void applyDense(const uint32_t cells, const Thresholds t, const Geometry m, const Geometry c,
                           float *occupancy, uint32_t *counter, int *delta) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cells && counter[i] != 0) {
        applyCell(i, t, m, c, occupancy, counter, delta);
    }
}

struct Slabs {  // by value to the kernel
    int axis, n;
    int local_id[RaycastBackend::kMaxSlabs];
};

// Map sliding: the cells of whole slabs leave the map and go back to unknown (ProbMap::resetCell
// over SlidingMap::clearMemoryOutOfMap). One thread per cell. Slabs of different axes overlap and
// their kernels run one after the other, but the exchange also makes a repeated cell harmless:
// only one thread sees its old value.
__global__ void clearSlabs_(const Slabs s, const Thresholds t, const Geometry m, const Geometry c, float *occupancy,
                           int *delta) {
    const int a1 = (s.axis + 1) % 3, a2 = (s.axis + 2) % 3;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t per_slab = static_cast<uint32_t>(m.map_size[a1]) * m.map_size[a2];
    if (i >= per_slab * s.n) {
        return;
    }
    int l[3];
    l[s.axis] = s.local_id[i / per_slab];
    l[a1] = static_cast<int>((i % per_slab) / m.map_size[a2]) - m.half_map_size[a1];
    l[a2] = static_cast<int>((i % per_slab) % m.map_size[a2]) - m.half_map_size[a2];
    const uint32_t cell = hashOf(l, m);
    const uint32_t from = typeOf(atomicExch(&occupancy[cell], 0.0f), t);
    if (from != kUnknown) {
        atomicAdd(&delta[counterCellOf(cell, m, c)], 65536 - static_cast<int>(from == kOccupied));
    }
}

// Kernel 3: the counter-map cells with a change, for the CPU; zeroes them.
__global__ void gatherDeltas(const uint32_t counter_cells, int *delta, CounterDelta *out, Counts *counts) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= counter_cells) {
        return;
    }
    const int d = delta[i];
    if (d == 0) {
        return;
    }
    delta[i] = 0;
    const uint32_t slot = atomicAdd(&counts->deltas, 1u);
    if (slot < kDeltaCap) {
        const int occupied = static_cast<int16_t>(d & 0xffff);  // the low 16 bits, signed
        out[slot] = CounterDelta{i, static_cast<int16_t>(occupied), static_cast<int16_t>((d - occupied) / 65536)};
    }
}

class Backend final : public RaycastBackend {
public:
    Backend(const size_t cells, const size_t counter_cells)
        : cells_(static_cast<uint32_t>(cells)), counter_cells_(static_cast<uint32_t>(counter_cells)) {
        try {
            allocate();
        } catch (...) {
            release();  // a half-built backend must not keep hundreds of MB pinned
            throw;
        }
    }

    ~Backend() override { release(); }

    Backend(const Backend &) = delete;
    Backend &operator=(const Backend &) = delete;

    void allocate() {
        ROGMAP_CUDA_CHECK(cudaStreamCreate(&stream_));
        // Pinned + mapped: the GPU uses these host buffers in place (on a Jetson the device
        // pointer is the host pointer), the CPU reads them at normal speed.
        occupancy_ = pinned<float>(cells_, occupancy_dev_);
        rays_ = pinned<Ray>(kMaxRays, rays_dev_);
        deltas_ = pinned<CounterDelta>(kDeltaCap, deltas_dev_);
        // GPU-only scratch: never read by the CPU in place
        ROGMAP_CUDA_CHECK(cudaMalloc(&counts_dev_, sizeof(Counts)));
        ROGMAP_CUDA_CHECK(cudaMalloc(&counter_, cells_ * sizeof(uint32_t)));
        ROGMAP_CUDA_CHECK(cudaMalloc(&touched_, kListCap * sizeof(uint32_t)));
        ROGMAP_CUDA_CHECK(cudaMalloc(&delta_, counter_cells_ * sizeof(int)));
        clearCounters();
    }

    void release() {  // every free below accepts a null pointer
        if (stream_) {
            cudaStreamSynchronize(stream_);
        }
        cudaFree(touched_);
        cudaFree(counter_);
        cudaFree(delta_);
        cudaFree(counts_dev_);
        cudaFreeHost(deltas_);
        cudaFreeHost(rays_);
        cudaFreeHost(occupancy_);
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    float *occupancy() override { return occupancy_; }
    Ray *rays() override { return rays_; }
    int maxRays() const override { return kMaxRays; }

    void trace(const int n, const Geometry &g) override {
        if (n <= 0) {
            return;
        }
        traceRays<<<(n + kBlock - 1) / kBlock, kBlock, 0, stream_>>>(rays_dev_, n, g, counter_, touched_, counts_dev_);
        ROGMAP_CUDA_CHECK(cudaGetLastError());
        // rays_ is reused by the caller for the next chunk: wait before returning
        ROGMAP_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    void apply(const Thresholds &t, const Geometry &map, const Geometry &counter_map, const CounterDelta *&deltas,
               size_t &n_deltas, size_t &n_touched) override {
        n_touched = readCounts().touched;
        if (n_touched <= kListCap) {
            const uint32_t n = static_cast<uint32_t>(n_touched);
            if (n > 0) {
                applyCells<<<(n + kBlock - 1) / kBlock, kBlock, 0, stream_>>>(touched_, t, map, counter_map,
                                                                             occupancy_dev_, counter_, delta_, counts_dev_);
            }
        } else {
            applyDense<<<(cells_ + kBlock - 1) / kBlock, kBlock, 0, stream_>>>(cells_, t, map, counter_map,
                                                                              occupancy_dev_, counter_, delta_);
        }
        ROGMAP_CUDA_CHECK(cudaGetLastError());
        gather(deltas, n_deltas);
    }

    void clearSlabs(const int axis, const int *local_ids, const int n, const Thresholds &t, const Geometry &map,
                    const Geometry &counter_map) override {
        Slabs s;
        s.axis = axis;
        s.n = n;
        for (int i = 0; i < n; ++i) {
            s.local_id[i] = local_ids[i];
        }
        const uint32_t cells = static_cast<uint32_t>(map.map_size[(axis + 1) % 3]) * map.map_size[(axis + 2) % 3] * n;
        clearSlabs_<<<(cells + kBlock - 1) / kBlock, kBlock, 0, stream_>>>(s, t, map, counter_map, occupancy_dev_, delta_);
        ROGMAP_CUDA_CHECK(cudaGetLastError());
    }

    void finishClear(const CounterDelta *&deltas, size_t &n_deltas) override {
        const uint32_t touched = readCounts().touched;  // rays traced but not applied yet (batch updates)
        gather(deltas, n_deltas);
        if (touched != 0) {  // gather() zeroed the list lengths: put the touched count back
            const Counts keep{touched, 0};
            ROGMAP_CUDA_CHECK(cudaMemcpyAsync(counts_dev_, &keep, sizeof(keep), cudaMemcpyHostToDevice, stream_));
            ROGMAP_CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
    }

    void clearCounters() override {
        ROGMAP_CUDA_CHECK(cudaMemsetAsync(counter_, 0, cells_ * sizeof(uint32_t), stream_));
        ROGMAP_CUDA_CHECK(cudaMemsetAsync(delta_, 0, counter_cells_ * sizeof(int), stream_));
        ROGMAP_CUDA_CHECK(cudaMemsetAsync(counts_dev_, 0, sizeof(Counts), stream_));
        ROGMAP_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

private:
    static constexpr int kBlock = 256;

    // Kernel 3 and the hand-over of its list.
    void gather(const CounterDelta *&deltas, size_t &n_deltas) {
        gatherDeltas<<<(counter_cells_ + kBlock - 1) / kBlock, kBlock, 0, stream_>>>(counter_cells_, delta_, deltas_dev_,
                                                                                    counts_dev_);
        ROGMAP_CUDA_CHECK(cudaGetLastError());
        n_deltas = readCounts().deltas;  // waits for the kernels
        ROGMAP_CUDA_CHECK(cudaMemsetAsync(counts_dev_, 0, sizeof(Counts), stream_));
        if (n_deltas > kDeltaCap) {
            // shortcut: more than 4 M counter-map cells changed in one update cannot be handed to
            // the CPU; fail loudly. Upgrade to a chunked read-back if this ever fires.
            throw std::runtime_error("rog_map CUDA: counter delta list overflow (" + std::to_string(n_deltas) + ")");
        }
        deltas = deltas_;
    }

    Counts readCounts() {  // after everything queued on the stream has finished
        Counts c;
        ROGMAP_CUDA_CHECK(cudaMemcpyAsync(&c, counts_dev_, sizeof(c), cudaMemcpyDeviceToHost, stream_));
        ROGMAP_CUDA_CHECK(cudaStreamSynchronize(stream_));
        return c;
    }

    template <typename T>
    static T *pinned(const size_t n, T *&dev) {
        void *host = nullptr, *device = nullptr;
        ROGMAP_CUDA_CHECK(cudaHostAlloc(&host, n * sizeof(T), cudaHostAllocMapped));
        ROGMAP_CUDA_CHECK(cudaHostGetDevicePointer(&device, host, 0));
        dev = static_cast<T *>(device);
        return static_cast<T *>(host);
    }

    uint32_t cells_, counter_cells_;
    cudaStream_t stream_{};
    float *occupancy_{}, *occupancy_dev_{};
    Ray *rays_{}, *rays_dev_{};
    CounterDelta *deltas_{}, *deltas_dev_{};
    int *delta_{};  // per counter-map cell, GPU only
    Counts *counts_dev_{};
    uint32_t *counter_{}, *touched_{};
};

}  // namespace

std::unique_ptr<RaycastBackend> RaycastBackend::create(const size_t cells, const size_t counter_cells,
                                                       std::string &why_not) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        why_not = "no CUDA device";
        return nullptr;
    }
    // Sleep while the GPU works instead of spinning a CPU core the planner needs. Must be set
    // before the context exists; if another library created it first, the default stays.
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync | cudaDeviceMapHost);
    try {
        return std::unique_ptr<RaycastBackend>(new Backend(cells, counter_cells));
    } catch (const std::exception &e) {
        why_not = e.what();
        return nullptr;
    }
}

}  // namespace cuda
}  // namespace rog_map
