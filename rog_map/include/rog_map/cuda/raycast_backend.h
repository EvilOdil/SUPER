// [px4_oa] GPU backend for ROG-Map's probabilistic update: ray tracing (kernel 1) and writing the
// traced cells into the log-odds map (kernel 2). Plain C++ interface: no CUDA, Eigen or PCL
// types, so prob_map.cpp does not need the CUDA headers. See docs/cuda_rogmap_plan.md.
//
// Memory (Jetson: CPU and GPU share RAM): the log-odds map is pinned host memory that the GPU
// works on in place and the planner keeps reading from the CPU; nothing large is ever copied.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace rog_map {
namespace cuda {

struct Geometry {      // a sliding map (SlidingMap), in cells
    double resolution; // [m]
    double resolution_inv;
    int map_size[3];
    int half_map_size[3];
    int origin[3];     // local map origin, global cell index
};

struct Thresholds {    // log-odds, as rog_map::Config
    float l_hit, l_miss, l_min, l_max, l_occ, l_free;
};

struct Ray {           // prepared on the CPU exactly as ProbMap::raycastProcess does
    double start[3];   // first traced point (range_min out from the sensor)
    double end[3];     // clipped end point
    int hit_cell;      // cell (hash index) to count as a hit, or -1
    int pad;
};

// What one update changed in a cell of the counter (inflation) map: how many of its sub-cells
// became / stopped being occupied and unknown, summed on the GPU.
struct CounterDelta {
    uint32_t cell;     // hash index in the counter map
    int16_t occupied;
    int16_t unknown;
};

class RaycastBackend {
public:
    // cells: of the probability map; counter_cells: of the counter (inflation) map.
    // nullptr (and why) if there is no usable GPU.
    static std::unique_ptr<RaycastBackend> create(size_t cells, size_t counter_cells, std::string &why_not);

    virtual ~RaycastBackend() = default;

    virtual float *occupancy() = 0;      // `cells` log-odds, pinned: ProbMap's occupancy buffer
    virtual Ray *rays() = 0;             // pinned, maxRays() entries: fill, then trace()
    virtual int maxRays() const = 0;

    // Kernel 1: count touches and hits of rays()[0, n) into the per-cell counters.
    virtual void trace(int n, const Geometry &g) = 0;

    // Kernel 2: fold the counters into the map and zero them. Cells that change type are summed
    // per counter-map cell (kernel 3 gathers them): `deltas` (valid until the next call) is what
    // the CPU applies to the counter map. n_touched: cells touched since the last apply().
    virtual void apply(const Thresholds &t, const Geometry &map, const Geometry &counter_map,
                       const CounterDelta *&deltas, size_t &n_deltas, size_t &n_touched) = 0;

    // Map sliding (SlidingMap::clearMemoryOutOfMap): reset to unknown every cell whose local index
    // along `axis` is one of local_ids[0, n) (kernel clearSlabs; n <= kMaxSlabs). Slabs of
    // different axes may overlap. finishClear() then returns what all the clears since the last
    // call changed in the counter map, as apply() does.
    enum { kMaxSlabs = 32 };
    virtual void clearSlabs(int axis, const int *local_ids, int n, const Thresholds &t, const Geometry &map,
                            const Geometry &counter_map) = 0;
    virtual void finishClear(const CounterDelta *&deltas, size_t &n_deltas) = 0;

    virtual void clearCounters() = 0;
};

}  // namespace cuda
}  // namespace rog_map
