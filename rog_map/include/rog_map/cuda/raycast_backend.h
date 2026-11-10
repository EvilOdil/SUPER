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

struct Geometry {      // the sliding map, in cells
    double resolution; // [m]
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

// Cell types in a Transition (not rog_map::GridType: no dependency on its values).
enum : uint32_t { kUnknown = 0, kKnownFree = 1, kOccupied = 2 };

// A cell whose type changed: cell index in the low 26 bits, then 2 bits from, 2 bits to.
inline uint32_t transitionCell(const uint32_t t) { return t & 0x03ffffffu; }
inline uint32_t transitionFrom(const uint32_t t) { return (t >> 26) & 3u; }
inline uint32_t transitionTo(const uint32_t t) { return (t >> 28) & 3u; }

class RaycastBackend {
public:
    // nullptr (and why) if there is no usable GPU or the map is too large for the cell packing.
    static std::unique_ptr<RaycastBackend> create(size_t cells, std::string &why_not);

    virtual ~RaycastBackend() = default;

    virtual float *occupancy() = 0;      // `cells` log-odds, pinned: ProbMap's occupancy buffer
    virtual Ray *rays() = 0;             // pinned, maxRays() entries: fill, then trace()
    virtual int maxRays() const = 0;

    // Kernel 1: count touches and hits of rays()[0, n) into the per-cell counters.
    virtual void trace(int n, const Geometry &g) = 0;

    // Kernel 2: fold the counters into the map and zero them. Returns the cells that changed
    // type (valid until the next call) and the number of cells touched since the last apply().
    virtual void apply(const Thresholds &t, const uint32_t *&transitions, size_t &n_transitions,
                       size_t &n_touched) = 0;

    virtual void clearCounters() = 0;
};

}  // namespace cuda
}  // namespace rog_map
