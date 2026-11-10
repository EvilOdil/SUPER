// [px4_oa] rogmap_bench: replays recorded lidar frames into ROG-Map without ROS and times every
// map update. The reference for the CUDA backend: same input, per-stage times, and a digest of
// the resulting map to compare backends exactly. See docs/cuda_rogmap_plan.md.
//
//   rogmap_bench <config.yaml> <frames.bin> [--max-frames N] [--csv out.csv] [--queries N]
//
// frames.bin (scripts/bag_to_rogmap_frames.py), little endian: "RMB1", then per frame
//   double t; double p[3]; double q[4] (w x y z); uint32 n; float xyzi[4 * n]
#include <rog_map/rog_map.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

double msSince(const Clock::time_point &t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

class BenchMap : public rog_map::ROGMap {
public:
    explicit BenchMap(const std::string &cfg_path) {
        cfg_ = rog_map::Config(cfg_path);
        init();
    }

    const double getSystemWalltimeNow() override {
        return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
    }

    bool flyable(const rog_map::Vec3f &p) const {  // updateProbMap returns early outside this band
        return p.z() >= cfg_.virtual_ground_height && p.z() <= cfg_.virtual_ceil_height;
    }

    const std::vector<double> &stageTimes() const { return time_consuming_; }  // [s], see time_consuming_name_

    // FNV-1a over the raw log-odds buffer, and the cell counts per type: equal maps, equal digest.
    void digest(uint64_t &hash, size_t &occupied, size_t &known_free, size_t &unknown) const {
        hash = 1469598103934665603ull;
        occupied = known_free = unknown = 0;
        for (const float v : occupancy_buffer_) {
            uint32_t bits;
            std::memcpy(&bits, &v, sizeof(bits));
            for (int i = 0; i < 4; ++i) {
                hash = (hash ^ ((bits >> (8 * i)) & 0xffu)) * 1099511628211ull;
            }
            if (isOccupied(v)) {
                ++occupied;
            } else if (isKnownFree(v)) {
                ++known_free;
            } else {
                ++unknown;
            }
        }
    }

    // SUPER-style reads (isOccupied / isUnknown of single points near the vehicle), [ns per read].
    double queryNs(const rog_map::Vec3f &centre, const int n) const {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> d(-5.0, 5.0);
        std::vector<rog_map::Vec3f> pts(n);
        for (auto &p : pts) {
            p = centre + rog_map::Vec3f(d(rng), d(rng), 0.3 * d(rng));
        }
        size_t hits = 0;
        const auto t0 = Clock::now();
        for (const auto &p : pts) {
            hits += isOccupied(p) + isUnknown(p);
        }
        const double ns = msSince(t0) * 1e6 / (2.0 * n);
        std::printf("map reads: %.1f ns per read (%d points, %zu positive)\n", ns, n, hits);
        return ns;
    }
};

long rssMb() {
    std::ifstream f("/proc/self/status");
    for (std::string line; std::getline(f, line);) {
        if (line.rfind("VmRSS:", 0) == 0) {
            return std::stol(line.substr(6)) / 1024;
        }
    }
    return -1;
}

double pct(std::vector<double> v, const double q) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(q * v.size()))];
}
}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <config.yaml> <frames.bin> [--max-frames N] [--csv out.csv] [--queries N]\n", argv[0]);
        return 2;
    }
    long max_frames = -1;
    int queries = 1000000;
    std::string csv_path;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--max-frames") {
            max_frames = std::stol(argv[i + 1]);
        } else if (a == "--csv") {
            csv_path = argv[i + 1];
        } else if (a == "--queries") {
            queries = std::stoi(argv[i + 1]);
        } else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }

    std::FILE *in = std::fopen(argv[2], "rb");
    char magic[4];
    if (!in || std::fread(magic, 1, 4, in) != 4 || std::memcmp(magic, "RMB1", 4) != 0) {
        std::fprintf(stderr, "%s: not a frames file\n", argv[2]);
        return 1;
    }

    BenchMap map(argv[1]);
    std::FILE *csv = csv_path.empty() ? nullptr : std::fopen(csv_path.c_str(), "w");
    if (csv) {
        std::fprintf(csv, "frame,t,points,total_ms,raycast_ms,update_ms,inflation_ms,cells\n");
    }

    std::vector<double> total, raycast, update;
    std::vector<float> buf;
    rog_map::PointCloud cloud;
    rog_map::Vec3f last_p(0, 0, 0);
    long frame = 0, skipped = 0;
    double hdr[8];
    uint32_t n;
    while ((max_frames < 0 || frame < max_frames) && std::fread(hdr, sizeof(double), 8, in) == 8 &&
           std::fread(&n, sizeof(n), 1, in) == 1) {
        buf.resize(4 * static_cast<size_t>(n));
        if (std::fread(buf.data(), sizeof(float), buf.size(), in) != buf.size()) {
            std::fprintf(stderr, "truncated frame %ld\n", frame);
            break;
        }
        const rog_map::Vec3f p(hdr[1], hdr[2], hdr[3]);
        const rog_map::Pose pose(p, super_utils::Quatf(hdr[4], hdr[5], hdr[6], hdr[7]));
        ++frame;
        if (!map.flyable(p)) {
            ++skipped;
            continue;
        }
        cloud.resize(n);
        for (uint32_t i = 0; i < n; ++i) {
            cloud[i].x = buf[4 * i];
            cloud[i].y = buf[4 * i + 1];
            cloud[i].z = buf[4 * i + 2];
            cloud[i].intensity = buf[4 * i + 3];
        }
        const auto t0 = Clock::now();
        map.updateProbMap(cloud, pose);
        const double ms = msSince(t0);
        const auto &st = map.stageTimes();
        total.push_back(ms);
        raycast.push_back(st[1] * 1e3);
        update.push_back(st[2] * 1e3);
        last_p = p;
        if (csv) {
            std::fprintf(csv, "%ld,%.3f,%u,%.3f,%.3f,%.3f,%.3f,%.0f\n", frame, hdr[0], n, ms, st[1] * 1e3,
                         st[2] * 1e3, st[3] * 1e3, st[5]);
        }
    }
    std::fclose(in);
    if (csv) {
        std::fclose(csv);
    }
    if (total.empty()) {
        std::fprintf(stderr, "no frame processed (%ld read, %ld on the ground)\n", frame, skipped);
        return 1;
    }

    std::printf("frames: %zu updated, %ld skipped (vehicle below the map's ground height)\n", total.size(), skipped);
    const auto row = [](const char *name, const std::vector<double> &v) {
        std::printf("%-12s median %8.1f  p90 %8.1f  p99 %8.1f  max %8.1f  [ms]\n", name, pct(v, 0.5), pct(v, 0.9),
                    pct(v, 0.99), *std::max_element(v.begin(), v.end()));
    };
    row("total", total);
    row("raycast", raycast);
    row("update", update);
    const size_t in_time = std::count_if(total.begin(), total.end(), [](double ms) { return ms < 100.0; });
    std::printf("frames under 100 ms (keeps up with a 10 Hz lidar): %.1f %%\n", 100.0 * in_time / total.size());
    map.queryNs(last_p, queries);
    uint64_t hash;
    size_t occ, fre, unk;
    map.digest(hash, occ, fre, unk);
    std::printf("map digest: %016llx  occupied %zu  free %zu  unknown %zu\n", static_cast<unsigned long long>(hash), occ,
                fre, unk);
    std::printf("memory: %ld MB resident\n", rssMb());
    return 0;
}
