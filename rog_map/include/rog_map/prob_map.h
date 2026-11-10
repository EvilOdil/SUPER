/**
* This file is part of ROG-Map
*
* Copyright 2024 Yunfan REN, MaRS Lab, University of Hong Kong, <mars.hku.hk>
* Developed by Yunfan REN <renyf at connect dot hku dot hk>
* for more information see <https://github.com/hku-mars/ROG-Map>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* ROG-Map is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ROG-Map is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with ROG-Map. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once

// [px4_oa] Always included, and the members below always present: this header is compiled by
// other packages too (super_planner), and the class layout must not depend on a build flag.
#include "rog_map/cuda/raycast_backend.h"
#include <cstdlib>
#include <memory>
#include <new>
#include <queue>
#include <rog_map/inf_map.h>
#include <rog_map/free_cnt_map.h>
#include <rog_map/esdf_map.h>
#include <rog_map/rog_map_core/raycaster.h>


namespace rog_map {
    using super_utils::Pose;


    class ProbMap : public SlidingMap {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        typedef std::shared_ptr<ProbMap> Ptr;

        ProbMap() = default;

        ~ProbMap() override = default;

        void initProbMap();

        bool isOccupied(const Vec3f &pos) const;

        bool isUnknown(const Vec3f &pos) const;

        bool isKnownFree(const Vec3f &pos) const;

        bool isOccupiedInflate(const Vec3f &pos) const;

        bool isUnknownInflate(const Vec3f &pos) const;

        bool isKnownFreeInflate(const Vec3f & pos) const;

        bool isFrontier(const Vec3f &pos) const;

        bool isFrontier(const Vec3i &id_g) const;

        // Query result
        GridType getGridType(Vec3i &id_g) const;

        GridType getGridType(const Vec3f &pos) const;

        GridType getInfGridType(const Vec3f &pos) const;

        double getMapValue(const Vec3f &pos) const;

        void boxSearch(const Vec3f &_box_min, const Vec3f &_box_max,
                       const GridType &gt, vec_E<Vec3f> &out_points) const;

        void boxSearchInflate(const Vec3f &box_min, const Vec3f &box_max,
                              const GridType &gt, vec_E<Vec3f> &out_points) const;

        void boundBoxByLocalMap(Vec3f &box_min, Vec3f &box_max) const;

        Vec3f getLocalMapOrigin() const;

        Vec3f getLocalMapSize() const;

        double getResolution() const{
            return sc_.resolution;
        }

        double getInfResolution()const {
            return inf_map_->getResolution();
        }

        void updateOccPointCloud(const PointCloud &input_cloud);

        void writeTimeConsumingToLog(std::ofstream &log_file);

        void writeMapInfoToLog(std::ofstream &log_file);

        void updateProbMap(const PointCloud &cloud, const Pose &pose);

    protected:
        rog_map::Config cfg_;
        InfMap::Ptr inf_map_;
        FreeCntMap::Ptr fcnt_map_;
        ESDFMap::Ptr esdf_map_;
        /// Spherical neighborhood lookup table
        // [px4_oa] The log-odds map: ordinary memory, or (CUDA backend) the pinned buffer the GPU
        // updates in place. Read by the planner at all times.
        class MapBuffer {
        public:
            MapBuffer() = default;
            ~MapBuffer() { if (owned_) { std::free(data_); } }
            MapBuffer(const MapBuffer &) = delete;
            MapBuffer &operator=(const MapBuffer &) = delete;
            void allocate(const size_t n) {
                data_ = static_cast<float *>(std::calloc(n, sizeof(float)));
                if (!data_) { throw std::bad_alloc(); }
                size_ = n;
                owned_ = true;
            }
            void adopt(float *external, const size_t n) { data_ = external; size_ = n; owned_ = false; }
            float &operator[](const size_t i) { return data_[i]; }
            const float &operator[](const size_t i) const { return data_[i]; }
            float *begin() { return data_; }
            float *end() { return data_ + size_; }
            const float *begin() const { return data_; }
            const float *end() const { return data_ + size_; }
            size_t size() const { return size_; }
        private:
            float *data_{nullptr};
            size_t size_{0};
            bool owned_{false};
        } occupancy_buffer_;
        std::unique_ptr<cuda::RaycastBackend> gpu_;  // set: raycast + cell update run on the GPU
        int gpu_rays_{0};                            // rays queued in gpu_->rays()
        bool gpu_clear_slabs_{false};                // map sliding: clear whole slabs on the GPU

        bool map_empty_{true};
        struct RaycastData {
            raycaster::RayCaster raycaster;
            std::queue<Vec3i> update_cache_id_g;
            std::vector<uint16_t> operation_cnt;
            std::vector<uint16_t> hit_cnt;
            Vec3f cache_box_max, cache_box_min, local_update_box_max, local_update_box_min;
            int batch_update_counter{0};
            std::mutex raycast_range_mtx;
        } raycast_data_;

        vector<double> time_consuming_;
        vector<string> time_consuming_name_{"Total", "Raycast", "Update_cache", "Inflation", "PointCloudNumber",
                                            "CacheNumber", "InflationNumber"};

        // standardization query
        // Known free < l_free
        // occupied >= l_occ
        bool isKnownFree(const double &prob) const {
            return prob < cfg_.l_free;
        }

        bool isOccupied(const double &prob) const {
            return prob >= cfg_.l_occ;
        }

        bool isUnknown(const double &prob) const {
            return prob >= cfg_.l_free && prob < cfg_.l_occ;
        }

        void slideAllMap(const Vec3f &pos);

        // warning using this function will cause memory leak if the id_g is not in the map
        bool isOccupied(const Vec3i &id_g) const;

        bool isUnknown(const Vec3i &id_g) const;

        bool isKnownFree(const Vec3i &id_g) const;

        //====================================================================
        void resetCell(const int &hash_id) override;

        void probabilisticMapFromCache();

        void hitPointUpdate(const Vec3f &pos, const int &hash_id, const int &hit_num);

        void missPointUpdate(const Vec3f &pos, const int &hash_id, const int &hit_num);

        void raycastProcess(const PointCloud &input_cloud, const Vec3f &cur_odom);

        void insertUpdateCandidate(const Vec3i &id_g, bool is_hit);

        void updateLocalBox(const Vec3f &cur_odom);

        void resetLocalMap() override;
        void flushGpuRays();  // trace the rays queued in gpu_->rays()
        void clearMemoryOutOfMap(const std::vector<int> &clear_id, const int &i) override;
    };
}