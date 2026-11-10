/**
* This file is part of SUPER
*
* Copyright 2025 Yunfan REN, MaRS Lab, University of Hong Kong, <mars.hku.hk>
* Developed by Yunfan REN <renyf at connect dot hku dot hk>
* for more information see <https://github.com/hku-mars/SUPER>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* SUPER is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* SUPER is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with SUPER. If not, see <http://www.gnu.org/licenses/>.
*/



#include <ros_interface/ros2/fsm_ros2.hpp>
#include <oa_interfaces/srv/map_query.hpp>
#include <rclcpp/version.h>

/*
 * Test code:
 *      roslaunch simulator test_env.launch
 * */
#define BACKWARD_HAS_DW 1

#include "utils/header/backward.hpp"

namespace backward {
    backward::SignalHandling sh;
}


using namespace fsm;
using namespace std;
FsmRos2::Ptr fsm_ptr;

#include "rclcpp/rclcpp.hpp"

int main(int argc, char** argv) {
    // 初始化ROS2
    rclcpp::init(argc, argv);

    // 设置PCL日志级别（保持不变）
    pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

    // 创建节点（替换ROS1的NodeHandle）
    auto node = std::make_shared<rclcpp::Node>("fsm_node");

    // 检查是否使用仿真时间
    while (rclcpp::ok()) {
        bool use_sim_time;

        // 获取参数
        if (node->get_parameter("use_sim_time", use_sim_time)) {
            if (!use_sim_time) {
                // 如果 use_sim_time 为 false，打印信息并退出循环
                std::cout << " -- [Bench] Use sim time is false, begin replay." << std::endl;
                break;
            } else {
                // 如果 use_sim_time 为 true，设置为 false
                node->set_parameter(rclcpp::Parameter("use_sim_time", false));
            }
        } else {
            // 如果参数不存在，设置为 false
            node->set_parameter(rclcpp::Parameter("use_sim_time", false));
        }

        // 休眠 1 秒
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // 初始化FSM

#define CONFIG_FILE_DIR(name) (std::string(std::string(ROOT_DIR) + "config/"+(name)))
    auto fsm_ptr = std::make_shared<FsmRos2>();
    std::string cfg_path = "click.yaml";
    node->declare_parameter("config_name", cfg_path);
    node->get_parameter("config_name", cfg_path);
    cfg_path = CONFIG_FILE_DIR(cfg_path);
    fsm_ptr->init(node,cfg_path);

    // [px4_oa] /oa/map_query: classify balls in the live map for Mission OA's rejoin search, so it
    // doesn't keep a second map. Reads race the map update like the planner's own reads do.
    // Occupied: the inflated layer already marks every cell within `inflation` of an occupied one,
    // so a ball that much smaller on it (0.2 m cells) covers the radius at ~1/36 of the reads of the
    // 0.1 m prob map (a 2.5 m ball: ~1.8k instead of 65k). Unknown: the prob map sampled on the same
    // 0.2 m grid (the inflated layer has no unknown without unk_inflation_en).
    using MapQuery = oa_interfaces::srv::MapQuery;
    const rog_map::Config rog_cfg(cfg_path);
    const double inf_res = rog_cfg.inflation_resolution;
    const double inflation = rog_cfg.inflation_resolution * rog_cfg.inflation_step;
    auto query_group = node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    auto query_srv = node->create_service<MapQuery>(
            "/oa/map_query",
            [map = fsm_ptr->getMap(), inf_res, inflation](const std::shared_ptr<MapQuery::Request> req,
                                                          std::shared_ptr<MapQuery::Response> res) {
                // offsets on the 0.2 m grid within r
                auto ball = [inf_res](double r) {
                    std::vector<rog_map::Vec3f> out;
                    const int n = static_cast<int>(std::floor(r / inf_res));
                    for (int i = -n; i <= n; i++) {
                        for (int j = -n; j <= n; j++) {
                            for (int k = -n; k <= n; k++) {
                                const rog_map::Vec3f d(i * inf_res, j * inf_res, k * inf_res);
                                if (d.norm() <= r) {
                                    out.push_back(d);
                                }
                            }
                        }
                    }
                    return out;
                };
                const auto occ_ball = ball(std::max(0.0, req->radius - inflation));
                const auto unk_ball = ball(req->radius);
                res->status.reserve(req->points.size());
                for (const auto &p: req->points) {
                    const rog_map::Vec3f c(p.x, p.y, p.z);
                    if (!map->insideLocalMap(c)) {
                        res->status.push_back(MapQuery::Response::OUTSIDE);
                        continue;
                    }
                    bool occupied = false;
                    for (const auto &d: occ_ball) {
                        if (map->getInfGridType(c + d) == super_utils::OCCUPIED) {
                            occupied = true;
                            break;
                        }
                    }
                    if (occupied) {
                        res->status.push_back(MapQuery::Response::OCCUPIED);
                        continue;
                    }
                    uint8_t status = MapQuery::Response::FREE;
                    for (const auto &d: unk_ball) {
                        const auto gt = map->getGridType(rog_map::Vec3f(c + d));
                        if (gt == super_utils::OCCUPIED) {  // missed by the coarser occupied test
                            status = MapQuery::Response::OCCUPIED;
                            break;
                        }
                        if (gt != super_utils::KNOWN_FREE) {
                            status = MapQuery::Response::UNKNOWN;  // keep looking for occupied
                        }
                    }
                    res->status.push_back(status);
                }
            },
#if RCLCPP_VERSION_MAJOR >= 17  // rclcpp::QoS overload (deprecates rmw_qos_profile_t); Humble is 16
            rclcpp::ServicesQoS(), query_group);
#else
            rmw_qos_profile_services_default, query_group);
#endif

    // 打印启动信息（ROS2风格）
    RCLCPP_INFO(node->get_logger(), "\033[32m -- [Fsm-Test] Begin.\033[0m");

    // 创建执行器（替换AsyncSpinner）
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    // 等待关闭
    rclcpp::shutdown();
    return 0;
}
