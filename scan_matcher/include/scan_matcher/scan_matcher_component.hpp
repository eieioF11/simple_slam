#pragma once
#include <atomic>
#include <execution>
#include <iostream>
#include <laser_geometry/laser_geometry.hpp>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
// Iridescence
#include <glk/indexed_pointcloud_buffer.hpp>
#include <glk/primitives/primitives.hpp>
#include <guik/spdlog_sink.hpp>
#include <guik/viewer/light_viewer.hpp>
#include <spdlog/fmt/ostr.h>
#include <spdlog/sinks/ringbuffer_sink.h>
#include <spdlog/spdlog.h>
// small gicp
// #include <small_gicp/benchmark/read_points.hpp>
// #include <small_gicp/pcl/pcl_point.hpp>
// #include <small_gicp/pcl/pcl_point_traits.hpp>
// #include <small_gicp/pcl/pcl_registration.hpp>
// #include <small_gicp/util/downsampling_omp.hpp>
// Eigen
#include <Eigen/Dense>
// extention node
#include "extension_node/extension_node.hpp"
// common_utils
#define USE_ROS2
#define USE_PCL
#include "common_utils/common_utils.hpp"
// OpenMP
#include <omp.h>

#define _ENABLE_ATOMIC_ALIGNMENT_FIX
//******************************************************************************
// デバック関連設定
#define N 1000
#define SCALE 0.01

#define T_X 1.0
#define T_Y 2.0
#define T_YAW 1.0

#define PCL_POINT_TYPE pcl::PointXYZ
//******************************************************************************
using namespace std::chrono_literals;
//-- [ToDo] common_utils Update --
// namespace pcl_utils {
//   /**
//    * @brief transform_cloud
//    *
//    * @tparam POINT_TYPE
//    * @param cloud pcl::PointCloud<POINT_TYPE>
//    * @param x double
//    * @param y double
//    * @param z double
//    * @param roll double
//    * @param pitch double
//    * @param yaw double
//    * @return pcl::PointCloud<POINT_TYPE>
//    */
//   template <typename POINT_TYPE = pcl::PointXYZ>
//   pcl::PointCloud<POINT_TYPE> transform_cloud(const pcl::PointCloud<POINT_TYPE>& cloud, double x, double y, double z, double roll, double pitch,
//                                               double yaw) {
//     pcl::PointCloud<POINT_TYPE> output_cloud;
//     Eigen::Affine3f transformatoin = pcl::getTransformation(x, y, z, roll, pitch, yaw);
//     pcl::transformPointCloud<POINT_TYPE>(cloud, output_cloud, transformatoin);
//     return output_cloud;
//   }

// } // namespace pcl_utils
//-- [ToDo] common_utils Update --
namespace simple_slam {

  template <typename POINT_TYPE>
  std::vector<Eigen::Vector3f> to_vector_cloud(const pcl::PointCloud<POINT_TYPE>& cloud) {
    std::vector<Eigen::Vector3f> points;
    points.reserve(cloud.size());
#pragma omp parallel for schedule(dynamic)
    for (const auto& point : cloud.points) {
      points.emplace_back(point.x, point.y, point.z);
    }
    return points;
  }

  template <typename POINT_TYPE>
  std::shared_ptr<glk::PointCloudBuffer> to_cloud_buffer(const pcl::PointCloud<POINT_TYPE>& cloud) {
    auto cloud_buffer = std::make_shared<glk::PointCloudBuffer>(to_vector_cloud<POINT_TYPE>(cloud));
    return cloud_buffer;
  }
  class ScanMatcher : public ext_rclcpp::ExtensionNode {
  public:
    ScanMatcher(const rclcpp::NodeOptions& options) : ScanMatcher("", options) {}
    ScanMatcher(const std::string& name_space = "", const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : ext_rclcpp::ExtensionNode("scan_matcher_node", name_space, options), tf_buffer_(this->get_clock()), listener_(tf_buffer_) {
      RCLCPP_INFO(this->get_logger(), "start scan_matcher_node");
      // setup
      SMAP_PUBLISH_RATE = param<double>("scan_matcher.storage_map.publish_rate", 1.0);
      VOXELGRID_SIZE    = param<double>("scan_matcher.voxel_grid.size", 0.1);
      kill_switch_      = false;
      // logger
      // const int ringbuffer_size = 100;
      // ringbuffer_sink_          = std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(ringbuffer_size);
      logger_ = spdlog::get("scan_matcher_logger");
      // logger_->set_level(spdlog::level::trace);
      // logger_->sinks().emplace_back(ringbuffer_sink_);
      // init
      // publisher
      out_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("scan_matcher/out_points", rclcpp::QoS(10));
      // subscriber
      cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("in_points", rclcpp::QoS(10),
                                                                            [&](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {});
      timer_     = this->create_wall_timer(1s * SMAP_PUBLISH_RATE, [&]() {
        // Eigen::Isometry3d T_target_source = scan_matching(target_cloud_.makeShared(), source_cloud_.makeShared()); 
      });
      // for (int i = 0; i < N; i++) {
      //   PCL_POINT_TYPE p;
      //   if (i < N / 2) {
      //     p.x = 0.0;
      //     p.y = i * SCALE;
      //     p.z = 0.0;
      //   } else {
      //     p.x = i * SCALE;
      //     p.y = 0.0;
      //     p.z = 0.0;
      //   }
      //   source_cloud_.push_back(p);
      // }
      // thread_ = std::thread([this] { viewer_loop(); });
    }
    ~ScanMatcher() {
      kill_switch_ = true;
      if (thread_.joinable()) {
        thread_.join();
      }
    }

  private:
    double SMAP_PUBLISH_RATE;
    double VOXELGRID_SIZE;
    bool kill_switch_;

    // viewer
    std::shared_ptr<spdlog::sinks::ringbuffer_sink_mt> ringbuffer_sink_;
    // Logging
    std::shared_ptr<spdlog::logger> logger_;
    // test
    float t_x   = T_X;
    float t_y   = T_Y;
    float t_yaw = T_YAW;
    pcl::PointCloud<PCL_POINT_TYPE> source_cloud_;
    pcl::PointCloud<PCL_POINT_TYPE> target_cloud_;
    // tf
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener listener_;
    // timer
    rclcpp::TimerBase::SharedPtr timer_;
    // subscriber
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    // publisher
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr out_cloud_pub_;

    std::mutex mtx_;
    std::thread thread_;

    void viewer_loop() {
      auto viewer                         = guik::LightViewer::instance();
      guik::ShaderSetting& global_setting = viewer->shader_setting();
      global_setting.set_point_scale_screenspace(); // Set the point scale mode to screenspace
      global_setting.set_point_size(5.0f);          // Set the base point size to 5.0
      while (viewer->spin_once()) {
        // // Register a callback for UI rendering
        // viewer->register_ui_callback("ui", [&]() {
        //   // In the callback, you can call ImGui commands to create your UI.
        //   // Here, we use "DragFloat" and "Button" to create a simple UI.
        //   ImGui::DragFloat("T_X", &t_x, 0.01f);
        //   ImGui::DragFloat("T_Y", &t_y, 0.01f);
        //   ImGui::DragFloat("T_Yaw", &t_yaw, 0.01f);

        //   if (ImGui::Button("Close")) {
        //     viewer->close();
        //   }
        // });
        // std::lock_guard<std::mutex> lock(mtx_);
        // target_cloud_ = pcl_utils::transform_cloud<PCL_POINT_TYPE>(source_cloud_, t_x, t_y, 0.0, 0.0, 0.0, t_yaw);

        // icp.setInputSource(source_cloud);
        // icp.setInputTarget(target_cloud);
        // auto [aligned_cloud, score, converged, final_transformation] = icp.transform(source_cloud, target_cloud);

        // if (converged) {
        //   spdlog::info("ICP has converged.");
        //   spdlog::info("Fitness score: {:.2f}", score);
        //   spdlog::info("Final transformation:\n{}", final_transformation);
        //   Eigen::Vector3d translation = final_transformation.block<3, 1>(0, 3);
        //   Eigen::Matrix3d rotation    = final_transformation.block<3, 3>(0, 0);
        //   Eigen::Vector3d euler       = rotation.eulerAngles(0, 1, 2);
        //   spdlog::info("Translation: {} \nRPY:{}", translation.transpose());
        //   spdlog::info("RPY:{}", euler.transpose());
        // } else {
        //   spdlog::warn("ICP did not converge.");
        // }

        viewer->update_drawable("source_points", to_cloud_buffer<PCL_POINT_TYPE>(source_cloud_), guik::FlatRed());
        viewer->update_drawable("target_points", to_cloud_buffer<PCL_POINT_TYPE>(target_cloud_), guik::FlatGreen());
        // viewer->update_drawable("aligned_points", to_cloud_buffer<PCL_POINT_TYPE>(aligned_cloud), guik::FlatBlue());

        // viewer->register_ui_callback("logging", guik::create_logger_ui(glim::get_ringbuffer_sink(), 0.5));
      }
      guik::LightViewer::destroy();
    }

    std::optional<std::tuple<double, Eigen::Matrix4d, pcl::PointCloud<pcl::PointNormal>>> scan_matching(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& raw_target, const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& raw_sourc){

    }

    // Eigen::Isometry3d scan_matching(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& raw_target, const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& raw_source) {
    //   // using namespace small_gicp;
    //   // Downsample points and convert them into pcl::PointCloud<pcl::PointCovariance>.
    //   pcl::PointCloud<pcl::PointCovariance>::Ptr target =
    //       voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZ>, pcl::PointCloud<pcl::PointCovariance>>(*raw_target, 0.25);
    //   pcl::PointCloud<pcl::PointCovariance>::Ptr source =
    //       voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZ>, pcl::PointCloud<pcl::PointCovariance>>(*raw_source, 0.25);

    //   // Estimate covariances of points.
    //   const int num_threads   = 4;
    //   const int num_neighbors = 20;
    //   estimate_covariances_omp(*target, num_neighbors, num_threads);
    //   estimate_covariances_omp(*source, num_neighbors, num_threads);

    //   // Create KdTree for target and source.
    //   auto target_tree = std::make_shared<KdTree<pcl::PointCloud<pcl::PointCovariance>>>(target, KdTreeBuilderOMP(num_threads));
    //   auto source_tree = std::make_shared<KdTree<pcl::PointCloud<pcl::PointCovariance>>>(source, KdTreeBuilderOMP(num_threads));

    //   Registration<GICPFactor, ParallelReductionOMP> registration;
    //   registration.reduction.num_threads = num_threads;
    //   registration.rejector.max_dist_sq  = 1.0;

    //   // Align point clouds. Note that the input point clouds are pcl::PointCloud<pcl::PointCovariance>.
    //   auto result = registration.align(*target, *source, *target_tree, Eigen::Isometry3d::Identity());
    //   Eigen::Isometry3d T_target_source = result.T_target_source.matrix();
    //   std::cout << "--- T_target_source ---" << std::endl << T_target_source << std::endl;
    //   std::cout << "converged:" << result.converged << std::endl;
    //   std::cout << "error:" << result.error << std::endl;
    //   std::cout << "iterations:" << result.iterations << std::endl;
    //   std::cout << "num_inliers:" << result.num_inliers << std::endl;
    //   std::cout << "--- H ---" << std::endl << result.H << std::endl;
    //   std::cout << "--- b ---" << std::endl << result.b.transpose() << std::endl;
    //   return T_target_source;
    // }
  };
} // namespace simple_slam
#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(simple_slam::ScanMatcher)
