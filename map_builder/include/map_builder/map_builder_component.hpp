#pragma once

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/trigger.hpp>

// Message Filters (同期用)
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>

// PCL
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

// extention node
#include "extension_node/extension_node.hpp"

// Eigen
#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace simple_slam {

  class MapBuilder : public ext_rclcpp::ExtensionNode {
  public:
    MapBuilder(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : ext_rclcpp::ExtensionNode("map_builder", options) {

      RCLCPP_INFO(this->get_logger(), "Starting Map Builder Node...");

      global_voxel_size_ = param<double>("map_builder.global_voxel_size", 0.2);
      grid_resolution_   = param<double>("map_builder.grid_resolution", 0.05);
      z_min_filter_      = param<double>("map_builder.z_min_filter", 0.1);
      z_max_filter_      = param<double>("map_builder.z_max_filter", 1.5);

      l_occ_  = std::log(0.7 / 0.3);
      l_free_ = std::log(0.3 / 0.7);
      l_max_  = 100.0;
      l_min_  = -100.0;

      global_map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("map_builder/global_map", rclcpp::QoS(1).transient_local());
      grid_map_pub_   = this->create_publisher<nav_msgs::msg::OccupancyGrid>("map", rclcpp::QoS(1).transient_local());

      path_sub_ = this->create_subscription<nav_msgs::msg::Path>("optimized_path", rclcpp::QoS(10),
                                                                 std::bind(&MapBuilder::pathCallback, this, std::placeholders::_1));

      odom_sub_.subscribe(this, "scan_matcher/keyframe_odom");
      cloud_sub_.subscribe(this, "scan_matcher/out_points");

      sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), odom_sub_, cloud_sub_);
      sync_->registerCallback(std::bind(&MapBuilder::keyframeCallback, this, std::placeholders::_1, std::placeholders::_2));

      save_map_srv_ = this->create_service<std_srvs::srv::Trigger>(
          "save_map", std::bind(&MapBuilder::saveMapService, this, std::placeholders::_1, std::placeholders::_2));
    }

  private:
    double global_voxel_size_;
    double grid_resolution_;
    double z_min_filter_;
    double z_max_filter_;

    double l_occ_;
    double l_free_;
    double l_max_;
    double l_min_;

    std::mutex mtx_;

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> keyframe_clouds_;
    std::vector<Eigen::Matrix4d> keyframe_raw_poses_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr latest_global_cloud_{new pcl::PointCloud<pcl::PointXYZ>()};

    nav_msgs::msg::OccupancyGrid::SharedPtr latest_grid_map_;

    using SyncPolicy = message_filters::sync_policies::ExactTime<nav_msgs::msg::Odometry, sensor_msgs::msg::PointCloud2>;
    message_filters::Subscriber<nav_msgs::msg::Odometry> odom_sub_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> cloud_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr global_map_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_map_pub_;

    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_map_srv_;

    void saveMapService(const std::shared_ptr<std_srvs::srv::Trigger::Request> req, std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
      std::lock_guard<std::mutex> lock(mtx_);
      (void)req;

      const char* home_dir = std::getenv("HOME");
      if (!home_dir) {
        res->success = false;
        res->message = "Failed to get HOME directory.";
        RCLCPP_ERROR(this->get_logger(), "%s", res->message.c_str());
        return;
      }

      auto now       = std::chrono::system_clock::now();
      auto in_time_t = std::chrono::system_clock::to_time_t(now);
      std::stringstream ss;
      ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
      std::string time_str = ss.str();

      std::string dir_path = std::string(home_dir) + "/map/" + time_str;
      try {
        std::filesystem::create_directories(dir_path);
      } catch (const std::filesystem::filesystem_error& e) {
        res->success = false;
        res->message = std::string("Failed to create directory: ") + e.what();
        RCLCPP_ERROR(this->get_logger(), "%s", res->message.c_str());
        return;
      }

      std::string pcd_path  = dir_path + "/map.pcd";
      std::string pgm_path  = dir_path + "/map.pgm";
      std::string yaml_path = dir_path + "/map.yaml";

      bool success_3d     = false;
      bool success_2d     = false;
      std::string res_msg = "Saved to " + dir_path + " -> ";

      if (latest_global_cloud_ && !latest_global_cloud_->empty()) {
        try {
          pcl::io::savePCDFileBinary(pcd_path, *latest_global_cloud_);
          success_3d = true;
          res_msg += "[map.pcd] ";
        } catch (const std::exception& e) {
          RCLCPP_ERROR(this->get_logger(), "Failed to save PCD: %s", e.what());
        }
      }

      if (latest_grid_map_) {
        std::ofstream pgm_out(pgm_path, std::ios::out | std::ios::binary);
        if (pgm_out) {
          pgm_out << "P5\n# CREATOR: simple_slam\n" << latest_grid_map_->info.width << " " << latest_grid_map_->info.height << "\n255\n";

          for (int y = latest_grid_map_->info.height - 1; y >= 0; --y) {
            for (unsigned int x = 0; x < latest_grid_map_->info.width; ++x) {
              int8_t val  = latest_grid_map_->data[y * latest_grid_map_->info.width + x];
              uint8_t pxl = 205;

              if (val == 0)
                pxl = 254;
              else if (val == 100)
                pxl = 0;
              else if (val > 0)
                pxl = 255 - static_cast<uint8_t>((val * 255) / 100);

              pgm_out.write(reinterpret_cast<const char*>(&pxl), 1);
            }
          }
          pgm_out.close();

          std::ofstream yaml_out(yaml_path);
          if (yaml_out) {
            yaml_out << "image: map.pgm\n";
            yaml_out << "resolution: " << latest_grid_map_->info.resolution << "\n";
            yaml_out << "origin: [" << latest_grid_map_->info.origin.position.x << ", " << latest_grid_map_->info.origin.position.y << ", "
                     << latest_grid_map_->info.origin.position.z << "]\n";
            yaml_out << "negate: 0\n";
            yaml_out << "occupied_thresh: 0.65\n";
            yaml_out << "free_thresh: 0.25\n";
            yaml_out.close();

            success_2d = true;
            res_msg += "[map.pgm, map.yaml]";
          }
        }
      }

      if (success_3d || success_2d) {
        res->success = true;
        res->message = res_msg;
        RCLCPP_INFO(this->get_logger(), "%s", res->message.c_str());
      } else {
        res->success = false;
        res->message = "Failed to save both 3D and 2D maps. No map data available.";
        RCLCPP_ERROR(this->get_logger(), "%s", res->message.c_str());
      }
    }

    void keyframeCallback(const nav_msgs::msg::Odometry::ConstSharedPtr& odom_msg, const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg) {
      std::lock_guard<std::mutex> lock(mtx_);

      Eigen::Matrix4d pose   = Eigen::Matrix4d::Identity();
      pose.block<3, 3>(0, 0) = Eigen::Quaterniond(odom_msg->pose.pose.orientation.w, odom_msg->pose.pose.orientation.x,
                                                  odom_msg->pose.pose.orientation.y, odom_msg->pose.pose.orientation.z)
                                   .toRotationMatrix();
      pose(0, 3)             = odom_msg->pose.pose.position.x;
      pose(1, 3)             = odom_msg->pose.pose.position.y;
      pose(2, 3)             = odom_msg->pose.pose.position.z;

      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::fromROSMsg(*cloud_msg, *cloud);

      pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
      voxel_filter.setLeafSize(global_voxel_size_, global_voxel_size_, global_voxel_size_);
      voxel_filter.setInputCloud(cloud);
      voxel_filter.filter(*downsampled_cloud);

      keyframe_raw_poses_.push_back(pose);
      keyframe_clouds_.push_back(downsampled_cloud);
    }

    void pathCallback(const nav_msgs::msg::Path::SharedPtr path_msg) {
      std::lock_guard<std::mutex> lock(mtx_);

      if (keyframe_clouds_.empty() || path_msg->poses.empty()) {
        return;
      }

      pcl::PointCloud<pcl::PointXYZ>::Ptr global_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      size_t num_frames = std::min(keyframe_clouds_.size(), path_msg->poses.size());

      std::vector<Eigen::Vector3d> frame_origins;
      std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> transformed_clouds;

      for (size_t i = 0; i < num_frames; ++i) {
        Eigen::Matrix4d opt_pose   = Eigen::Matrix4d::Identity();
        const auto& p              = path_msg->poses[i].pose;
        opt_pose.block<3, 3>(0, 0) = Eigen::Quaterniond(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z).toRotationMatrix();
        opt_pose(0, 3)             = p.position.x;
        opt_pose(1, 3)             = p.position.y;
        opt_pose(2, 3)             = p.position.z;

        Eigen::Vector3d origin(p.position.x, p.position.y, p.position.z);
        frame_origins.push_back(origin);

        Eigen::Matrix4d correction_transform = opt_pose * keyframe_raw_poses_[i].inverse();

        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*keyframe_clouds_[i], *transformed_cloud, correction_transform);

        transformed_clouds.push_back(transformed_cloud);
        *global_cloud += *transformed_cloud;
      }

      pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_global_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
      voxel_filter.setLeafSize(global_voxel_size_, global_voxel_size_, global_voxel_size_);
      voxel_filter.setInputCloud(global_cloud);
      voxel_filter.filter(*filtered_global_cloud);

      latest_global_cloud_ = filtered_global_cloud;

      sensor_msgs::msg::PointCloud2 map_msg;
      pcl::toROSMsg(*filtered_global_cloud, map_msg);
      map_msg.header = path_msg->header;
      global_map_pub_->publish(map_msg);

      publish2DGridMapWithBayes(transformed_clouds, frame_origins, path_msg->header);
    }

    void bresenhamRay(int x0, int y0, int x1, int y1, std::vector<std::pair<int, int>>& line_cells) {
      int dx  = std::abs(x1 - x0);
      int dy  = std::abs(y1 - y0);
      int sx  = (x0 < x1) ? 1 : -1;
      int sy  = (y0 < y1) ? 1 : -1;
      int err = dx - dy;

      int x = x0;
      int y = y0;

      while (true) {
        line_cells.push_back({x, y});
        if (x == x1 && y == y1) break;
        int e2 = 2 * err;
        if (e2 > -dy) {
          err -= dy;
          x += sx;
        }
        if (e2 < dx) {
          err += dx;
          y += sy;
        }
      }
    }

    void publish2DGridMapWithBayes(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& transformed_clouds,
                                   const std::vector<Eigen::Vector3d>& frame_origins, const std_msgs::msg::Header& header) {
      if (transformed_clouds.empty()) return;

      float min_x = std::numeric_limits<float>::max();
      float max_x = -std::numeric_limits<float>::max();
      float min_y = std::numeric_limits<float>::max();
      float max_y = -std::numeric_limits<float>::max();

      for (const auto& cloud : transformed_clouds) {
        for (const auto& pt : cloud->points) {
          if (pt.z < z_min_filter_ || pt.z > z_max_filter_) continue;
          if (pt.x < min_x) min_x = pt.x;
          if (pt.x > max_x) max_x = pt.x;
          if (pt.y < min_y) min_y = pt.y;
          if (pt.y > max_y) max_y = pt.y;
        }
      }

      min_x -= 1.0;
      max_x += 1.0;
      min_y -= 1.0;
      max_y += 1.0;

      int width  = std::ceil((max_x - min_x) / grid_resolution_);
      int height = std::ceil((max_y - min_y) / grid_resolution_);

      if (width <= 0 || height <= 0) return;

      std::vector<double> log_odds_map(width * height, 0.0);

      for (size_t i = 0; i < transformed_clouds.size(); ++i) {
        const auto& cloud  = transformed_clouds[i];
        const auto& origin = frame_origins[i];

        int origin_x = std::floor((origin.x() - min_x) / grid_resolution_);
        int origin_y = std::floor((origin.y() - min_y) / grid_resolution_);

        for (const auto& pt : cloud->points) {
          if (pt.z < z_min_filter_ || pt.z > z_max_filter_) continue;

          int pt_x = std::floor((pt.x - min_x) / grid_resolution_);
          int pt_y = std::floor((pt.y - min_y) / grid_resolution_);

          if (pt_x < 0 || pt_x >= width || pt_y < 0 || pt_y >= height) continue;

          std::vector<std::pair<int, int>> ray_cells;
          bresenhamRay(origin_x, origin_y, pt_x, pt_y, ray_cells);

          for (size_t j = 0; j + 1 < ray_cells.size(); ++j) {
            int cx = ray_cells[j].first;
            int cy = ray_cells[j].second;
            if (cx >= 0 && cx < width && cy >= 0 && cy < height) {
              int idx = cy * width + cx;
              log_odds_map[idx] += l_free_;
              if (log_odds_map[idx] < l_min_) log_odds_map[idx] = l_min_;
            }
          }

          int end_idx = pt_y * width + pt_x;
          if (end_idx >= 0 && end_idx < static_cast<int>(log_odds_map.size())) {
            log_odds_map[end_idx] += l_occ_;
            if (log_odds_map[end_idx] > l_max_) log_odds_map[end_idx] = l_max_;
          }
        }
      }

      nav_msgs::msg::OccupancyGrid grid;
      grid.header                    = header;
      grid.info.resolution           = grid_resolution_;
      grid.info.width                = width;
      grid.info.height               = height;
      grid.info.origin.position.x    = min_x;
      grid.info.origin.position.y    = min_y;
      grid.info.origin.position.z    = 0.0;
      grid.info.origin.orientation.w = 1.0;

      grid.data.resize(width * height);

      for (size_t i = 0; i < log_odds_map.size(); ++i) {
        double l = log_odds_map[i];
        if (std::abs(l) < 1e-5) {
          grid.data[i] = -1;
        } else {
          double prob  = 1.0 / (1.0 + std::exp(-l));
          int val      = static_cast<int>(prob * 100.0);
          grid.data[i] = std::clamp(val, 0, 100);
        }
      }

      latest_grid_map_ = std::make_shared<nav_msgs::msg::OccupancyGrid>(grid);
      grid_map_pub_->publish(grid);
    }
  };

} // namespace simple_slam

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(simple_slam::MapBuilder)