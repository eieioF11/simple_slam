#pragma once

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

// Message Filters (同期用)
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>

// PCL
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

// extention node
#include "extension_node/extension_node.hpp"

// Eigen
#include <Eigen/Dense>

#include <mutex>
#include <vector>

namespace simple_slam {

  class MapBuilder : public ext_rclcpp::ExtensionNode {
  public:
    MapBuilder(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : ext_rclcpp::ExtensionNode("map_builder", options) {

      RCLCPP_INFO(this->get_logger(), "Starting Map Builder Node...");

      // パラメータ
      global_voxel_size_ = param<double>("map_builder.global_voxel_size", 0.2);
      grid_resolution_   = param<double>("map_builder.grid_resolution", 0.1);
      z_min_filter_      = param<double>("map_builder.z_min_filter", 0.1); // 床面を除去するためのZ下限
      z_max_filter_      = param<double>("map_builder.z_max_filter", 1.5); // 天井を除去するためのZ上限

      // Publishers
      global_map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("map_builder/global_map", rclcpp::QoS(1).transient_local());
      grid_map_pub_   = this->create_publisher<nav_msgs::msg::OccupancyGrid>("map", rclcpp::QoS(1).transient_local());

      // Subscribers
      // PGOからの最適化済みPathを受信
      path_sub_ = this->create_subscription<nav_msgs::msg::Path>("optimized_path", rclcpp::QoS(10),
                                                                 std::bind(&MapBuilder::pathCallback, this, std::placeholders::_1));

      // ScanMatcherからのキーフレーム情報を受信（同期）
      odom_sub_.subscribe(this, "scan_matcher/keyframe_odom");
      cloud_sub_.subscribe(this, "scan_matcher/out_points");

      sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), odom_sub_, cloud_sub_);
      sync_->registerCallback(std::bind(&MapBuilder::keyframeCallback, this, std::placeholders::_1, std::placeholders::_2));
    }

  private:
    double global_voxel_size_;
    double grid_resolution_;
    double z_min_filter_;
    double z_max_filter_;

    std::mutex mtx_;

    // キーフレームのデータを保持
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> keyframe_clouds_;
    std::vector<Eigen::Matrix4d> keyframe_raw_poses_;

    // 同期用の型定義
    using SyncPolicy = message_filters::sync_policies::ExactTime<nav_msgs::msg::Odometry, sensor_msgs::msg::PointCloud2>;
    message_filters::Subscriber<nav_msgs::msg::Odometry> odom_sub_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> cloud_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr global_map_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_map_pub_;

    // 1. キーフレームの保存
    void keyframeCallback(const nav_msgs::msg::Odometry::ConstSharedPtr& odom_msg, const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg) {
      std::lock_guard<std::mutex> lock(mtx_);

      // オドメトリ（ScanMatcher実行直後のPose）を保存
      Eigen::Matrix4d pose   = Eigen::Matrix4d::Identity();
      pose.block<3, 3>(0, 0) = Eigen::Quaterniond(odom_msg->pose.pose.orientation.w, odom_msg->pose.pose.orientation.x,
                                                  odom_msg->pose.pose.orientation.y, odom_msg->pose.pose.orientation.z)
                                   .toRotationMatrix();
      pose(0, 3)             = odom_msg->pose.pose.position.x;
      pose(1, 3)             = odom_msg->pose.pose.position.y;
      pose(2, 3)             = odom_msg->pose.pose.position.z;

      // 点群を保存（メモリ節約のためダウンサンプリング）
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

    // 2. 最適化済みPathを受信してマップを再構築
    void pathCallback(const nav_msgs::msg::Path::SharedPtr path_msg) {
      std::lock_guard<std::mutex> lock(mtx_);

      if (keyframe_clouds_.empty() || path_msg->poses.empty()) {
        return;
      }

      pcl::PointCloud<pcl::PointXYZ>::Ptr global_cloud(new pcl::PointCloud<pcl::PointXYZ>);

      // Pathに含まれる各Poseを使って、対応するキーフレーム点群を補正・結合
      size_t num_frames = std::min(keyframe_clouds_.size(), path_msg->poses.size());

      for (size_t i = 0; i < num_frames; ++i) {
        // PGOによって最適化されたPose
        Eigen::Matrix4d opt_pose   = Eigen::Matrix4d::Identity();
        const auto& p              = path_msg->poses[i].pose;
        opt_pose.block<3, 3>(0, 0) = Eigen::Quaterniond(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z).toRotationMatrix();
        opt_pose(0, 3)             = p.position.x;
        opt_pose(1, 3)             = p.position.y;
        opt_pose(2, 3)             = p.position.z;

        // 元の点群はScanMatcherのローカルマップ基準（keyframe_raw_poses_[i]）で登録されているため、
        // 元のPoseの逆行列を掛けてローカル座標に戻してから、「最適化Pose」を掛けて新しい位置に配置する。
        Eigen::Matrix4d correction_transform = opt_pose * keyframe_raw_poses_[i].inverse();

        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*keyframe_clouds_[i], *transformed_cloud, correction_transform);

        *global_cloud += *transformed_cloud;
      }

      // グローバルマップ全体のダウンサンプリング
      pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_global_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
      voxel_filter.setLeafSize(global_voxel_size_, global_voxel_size_, global_voxel_size_);
      voxel_filter.setInputCloud(global_cloud);
      voxel_filter.filter(*filtered_global_cloud);

      // 3Dマップのパブリッシュ
      sensor_msgs::msg::PointCloud2 map_msg;
      pcl::toROSMsg(*filtered_global_cloud, map_msg);
      map_msg.header = path_msg->header;
      global_map_pub_->publish(map_msg);

      // 2D Occupancy Grid Mapの生成とパブリッシュ
      publish2DGridMap(filtered_global_cloud, path_msg->header);
    }

    // 3. 2D Occupancy Grid の生成
    void publish2DGridMap(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std_msgs::msg::Header& header) {
      if (cloud->empty()) return;

      // 指定したZ座標（高さ）の範囲にある点（障害物）だけを抽出
      pcl::PointCloud<pcl::PointXYZ>::Ptr sliced_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PassThrough<pcl::PointXYZ> pass;
      pass.setInputCloud(cloud);
      pass.setFilterFieldName("z");
      pass.setFilterLimits(z_min_filter_, z_max_filter_);
      pass.filter(*sliced_cloud);

      // マップのXYの最小・最大値を計算してグリッドのサイズを決定
      float min_x = std::numeric_limits<float>::max();
      float max_x = -std::numeric_limits<float>::max();
      float min_y = std::numeric_limits<float>::max();
      float max_y = -std::numeric_limits<float>::max();

      for (const auto& pt : sliced_cloud->points) {
        if (pt.x < min_x) min_x = pt.x;
        if (pt.x > max_x) max_x = pt.x;
        if (pt.y < min_y) min_y = pt.y;
        if (pt.y > max_y) max_y = pt.y;
      }

      int width  = std::ceil((max_x - min_x) / grid_resolution_);
      int height = std::ceil((max_y - min_y) / grid_resolution_);

      if (width <= 0 || height <= 0) return;

      nav_msgs::msg::OccupancyGrid grid;
      grid.header                    = header;
      grid.info.resolution           = grid_resolution_;
      grid.info.width                = width;
      grid.info.height               = height;
      grid.info.origin.position.x    = min_x;
      grid.info.origin.position.y    = min_y;
      grid.info.origin.position.z    = 0.0;
      grid.info.origin.orientation.w = 1.0;

      // 初期値は 0 (Free Space)
      grid.data.assign(width * height, 0);

      // 点群が存在するセルを 100 (Obstacle) に設定
      for (const auto& pt : sliced_cloud->points) {
        int grid_x = std::floor((pt.x - min_x) / grid_resolution_);
        int grid_y = std::floor((pt.y - min_y) / grid_resolution_);

        if (grid_x >= 0 && grid_x < width && grid_y >= 0 && grid_y < height) {
          int index        = grid_y * width + grid_x;
          grid.data[index] = 100; // 障害物
        }
      }

      grid_map_pub_->publish(grid);
    }
  };

} // namespace simple_slam

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(simple_slam::MapBuilder)