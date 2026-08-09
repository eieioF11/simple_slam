#pragma once

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

// extention node
#include "extension_node/extension_node.hpp"

// Message Filters (同期用)
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>

// PCL
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>

// Eigen
#include <Eigen/Dense>

#include <mutex>
#include <string>
#include <vector>

namespace simple_slam {

  // キーフレームのデータを保持する構造体
  struct KeyFrame {
    int id;
    Eigen::Matrix4d pose;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
  };

  class LoopClosure : public ext_rclcpp::ExtensionNode {
  public:
    LoopClosure(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : ext_rclcpp::ExtensionNode("loop_closure", options) {

      RCLCPP_INFO(this->get_logger(), "Starting Loop Closure Node...");

      search_radius_        = param<double>("loop_closure.search_radius", 3.0);
      skip_recent_frames_   = param<int>("loop_closure.skip_recent_frames", 50);
      fitness_score_thresh_ = param<double>("loop_closure.fitness_score_thresh", 0.3);
      voxel_size_           = param<double>("loop_closure.voxel_size", 0.3);

      // Publisher (Pose Graph Optimizer へ送信する制約)
      loop_constraint_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("loop_closure/constraint", rclcpp::QoS(10));

      // Subscribers (OdometryとPointCloud2を同期して受信)
      odom_sub_.subscribe(this, "scan_matcher/keyframe_odom");
      cloud_sub_.subscribe(this, "scan_matcher/local_map");
      // cloud_sub_.subscribe(this, "scan_matcher/out_points");

      sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), odom_sub_, cloud_sub_);
      sync_->registerCallback(std::bind(&LoopClosure::syncCallback, this, std::placeholders::_1, std::placeholders::_2));
    }

  private:
    double search_radius_;
    int skip_recent_frames_;
    double fitness_score_thresh_;
    double voxel_size_;

    std::vector<KeyFrame> keyframes_;
    std::mutex mtx_;
    int current_id_ = 0;

    // 同期用の型定義
    using SyncPolicy = message_filters::sync_policies::ExactTime<nav_msgs::msg::Odometry, sensor_msgs::msg::PointCloud2>;
    message_filters::Subscriber<nav_msgs::msg::Odometry> odom_sub_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> cloud_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr loop_constraint_pub_;

    void syncCallback(const nav_msgs::msg::Odometry::ConstSharedPtr& odom_msg, const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg) {

      std::lock_guard<std::mutex> lock(mtx_);

      // 1. データの変換
      Eigen::Matrix4d current_pose   = Eigen::Matrix4d::Identity();
      current_pose.block<3, 3>(0, 0) = Eigen::Quaterniond(odom_msg->pose.pose.orientation.w, odom_msg->pose.pose.orientation.x,
                                                          odom_msg->pose.pose.orientation.y, odom_msg->pose.pose.orientation.z)
                                           .toRotationMatrix();
      current_pose(0, 3)             = odom_msg->pose.pose.position.x;
      current_pose(1, 3)             = odom_msg->pose.pose.position.y;
      current_pose(2, 3)             = odom_msg->pose.pose.position.z;

      pcl::PointCloud<pcl::PointXYZ>::Ptr current_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::fromROSMsg(*cloud_msg, *current_cloud);

      // ダウンサンプリングして保持（メモリと計算コストの削減）
      pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
      voxel_filter.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
      voxel_filter.setInputCloud(current_cloud);
      voxel_filter.filter(*downsampled_cloud);

      // キーフレームとして履歴に追加
      KeyFrame kf;
      kf.id    = current_id_++;
      kf.pose  = current_pose;
      kf.cloud = downsampled_cloud;
      keyframes_.push_back(kf);

      // 2. ループ探索 (過去のキーフレームとの距離を比較)
      if (keyframes_.size() < skip_recent_frames_ + 1) {
        return; // 履歴が少ない場合はスキップ
      }

      int loop_candidate_id = -1;
      double min_dist       = search_radius_;

      Eigen::Vector3d current_pos = current_pose.block<3, 1>(0, 3);

      // 直近のフレームは探索対象から外す
      for (size_t i = 0; i < keyframes_.size() - skip_recent_frames_; ++i) {
        Eigen::Vector3d past_pos = keyframes_[i].pose.block<3, 1>(0, 3);
        double dist              = (current_pos - past_pos).norm();

        if (dist < min_dist) {
          min_dist          = dist;
          loop_candidate_id = keyframes_[i].id;
        }
      }

      // 3. ループ候補が見つかった場合、ICPマッチングで検証
      if (loop_candidate_id != -1) {
        verifyLoopClosure(loop_candidate_id, kf, odom_msg->header.stamp);
      }
    }

    void verifyLoopClosure(int target_id, const KeyFrame& current_kf, const rclcpp::Time& stamp) {

      pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud = keyframes_[target_id].cloud;
      pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud = current_kf.cloud;

      // GICPによる位置合わせ
      pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
      gicp.setInputTarget(target_cloud);
      gicp.setInputSource(source_cloud);
      gicp.setMaximumIterations(50);
      gicp.setMaxCorrespondenceDistance(search_radius_ * 2.0);

      // ICPの初期値（Initial Guess）として、オドメトリ上の相対ポーズを与える
      Eigen::Matrix4d initial_guess = keyframes_[target_id].pose.inverse() * current_kf.pose;

      pcl::PointCloud<pcl::PointXYZ> aligned_cloud;
      gicp.align(aligned_cloud, initial_guess.cast<float>());

      if (gicp.hasConverged() && gicp.getFitnessScore() < fitness_score_thresh_) {

        // 最適化された相対ポーズを取得
        Eigen::Matrix4d relative_pose = gicp.getFinalTransformation().cast<double>();

        RCLCPP_INFO(this->get_logger(), "Loop Closure Detected! %d -> %d (Score: %f)", current_kf.id, target_id, gicp.getFitnessScore());

        publishLoopConstraint(target_id, relative_pose, stamp);
      }
    }

    void publishLoopConstraint(int target_id, const Eigen::Matrix4d& relative_pose, const rclcpp::Time& stamp) {
      nav_msgs::msg::Odometry constraint_msg;
      constraint_msg.header.stamp = stamp;

      // PGOノード側で読み取るため、frame_idにターゲットIDを文字列として仕込む
      constraint_msg.header.frame_id = std::to_string(target_id);
      constraint_msg.child_frame_id  = "loop_constraint";

      constraint_msg.pose.pose.position.x = relative_pose(0, 3);
      constraint_msg.pose.pose.position.y = relative_pose(1, 3);
      constraint_msg.pose.pose.position.z = relative_pose(2, 3);

      Eigen::Matrix3d R = relative_pose.block<3, 3>(0, 0);
      Eigen::Quaterniond q(R);
      constraint_msg.pose.pose.orientation.x = q.x();
      constraint_msg.pose.pose.orientation.y = q.y();
      constraint_msg.pose.pose.orientation.z = q.z();
      constraint_msg.pose.pose.orientation.w = q.w();

      loop_constraint_pub_->publish(constraint_msg);
    }
  };

} // namespace simple_slam

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(simple_slam::LoopClosure)