#pragma once

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

// extention node
#include "extension_node/extension_node.hpp"

// Message Filters
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

// PCL
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>

// small GICP
#define USE_SMALL_GICP
#include <small_gicp/pcl/pcl_registration.hpp>

// Eigen
#include <Eigen/Dense>

#include <mutex>
#include <string>
#include <vector>

namespace simple_slam {

  struct KeyFrame {
    int id;
    Eigen::Matrix4d pose;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
  };

  class LoopClosure : public ext_rclcpp::ExtensionNode {
  public:
    LoopClosure(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : ext_rclcpp::ExtensionNode("loop_closure", options) {

      RCLCPP_INFO(this->get_logger(), "Starting Loop Closure Node...");

      search_radius_      = param<double>("loop_closure.search_radius", 3.0);
      skip_recent_frames_ = param<int>("loop_closure.skip_recent_frames", 50);

      // 似た環境での誤検知を防ぐため、デフォルトのフィットネススコアを0.15に厳格化
      fitness_score_thresh_ = param<double>("loop_closure.fitness_score_thresh", 0.15);
      voxel_size_           = param<double>("loop_closure.voxel_size", 0.3);

      GICP_MAX_ITERATIONS             = param<int>("loop_closure.gicp.max_iterations", 50);
      SGICP_NUM_THREADS               = param<int>("loop_closure.sgicp.num_threads", 4);
      SGICP_CORRESPONDENCE_RANDOMNESS = param<int>("loop_closure.sgicp.correspondence_randomness", 20);
      SGICP_VOXEL_RESOLUTION          = param<double>("loop_closure.sgicp.voxel_resolution", 1.0);
      SGICP_REGISTRATION_TYPE         = param<std::string>("loop_closure.sgicp.registration_type", "VGICP");

      loop_constraint_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("loop_closure/constraint", rclcpp::QoS(10));

      odom_sub_.subscribe(this, "scan_matcher/keyframe_odom");
      cloud_sub_.subscribe(this, "scan_matcher/out_points");

      sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(50), odom_sub_, cloud_sub_);
      sync_->registerCallback(std::bind(&LoopClosure::syncCallback, this, std::placeholders::_1, std::placeholders::_2));
    }

  private:
    double search_radius_;
    int skip_recent_frames_;
    double fitness_score_thresh_;
    double voxel_size_;

    int GICP_MAX_ITERATIONS;
    int SGICP_NUM_THREADS;
    int SGICP_CORRESPONDENCE_RANDOMNESS;
    double SGICP_VOXEL_RESOLUTION;
    std::string SGICP_REGISTRATION_TYPE;

    std::vector<KeyFrame> keyframes_;
    std::mutex mtx_;
    int current_id_ = 0;

    // ループIDの連続性チェック用履歴
    int last_matched_current_id_ = -1;
    int last_matched_target_id_  = -1;

    using SyncPolicy = message_filters::sync_policies::ApproximateTime<nav_msgs::msg::Odometry, sensor_msgs::msg::PointCloud2>;
    message_filters::Subscriber<nav_msgs::msg::Odometry> odom_sub_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> cloud_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr loop_constraint_pub_;

    void syncCallback(const nav_msgs::msg::Odometry::ConstSharedPtr& odom_msg, const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg) {
      std::lock_guard<std::mutex> lock(mtx_);

      Eigen::Matrix4d current_pose   = Eigen::Matrix4d::Identity();
      current_pose.block<3, 3>(0, 0) = Eigen::Quaterniond(odom_msg->pose.pose.orientation.w, odom_msg->pose.pose.orientation.x,
                                                          odom_msg->pose.pose.orientation.y, odom_msg->pose.pose.orientation.z)
                                           .toRotationMatrix();
      current_pose(0, 3)             = odom_msg->pose.pose.position.x;
      current_pose(1, 3)             = odom_msg->pose.pose.position.y;
      current_pose(2, 3)             = odom_msg->pose.pose.position.z;

      pcl::PointCloud<pcl::PointXYZ>::Ptr current_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::fromROSMsg(*cloud_msg, *current_cloud);

      pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
      voxel_filter.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
      voxel_filter.setInputCloud(current_cloud);
      voxel_filter.filter(*downsampled_cloud);

      KeyFrame kf;
      kf.id    = current_id_++;
      kf.pose  = current_pose;
      kf.cloud = downsampled_cloud;
      keyframes_.push_back(kf);

      if (keyframes_.size() < skip_recent_frames_ + 1) {
        return;
      }

      int loop_candidate_id       = -1;
      double min_dist             = search_radius_;
      Eigen::Vector3d current_pos = current_pose.block<3, 1>(0, 3);

      for (size_t i = 0; i < keyframes_.size() - skip_recent_frames_; ++i) {
        Eigen::Vector3d past_pos = keyframes_[i].pose.block<3, 1>(0, 3);
        double dist_2d           = (current_pos.head<2>() - past_pos.head<2>()).norm();

        if (dist_2d < min_dist) {
          min_dist          = dist_2d;
          loop_candidate_id = keyframes_[i].id;
        }
      }

      if (loop_candidate_id != -1) {
        verifyLoopClosure(loop_candidate_id, kf, odom_msg->header.stamp);
      }
    }

    void verifyLoopClosure(int target_id, const KeyFrame& current_kf, const rclcpp::Time& stamp) {
      pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud = keyframes_[target_id].cloud;
      pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud = current_kf.cloud;
#ifdef USE_SMALL_GICP
      small_gicp::RegistrationPCL<pcl::PointXYZ, pcl::PointXYZ> gicp;
      gicp.setInputTarget(target_cloud);
      gicp.setInputSource(source_cloud);
      gicp.setMaximumIterations(GICP_MAX_ITERATIONS);
      gicp.setNumThreads(SGICP_NUM_THREADS);
      gicp.setCorrespondenceRandomness(SGICP_CORRESPONDENCE_RANDOMNESS);
      gicp.setMaxCorrespondenceDistance(search_radius_ * 2.0);
      gicp.setVoxelResolution(SGICP_VOXEL_RESOLUTION);
      gicp.setRegistrationType(SGICP_REGISTRATION_TYPE);
#else
      pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
      gicp.setInputTarget(target_cloud);
      gicp.setInputSource(source_cloud);
      gicp.setMaximumIterations(GICP_MAX_ITERATIONS);
      gicp.setMaxCorrespondenceDistance(search_radius_ * 2.0);
#endif

      Eigen::Matrix4d initial_guess = keyframes_[target_id].pose.inverse() * current_kf.pose;

      pcl::PointCloud<pcl::PointXYZ> aligned_cloud;
      gicp.align(aligned_cloud, initial_guess.cast<float>());

      if (gicp.hasConverged() && gicp.getFitnessScore() < fitness_score_thresh_) {
        Eigen::Matrix4d relative_pose = gicp.getFinalTransformation().cast<double>();

        // 1. Z軸の異常なズレを弾く (階段でない限り1.0m以上の高さ変化は異常)
        double z_diff = std::abs(relative_pose(2, 3));
        if (z_diff > 1.0) {
          RCLCPP_WARN(this->get_logger(), "Rejected Loop Closure: Large Z diff (%f m)", z_diff);
          return;
        }

        // 2. ICPの移動量制限 (探索半径以上に強引に引き寄せた場合は別の部屋の可能性大)
        Eigen::Matrix4d diff = initial_guess.inverse() * relative_pose;
        double delta_trans   = diff.block<3, 1>(0, 3).norm();
        if (delta_trans > search_radius_) {
          RCLCPP_WARN(this->get_logger(), "Rejected Loop Closure: ICP moved too far (%f m)", delta_trans);
          return;
        }

        // ※連続性チェック（ジャンプ防止）は副作用が強いため削除しました。

        RCLCPP_INFO(this->get_logger(), "Loop Closure Detected! %d -> %d (Score: %f)", current_kf.id, target_id, gicp.getFitnessScore());

        publishLoopConstraint(target_id, relative_pose, stamp);
      }
    }

    void publishLoopConstraint(int target_id, const Eigen::Matrix4d& relative_pose, const rclcpp::Time& stamp) {
      nav_msgs::msg::Odometry constraint_msg;
      constraint_msg.header.stamp    = stamp;
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