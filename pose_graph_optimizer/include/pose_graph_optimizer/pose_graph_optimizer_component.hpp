#pragma once

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// extention node
#include "extension_node/extension_node.hpp"

// GTSAM
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

// Iridescence
#include <glk/indexed_pointcloud_buffer.hpp>
#include <glk/primitives/primitives.hpp>
#include <guik/spdlog_sink.hpp>
#include <guik/viewer/light_viewer.hpp>
#include <spdlog/fmt/ostr.h>
#include <spdlog/sinks/ringbuffer_sink.h>
#include <spdlog/spdlog.h>

#include <mutex>
#include <vector>

using gtsam::symbol_shorthand::B; // IMU Bias (ConstantBias)
using gtsam::symbol_shorthand::V; // Velocity (Vector3)
using gtsam::symbol_shorthand::X; // Pose3

namespace simple_slam {

  class PoseGraphOptimizer : public ext_rclcpp::ExtensionNode {
  public:
    PoseGraphOptimizer(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : ext_rclcpp::ExtensionNode("pose_graph_optimizer", options) {

      use_imu_ = param<bool>("pose_graph_optimizer.use_imu", false);

      if (use_imu_) {
        RCLCPP_INFO(this->get_logger(), "Starting GTSAM Pose Graph Optimizer WITH IMU (Tightly-coupled)...");
      } else {
        RCLCPP_INFO(this->get_logger(), "Starting GTSAM Pose Graph Optimizer WITHOUT IMU (LiDAR Odometry Only)...");
      }

      logger_ = spdlog::get("pose_graph_optimizer_logger");

      // ISAM2のパラメータ設定
      gtsam::ISAM2Params parameters;
      parameters.relinearizeThreshold = param<double>("pose_graph_optimizer.isam2.relinearizeThreshold", 0.1);
      parameters.relinearizeSkip      = param<int>("pose_graph_optimizer.isam2.relinearizeSkip", 1);
      ISAM2_UPDATE_NUM                = param<int>("pose_graph_optimizer.isam2.update_num", 3);
      isam2_                          = std::make_unique<gtsam::ISAM2>(parameters);

      // ノイズモデルの定義
      prior_noise_ = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
      odom_noise_  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3).finished());
      loop_noise_  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 5e-2, 5e-2, 5e-2, 1e-2, 1e-2, 1e-2).finished());

      if (use_imu_) {
        prior_vel_noise_  = gtsam::noiseModel::Isotropic::Sigma(3, 1e-2);
        prior_bias_noise_ = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);
        bias_rw_noise_    = gtsam::noiseModel::Isotropic::Sigma(6, 1e-4);

        auto preint_params                     = gtsam::PreintegrationParams::MakeSharedU(9.81);
        preint_params->accelerometerCovariance = gtsam::Matrix33::Identity() * std::pow(0.1, 2);
        preint_params->gyroscopeCovariance     = gtsam::Matrix33::Identity() * std::pow(0.1, 2);
        preint_params->integrationCovariance   = gtsam::Matrix33::Identity() * std::pow(1e-4, 2);

        prev_bias_         = gtsam::imuBias::ConstantBias();
        imu_preintegrated_ = std::make_shared<gtsam::PreintegratedImuMeasurements>(preint_params, prev_bias_);

        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>("imu/data", rclcpp::QoS(500),
                                                                    std::bind(&PoseGraphOptimizer::imuCallback, this, std::placeholders::_1));
      }

      // Subscribers
      odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("scan_matcher/keyframe_odom", rclcpp::QoS(10),
                                                                     std::bind(&PoseGraphOptimizer::odomCallback, this, std::placeholders::_1));

      loop_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
          "loop_closure/constraint", rclcpp::QoS(10), std::bind(&PoseGraphOptimizer::loopClosureCallback, this, std::placeholders::_1));

      // Publishers
      path_pub_  = this->create_publisher<nav_msgs::msg::Path>("optimized_path", rclcpp::QoS(1));
      graph_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("factor_graph_markers", rclcpp::QoS(1));
    }

  private:
    bool use_imu_;
    int ISAM2_UPDATE_NUM;
    std::unique_ptr<gtsam::ISAM2> isam2_;
    gtsam::NonlinearFactorGraph gtSAMgraph_;
    gtsam::Values initialEstimate_;
    gtsam::Values currentEstimate_;
    std::shared_ptr<spdlog::logger> logger_;

    std::mutex mtx_;
    int keyframe_id_ = 0;

    // 状態管理
    gtsam::Pose3 last_pose_;
    gtsam::Vector3 prev_velocity_ = gtsam::Vector3::Zero();
    gtsam::imuBias::ConstantBias prev_bias_;
    std::shared_ptr<gtsam::PreintegratedImuMeasurements> imu_preintegrated_;
    double last_imu_time_ = -1.0;

    // Noise Models
    gtsam::noiseModel::Diagonal::shared_ptr prior_noise_;
    gtsam::noiseModel::Diagonal::shared_ptr odom_noise_;
    gtsam::noiseModel::Diagonal::shared_ptr loop_noise_;
    gtsam::noiseModel::Isotropic::shared_ptr prior_vel_noise_;
    gtsam::noiseModel::Isotropic::shared_ptr prior_bias_noise_;
    gtsam::noiseModel::Isotropic::shared_ptr bias_rw_noise_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr loop_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr graph_pub_;

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
      if (!use_imu_) return;

      std::lock_guard<std::mutex> lock(mtx_);
      double current_time = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

      if (last_imu_time_ < 0) {
        last_imu_time_ = current_time;
        return;
      }
      double dt      = current_time - last_imu_time_;
      last_imu_time_ = current_time;

      gtsam::Vector3 acc(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
      gtsam::Vector3 gyro(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

      imu_preintegrated_->integrateMeasurement(acc, gyro, dt);
    }

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(mtx_);

      gtsam::Pose3 current_pose = gtsam::Pose3(gtsam::Rot3::Quaternion(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                                                                       msg->pose.pose.orientation.y, msg->pose.pose.orientation.z),
                                               gtsam::Point3(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z));

      if (keyframe_id_ == 0) {
        // 初期ノードの Prior 登録
        gtSAMgraph_.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), current_pose, prior_noise_));
        initialEstimate_.insert(X(0), current_pose);

        if (use_imu_) {
          gtSAMgraph_.add(gtsam::PriorFactor<gtsam::Vector3>(V(0), prev_velocity_, prior_vel_noise_));
          gtSAMgraph_.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(B(0), prev_bias_, prior_bias_noise_));
          initialEstimate_.insert(V(0), prev_velocity_);
          initialEstimate_.insert(B(0), prev_bias_);
        }
      } else {
        // LiDAR Odometry Factor (BetweenFactor)
        gtsam::Pose3 relative_pose = last_pose_.between(current_pose);
        gtSAMgraph_.add(gtsam::BetweenFactor<gtsam::Pose3>(X(keyframe_id_ - 1), X(keyframe_id_), relative_pose, odom_noise_));

        if (use_imu_) {
          // IMU Factor の追加
          gtsam::ImuFactor imu_factor(X(keyframe_id_ - 1), V(keyframe_id_ - 1), X(keyframe_id_), V(keyframe_id_), B(keyframe_id_ - 1),
                                      *imu_preintegrated_);
          gtSAMgraph_.add(imu_factor);

          // IMUバイアス変化のランダムウォーク制約
          gtSAMgraph_.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(keyframe_id_ - 1), B(keyframe_id_), gtsam::imuBias::ConstantBias(),
                                                                             bias_rw_noise_));

          // IMUを用いた予測値で初期推定
          gtsam::NavState prev_state(last_pose_, prev_velocity_);
          gtsam::NavState prop_state = imu_preintegrated_->predict(prev_state, prev_bias_);

          initialEstimate_.insert(X(keyframe_id_), current_pose);
          initialEstimate_.insert(V(keyframe_id_), prop_state.velocity());
          initialEstimate_.insert(B(keyframe_id_), prev_bias_);
        } else {
          // IMUを使わない場合はLiDARの結果だけをそのまま挿入
          initialEstimate_.insert(X(keyframe_id_), current_pose);
        }
      }

      last_pose_ = current_pose;

      isam2_->update(gtSAMgraph_, initialEstimate_);
      for (int i = 0; i < ISAM2_UPDATE_NUM; ++i)
        isam2_->update();

      gtSAMgraph_.resize(0);
      initialEstimate_.clear();

      currentEstimate_ = isam2_->calculateEstimate();

      // IMU使用時のみ、バイアスと速度の更新処理を行う
      if (use_imu_) {
        prev_velocity_ = currentEstimate_.at<gtsam::Vector3>(V(keyframe_id_));
        prev_bias_     = currentEstimate_.at<gtsam::imuBias::ConstantBias>(B(keyframe_id_));
        imu_preintegrated_->resetIntegrationAndSetBias(prev_bias_);
      }

      keyframe_id_++;

      publishOptimizedPath(msg->header.stamp);
    }

    void loopClosureCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(mtx_);
      int loop_target_id = std::stoi(msg->header.frame_id);
      int current_id     = keyframe_id_ - 1;

      if (loop_target_id >= current_id) return;

      gtsam::Pose3 loop_relative_pose = gtsam::Pose3(gtsam::Rot3::Quaternion(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                                                                             msg->pose.pose.orientation.y, msg->pose.pose.orientation.z),
                                                     gtsam::Point3(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z));

      gtSAMgraph_.add(gtsam::BetweenFactor<gtsam::Pose3>(X(loop_target_id), X(current_id), loop_relative_pose, loop_noise_));

      RCLCPP_INFO(this->get_logger(), "Loop Closure Added: %d -> %d", current_id, loop_target_id);

      isam2_->update(gtSAMgraph_, initialEstimate_);
      for (int i = 0; i < ISAM2_UPDATE_NUM; ++i)
        isam2_->update();

      gtSAMgraph_.resize(0);
      initialEstimate_.clear();

      currentEstimate_ = isam2_->calculateEstimate();
      publishOptimizedPath(msg->header.stamp);
    }

    void publishOptimizedPath(const rclcpp::Time& stamp) {
      nav_msgs::msg::Path path_msg;
      path_msg.header.stamp    = stamp;
      path_msg.header.frame_id = "map";

      for (int i = 0; i < keyframe_id_; ++i) {
        if (currentEstimate_.exists(X(i))) {
          gtsam::Pose3 pose = currentEstimate_.at<gtsam::Pose3>(X(i));
          geometry_msgs::msg::PoseStamped pose_stamped;
          pose_stamped.header = path_msg.header;

          pose_stamped.pose.position.x = pose.x();
          pose_stamped.pose.position.y = pose.y();
          pose_stamped.pose.position.z = pose.z();

          auto q                          = pose.rotation().toQuaternion();
          pose_stamped.pose.orientation.w = q.w();
          pose_stamped.pose.orientation.x = q.x();
          pose_stamped.pose.orientation.y = q.y();
          pose_stamped.pose.orientation.z = q.z();

          path_msg.poses.push_back(pose_stamped);
        }
      }
      path_pub_->publish(path_msg);
    }

} // namespace simple_slam

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(simple_slam::PoseGraphOptimizer)