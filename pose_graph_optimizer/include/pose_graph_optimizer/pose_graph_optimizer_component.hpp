#pragma once

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

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

#include <deque>
#include <map>
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
        RCLCPP_INFO(this->get_logger(), "Starting GTSAM Pose Graph Optimizer WITHOUT IMU (LiOdometry Only)...");
      }

      logger_ = spdlog::get("pose_graph_optimizer_logger");

      gtsam::ISAM2Params parameters;
      parameters.relinearizeThreshold = param<double>("pose_graph_optimizer.isam2.relinearizeThreshold", 0.1);
      parameters.relinearizeSkip      = param<int>("pose_graph_optimizer.isam2.relinearizeSkip", 1);
      ISAM2_UPDATE_NUM                = param<int>("pose_graph_optimizer.isam2.update_num", 3);
      isam2_                          = std::make_unique<gtsam::ISAM2>(parameters);

      prior_noise_ = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
      odom_noise_  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3).finished());

      auto base_loop_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-1, 1e-1, 1e-1, 5e-2, 5e-2, 5e-2).finished());
      auto huber           = gtsam::noiseModel::mEstimator::Huber::Create(1.0);
      loop_noise_          = gtsam::noiseModel::Robust::Create(huber, base_loop_noise);

      if (use_imu_) {
        prior_vel_noise_  = gtsam::noiseModel::Isotropic::Sigma(3, 1e-2);
        prior_bias_noise_ = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);
        bias_rw_noise_    = gtsam::noiseModel::Isotropic::Sigma(6, 1e-4);

        auto preint_params                     = gtsam::PreintegrationParams::MakeSharedU(9.81);
        preint_params->accelerometerCovariance = gtsam::Matrix33::Identity() * std::pow(0.1, 2);
        preint_params->gyroscopeCovariance     = gtsam::Matrix33::Identity() * std::pow(0.1, 2);
        preint_params->integrationCovariance   = gtsam::Matrix33::Identity() * std::pow(1e-4, 2);

        // GLIMフォーマット: TUM形式 [x, y, z, qx, qy, qz, qw]
        std::vector<double> ext    = param<std::vector<double>>("pose_graph_optimizer.T_lidar_imu", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0});
        gtsam::Pose3 body_P_sensor = gtsam::Pose3();

        if (ext.size() == 7) {
          // GTSAMのQuaternionコンストラクタは (w, x, y, z) の順で受け取るためインデックスを調整
          body_P_sensor = gtsam::Pose3(gtsam::Rot3::Quaternion(ext[6], ext[3], ext[4], ext[5]), gtsam::Point3(ext[0], ext[1], ext[2]));
        } else {
          RCLCPP_WARN(this->get_logger(), "Invalid T_lidar_imu size (expected 7 for [x,y,z,qx,qy,qz,qw]). Using identity.");
        }

        preint_params->setBodyPSensor(body_P_sensor);

        prev_bias_         = gtsam::imuBias::ConstantBias();
        imu_preintegrated_ = std::make_shared<gtsam::PreintegratedImuMeasurements>(preint_params, prev_bias_);

        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>("imu", rclcpp::QoS(500),
                                                                    std::bind(&PoseGraphOptimizer::imuCallback, this, std::placeholders::_1));
      }

      odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("scan_matcher/keyframe_odom", rclcpp::QoS(10),
                                                                     std::bind(&PoseGraphOptimizer::odomCallback, this, std::placeholders::_1));

      loop_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
          "loop_closure/constraint", rclcpp::QoS(10), std::bind(&PoseGraphOptimizer::loopClosureCallback, this, std::placeholders::_1));

      path_pub_  = this->create_publisher<nav_msgs::msg::Path>("optimized_path", rclcpp::QoS(1));
      graph_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("factor_graph_markers", rclcpp::QoS(1));

      tf_broadcaster_        = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
      map_to_odom_transform_ = gtsam::Pose3();

      tf_timer_ = this->create_wall_timer(std::chrono::milliseconds(50), [this]() { publishMapToOdomTf(); });
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
    int last_kf_id_ = -1;

    gtsam::Pose3 last_pose_;
    gtsam::Vector3 prev_velocity_ = gtsam::Vector3::Zero();
    gtsam::imuBias::ConstantBias prev_bias_;
    std::shared_ptr<gtsam::PreintegratedImuMeasurements> imu_preintegrated_;
    double last_imu_time_ = -1.0;

    std::deque<sensor_msgs::msg::Imu::SharedPtr> imu_buffer_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr tf_timer_;
    gtsam::Pose3 map_to_odom_transform_;

    gtsam::noiseModel::Diagonal::shared_ptr prior_noise_;
    gtsam::noiseModel::Diagonal::shared_ptr odom_noise_;
    gtsam::noiseModel::Base::shared_ptr loop_noise_;
    gtsam::noiseModel::Isotropic::shared_ptr prior_vel_noise_;
    gtsam::noiseModel::Isotropic::shared_ptr prior_bias_noise_;
    gtsam::noiseModel::Isotropic::shared_ptr bias_rw_noise_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr loop_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr graph_pub_;

    void publishMapToOdomTf() {
      std::lock_guard<std::mutex> lock(mtx_);
      geometry_msgs::msg::TransformStamped t;
      t.header.stamp    = this->now();
      t.header.frame_id = "map";
      t.child_frame_id  = "odom";

      t.transform.translation.x = map_to_odom_transform_.x();
      t.transform.translation.y = map_to_odom_transform_.y();
      t.transform.translation.z = map_to_odom_transform_.z();

      auto q                 = map_to_odom_transform_.rotation().toQuaternion();
      t.transform.rotation.x = q.x();
      t.transform.rotation.y = q.y();
      t.transform.rotation.z = q.z();
      t.transform.rotation.w = q.w();

      tf_broadcaster_->sendTransform(t);
    }

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
      if (!use_imu_) return;
      std::lock_guard<std::mutex> lock(mtx_);
      imu_buffer_.push_back(msg);
    }

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(mtx_);

      int kf_id = -1;
      try {
        kf_id = std::stoi(msg->child_frame_id);
      } catch (const std::exception& e) {
        return;
      }

      double current_odom_time = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

      gtsam::Pose3 current_pose = gtsam::Pose3(gtsam::Rot3::Quaternion(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                                                                       msg->pose.pose.orientation.y, msg->pose.pose.orientation.z),
                                               gtsam::Point3(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z));

      if (last_kf_id_ == -1) {
        if (use_imu_) {
          while (!imu_buffer_.empty()) {
            double imu_time = imu_buffer_.front()->header.stamp.sec + imu_buffer_.front()->header.stamp.nanosec * 1e-9;
            if (imu_time <= current_odom_time) {
              last_imu_time_ = imu_time;
              imu_buffer_.pop_front();
            } else {
              break;
            }
          }
          if (last_imu_time_ < 0) {
            last_imu_time_ = current_odom_time;
          }
        }

        gtSAMgraph_.add(gtsam::PriorFactor<gtsam::Pose3>(X(kf_id), current_pose, prior_noise_));
        initialEstimate_.insert(X(kf_id), current_pose);

        if (use_imu_) {
          gtSAMgraph_.add(gtsam::PriorFactor<gtsam::Vector3>(V(kf_id), prev_velocity_, prior_vel_noise_));
          gtSAMgraph_.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(B(kf_id), prev_bias_, prior_bias_noise_));
          initialEstimate_.insert(V(kf_id), prev_velocity_);
          initialEstimate_.insert(B(kf_id), prev_bias_);
          imu_preintegrated_->resetIntegrationAndSetBias(prev_bias_);
        }
      } else {
        if (use_imu_) {
          while (!imu_buffer_.empty()) {
            auto imu_msg    = imu_buffer_.front();
            double imu_time = imu_msg->header.stamp.sec + imu_msg->header.stamp.nanosec * 1e-9;

            if (imu_time > current_odom_time) {
              break;
            }

            double dt = imu_time - last_imu_time_;
            if (dt > 0.0) {
              gtsam::Vector3 acc(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
              gtsam::Vector3 gyro(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);

              imu_preintegrated_->integrateMeasurement(acc, gyro, dt);
              last_imu_time_ = imu_time;
            }
            imu_buffer_.pop_front();
          }

          if (imu_preintegrated_->deltaTij() <= 1e-4) {
            RCLCPP_WARN(this->get_logger(), "No IMU data integrated between keyframes (dt = %f). Waiting for IMU...", imu_preintegrated_->deltaTij());
            return;
          }
        }

        gtsam::Pose3 relative_pose = last_pose_.between(current_pose);
        gtSAMgraph_.add(gtsam::BetweenFactor<gtsam::Pose3>(X(last_kf_id_), X(kf_id), relative_pose, odom_noise_));

        gtsam::Pose3 optimized_prev_pose = currentEstimate_.at<gtsam::Pose3>(X(last_kf_id_));

        if (use_imu_) {
          gtsam::ImuFactor imu_factor(X(last_kf_id_), V(last_kf_id_), X(kf_id), V(kf_id), B(last_kf_id_), *imu_preintegrated_);
          gtSAMgraph_.add(imu_factor);

          gtSAMgraph_.add(
              gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(last_kf_id_), B(kf_id), gtsam::imuBias::ConstantBias(), bias_rw_noise_));

          gtsam::NavState prev_state(optimized_prev_pose, prev_velocity_);
          gtsam::NavState prop_state = imu_preintegrated_->predict(prev_state, prev_bias_);

          initialEstimate_.insert(X(kf_id), prop_state.pose());
          initialEstimate_.insert(V(kf_id), prop_state.velocity());
          initialEstimate_.insert(B(kf_id), prev_bias_);
        } else {
          gtsam::Pose3 initial_guess_pose = optimized_prev_pose * relative_pose;
          initialEstimate_.insert(X(kf_id), initial_guess_pose);
        }
      }

      last_kf_id_ = kf_id;
      last_pose_  = current_pose;

      isam2_->update(gtSAMgraph_, initialEstimate_);
      for (int i = 0; i < ISAM2_UPDATE_NUM; ++i)
        isam2_->update();

      gtSAMgraph_.resize(0);
      initialEstimate_.clear();

      currentEstimate_ = isam2_->calculateEstimate();

      if (use_imu_) {
        prev_velocity_ = currentEstimate_.at<gtsam::Vector3>(V(kf_id));
        prev_bias_     = currentEstimate_.at<gtsam::imuBias::ConstantBias>(B(kf_id));
        imu_preintegrated_->resetIntegrationAndSetBias(prev_bias_);
      }

      if (currentEstimate_.exists(X(kf_id))) {
        gtsam::Pose3 opt_pose  = currentEstimate_.at<gtsam::Pose3>(X(kf_id));
        map_to_odom_transform_ = opt_pose * current_pose.inverse();
      }

      publishOptimizedPath(msg->header.stamp);
    }

    void loopClosureCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(mtx_);
      int loop_target_id = -1;
      int current_id     = -1;
      try {
        loop_target_id = std::stoi(msg->header.frame_id);
        current_id     = std::stoi(msg->child_frame_id);
      } catch (const std::exception& e) {
        return;
      }

      if (loop_target_id >= current_id) return;

      if (!currentEstimate_.exists(X(current_id)) || !currentEstimate_.exists(X(loop_target_id))) {
        RCLCPP_WARN(this->get_logger(), "Loop closure node not found in graph. Ignored.");
        return;
      }

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

      if (last_kf_id_ != -1 && currentEstimate_.exists(X(last_kf_id_))) {
        gtsam::Pose3 opt_pose  = currentEstimate_.at<gtsam::Pose3>(X(last_kf_id_));
        map_to_odom_transform_ = opt_pose * last_pose_.inverse();
      }

      publishOptimizedPath(msg->header.stamp);
    }

    void publishOptimizedPath(const rclcpp::Time& stamp) {
      nav_msgs::msg::Path path_msg;
      path_msg.header.stamp    = stamp;
      path_msg.header.frame_id = "map";

      std::map<int, gtsam::Pose3> sorted_poses;
      for (const auto& key_value : currentEstimate_) {
        gtsam::Symbol sym(key_value.key);
        if (sym.chr() == 'x') {
          sorted_poses[sym.index()] = currentEstimate_.at<gtsam::Pose3>(sym);
        }
      }

      for (const auto& [id, pose] : sorted_poses) {
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header          = path_msg.header;
        pose_stamped.header.frame_id = std::to_string(id);

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
      path_pub_->publish(path_msg);
    }
  };
} // namespace simple_slam

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(simple_slam::PoseGraphOptimizer)