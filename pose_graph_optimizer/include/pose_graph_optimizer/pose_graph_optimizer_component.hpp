#pragma once

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>

// extention node
#include "extension_node/extension_node.hpp"

// GTSAM
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
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

namespace simple_slam {

  class PoseGraphOptimizer : public ext_rclcpp::ExtensionNode {
  public:
    PoseGraphOptimizer(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : ext_rclcpp::ExtensionNode("pose_graph_optimizer", options) {

      RCLCPP_INFO(this->get_logger(), "Starting GTSAM Pose Graph Optimizer (ISAM2)...");
      logger_ = spdlog::get("pose_graph_optimizer_logger");

      // ISAM2のパラメータ設定
      gtsam::ISAM2Params parameters;
      parameters.relinearizeThreshold = param<double>("pose_graph_optimizer.isam2.relinearizeThreshold", 0.1);
      parameters.relinearizeSkip      = param<int>("pose_graph_optimizer.isam2.relinearizeSkip", 1);
      ISAM2_UPDATE_NUM                = param<int>("pose_graph_optimizer.isam2.update_num", 3);
      // parameters.relinearizeThreshold = 0.1;
      // parameters.relinearizeSkip      = 1;
      isam2_ = std::make_unique<gtsam::ISAM2>(parameters);
      // spdlog::info("ISAM2 parameters: \nrelinearizeThreshold={}\nrelinearizeSkip={}", parameters.relinearizeThreshold, parameters.relinearizeSkip);

      // ノイズモデル（共分散）の定義
      // 実際にはスキャンマッチングのFitness Score等から動的に設定するのが理想です
      prior_noise_ = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
      odom_noise_  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3).finished());
      loop_noise_  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 5e-2, 5e-2, 5e-2, 1e-2, 1e-2, 1e-2).finished());

      // Subscribers
      // フロントエンド(ScanMatcher)からのキーフレームごとのオドメトリを受信
      odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("scan_matcher/keyframe_odom", rclcpp::QoS(10),
                                                                     std::bind(&PoseGraphOptimizer::odomCallback, this, std::placeholders::_1));

      // 外部のLoop Closureノードからの相対ポーズを受信（例）
      loop_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
          "loop_closure/constraint", rclcpp::QoS(10), std::bind(&PoseGraphOptimizer::loopClosureCallback, this, std::placeholders::_1));

      // Publishers
      path_pub_ = this->create_publisher<nav_msgs::msg::Path>("optimized_path", rclcpp::QoS(1));
    }

  private:
    // GTSAM Graph & ISAM2
    int ISAM2_UPDATE_NUM;
    std::unique_ptr<gtsam::ISAM2> isam2_;
    gtsam::NonlinearFactorGraph gtSAMgraph_;
    gtsam::Values initialEstimate_;
    gtsam::Values currentEstimate_;
    std::shared_ptr<spdlog::logger> logger_;

    // 状態管理
    std::mutex mtx_;
    int keyframe_id_ = 0;
    gtsam::Pose3 last_pose_; // 前回のポーズ

    // Noise Models
    gtsam::noiseModel::Diagonal::shared_ptr prior_noise_;
    gtsam::noiseModel::Diagonal::shared_ptr odom_noise_;
    gtsam::noiseModel::Diagonal::shared_ptr loop_noise_;

    // ROS
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr loop_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;

    // フロントエンドからのオドメトリ（Scan-to-Mapの結果など）を受け取る
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(mtx_);

      // ROS msg -> GTSAM Pose3 変換
      gtsam::Pose3 current_pose = gtsam::Pose3(gtsam::Rot3::Quaternion(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                                                                       msg->pose.pose.orientation.y, msg->pose.pose.orientation.z),
                                               gtsam::Point3(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z));

      if (keyframe_id_ == 0) {
        // 最初のノード（原点）をグラフに固定する PriorFactor
        gtSAMgraph_.add(gtsam::PriorFactor<gtsam::Pose3>(0, current_pose, prior_noise_));
        initialEstimate_.insert(0, current_pose);
      } else {
        // 前回からの相対移動（オドメトリ）を計算
        gtsam::Pose3 relative_pose = last_pose_.between(current_pose);

        // ノード間の制約（BetweenFactor）を追加
        gtSAMgraph_.add(gtsam::BetweenFactor<gtsam::Pose3>(keyframe_id_ - 1, keyframe_id_, relative_pose, odom_noise_));

        // 最新ポーズの初期推定値としてスキャンマッチャーの結果をそのまま入れる
        initialEstimate_.insert(keyframe_id_, current_pose);
      }

      last_pose_ = current_pose;
      keyframe_id_++;

      // ISAM2で最適化を実行
      isam2_->update(gtSAMgraph_, initialEstimate_);
      for (int i = 0; i < ISAM2_UPDATE_NUM; ++i)
        isam2_->update();

      // グラフと初期推定値をクリア（ISAM2の内部に保持されるため）
      gtSAMgraph_.resize(0);
      initialEstimate_.clear();

      // 最適化後の軌跡を取得・パブリッシュ
      currentEstimate_ = isam2_->calculateEstimate();
      publishOptimizedPath(msg->header.stamp);
    }

    // ループ検出時のコールバック（例）
    // msgには、過去のID(frame_id等に格納)と現在のID間の相対ポーズが入っている想定
    void loopClosureCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(mtx_);

      // 例: header.frame_id に対象の古いIDが含まれていると仮定
      int loop_target_id = std::stoi(msg->header.frame_id);
      int current_id     = keyframe_id_ - 1;

      if (loop_target_id >= current_id) return;

      gtsam::Pose3 loop_relative_pose = gtsam::Pose3(gtsam::Rot3::Quaternion(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                                                                             msg->pose.pose.orientation.y, msg->pose.pose.orientation.z),
                                                     gtsam::Point3(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z));

      // 過去のノードと現在のノードを繋ぐ BetweenFactor を追加
      gtSAMgraph_.add(gtsam::BetweenFactor<gtsam::Pose3>(loop_target_id, current_id, loop_relative_pose, loop_noise_));

      // spdlog::info("Loop Closure Added: {} -> {}", current_id, loop_target_id);
      RCLCPP_INFO(this->get_logger(), "Loop Closure Added: %d -> %d", current_id, loop_target_id);

      // ループを閉じた際はグラフ全体が大きく動くため、ISAM2を更新
      isam2_->update(gtSAMgraph_, initialEstimate_);
      for (int i = 0; i < ISAM2_UPDATE_NUM; ++i)
        isam2_->update();

      gtSAMgraph_.resize(0);
      initialEstimate_.clear();

      currentEstimate_ = isam2_->calculateEstimate();
    }

    void publishOptimizedPath(const rclcpp::Time& stamp) {
      nav_msgs::msg::Path path_msg;
      path_msg.header.stamp    = stamp;
      path_msg.header.frame_id = "map";

      for (int i = 0; i < keyframe_id_; ++i) {
        if (currentEstimate_.exists(i)) {
          gtsam::Pose3 pose = currentEstimate_.at<gtsam::Pose3>(i);
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
  };

} // namespace simple_slam

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(simple_slam::PoseGraphOptimizer)
