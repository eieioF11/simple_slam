#pragma once
#include <atomic>
#include <deque>
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

// PCL
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl_conversions/pcl_conversions.h>

// small GICP
#include <small_gicp/pcl/pcl_registration.hpp>

// ROS 2 TF & Msgs
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/laser_scan.hpp> // ★追加: 2D LiDARスキャン用
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

// Iridescence
#include <glk/indexed_pointcloud_buffer.hpp>
#include <glk/primitives/primitives.hpp>
#include <guik/spdlog_sink.hpp>
#include <guik/viewer/light_viewer.hpp>
#include <spdlog/fmt/ostr.h>
#include <spdlog/sinks/ringbuffer_sink.h>
#include <spdlog/spdlog.h>

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

#define PCL_POINT_TYPE pcl::PointXYZ
#define FAKE_3D_POINT_Z 0.1

using namespace std::chrono_literals;

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

  class ScanMatcher : public ext_rclcpp::ExtensionNode {
  public:
    ScanMatcher(const rclcpp::NodeOptions& options) : ScanMatcher("", options) {}
    ScanMatcher(const std::string& name_space = "", const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : ext_rclcpp::ExtensionNode("scan_matcher", name_space, options), tf_buffer_(this->get_clock()), listener_(tf_buffer_) {
      RCLCPP_INFO(this->get_logger(), "Starting Scan Matcher Node...");

      // setup
      map_frame_id_   = param<std::string>("scan_matcher.map_frame_id", "map");
      base_frame_id_  = param<std::string>("scan_matcher.base_frame_id", "base_link");
      SMAP_PUBLISH_RATE                    = param<double>("scan_matcher.storage_map.publish_rate", 1.0);
      VOXELGRID_SIZE                       = param<double>("scan_matcher.voxel_grid.size", 0.2);
      RADIUS_OUTLIER_REMOVAL_RADIUS        = param<double>("scan_matcher.radius_outlier_removal.radius", 0.5);
      RADIUS_OUTLIER_REMOVAL_MIN_NEIGHBORS = param<int>("scan_matcher.radius_outlier_removal.min_neighbors", 2);
      // キーフレーム更新の閾値
      KF_MIN_TRANS = param<double>("scan_matcher.keyframe.min_trans", 0.5);
      KF_MIN_ROT   = param<double>("scan_matcher.keyframe.min_rot", 0.3);
      // ローカルマップのウィンドウサイズ
      LOCAL_MAP_WINDOW_SIZE = param<int>("scan_matcher.local_map.window_size", 10);
      // 停止判定の閾値
      STOP_VELOCITY_TRANS = param<double>("scan_matcher.stop_velocity.trans", 0.01);
      STOP_VELOCITY_ROT   = param<double>("scan_matcher.stop_velocity.rot", 0.008);
      // Scan Matcher Parameters
      ICP_MAX_ITERATIONS               = param<int>("scan_matcher.icp.max_iterations", 50);
      ICP_TRANSFORMATION_EPSILON       = param<double>("scan_matcher.icp.transformation_epsilon", 1e-6);
      ICP_MAX_CORRESPONDENCE_DISTANCE  = param<double>("scan_matcher.icp.max_correspondence_distance", 1.0);
      NDT_RESOLUTION                   = param<double>("scan_matcher.ndt.resolution", 1.0);
      NDT_MAX_ITERATIONS               = param<int>("scan_matcher.ndt.max_iterations", 35);
      NDT_TRANSFORMATION_EPSILON       = param<double>("scan_matcher.ndt.transformation_epsilon", 0.01);
      NDT_STEP_SIZE                    = param<double>("scan_matcher.ndt.step_size", 0.1);
      GICP_MAX_ITERATIONS              = param<int>("scan_matcher.gicp.max_iterations", 50);
      GICP_TRANSFORMATION_EPSILON      = param<double>("scan_matcher.gicp.transformation_epsilon", 1e-6);
      GICP_MAX_CORRESPONDENCE_DISTANCE = param<double>("scan_matcher.gicp.max_correspondence_distance", 1.0);
      SGICP_NUM_THREADS = param<int>("scan_matcher.sgicp.num_threads", 4);
      SGICP_CORRESPONDENCE_RANDOMNESS = param<int>("scan_matcher.sgicp.correspondence_randomness", 20);
      SGICP_MAX_CORRESPONDENCE_DISTANCE = param<double>("scan_matcher.sgicp.max_correspondence_distance", 1.0);
      SGICP_VOXEL_RESOLUTION = param<double>("scan_matcher.sgicp.voxel_resolution", 1.0);
      SGICP_REGISTRATION_TYPE = param<std::string>("scan_matcher.sgicp.registration_type", "VGICP");

      // Scan Matcher Type
      scan_matcher_type_ = static_cast<ScanMatcherType>(param<int>("scan_matcher.type", 0));
      // Logger
      logger_                                                                = spdlog::get("scan_matcher_logger");
      std::unordered_map<ScanMatcherType, std::string> scan_matcher_type_map = {{ScanMatcherType::ICP, "ICP"},
                                                                                {ScanMatcherType::NDT, "NDT"},
                                                                                {ScanMatcherType::GICP, "GICP"},
                                                                                {ScanMatcherType::SMALL_GICP, "SMALL_GICP"}};
      if (scan_matcher_type_map.find(scan_matcher_type_) == scan_matcher_type_map.end()) {
        spdlog::error("Invalid scan matcher type: {}", static_cast<int>(scan_matcher_type_));
        throw std::runtime_error("Invalid scan matcher type");
      }
      spdlog::info("Scan Matcher Type = {}", scan_matcher_type_map[scan_matcher_type_]);

      // TF Broadcaster
      tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

      // publisher
      out_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("scan_matcher/out_points", rclcpp::QoS(10));
      map_pub_       = this->create_publisher<sensor_msgs::msg::PointCloud2>("scan_matcher/local_map", rclcpp::QoS(1), rclcpp::PublisherOptions());
      keyframe_odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("scan_matcher/keyframe_odom", rclcpp::QoS(10));

      // subscriber (3D と 2D の両方を待ち受ける)
      cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("in_points", rclcpp::QoS(10),
                                                                            std::bind(&ScanMatcher::cloud_callback, this, std::placeholders::_1));
      scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>("in_scan", rclcpp::QoS(10),
                                                                         std::bind(&ScanMatcher::scan_callback, this, std::placeholders::_1));
      // timer
      double smap_publish_rate = 1.0 / SMAP_PUBLISH_RATE;
      timer_                   = this->create_wall_timer(1s * smap_publish_rate, [&]() {
        if (!local_map_->empty()) {
          sensor_msgs::msg::PointCloud2 map_msg;
          pcl::toROSMsg(*local_map_, map_msg);
          map_msg.header.frame_id = "map";
          map_msg.header.stamp    = this->now();
          map_pub_->publish(map_msg);
        }
      });
    }

  private:
    double SMAP_PUBLISH_RATE;
    double VOXELGRID_SIZE;
    double RADIUS_OUTLIER_REMOVAL_RADIUS;
    int RADIUS_OUTLIER_REMOVAL_MIN_NEIGHBORS;
    double KF_MIN_TRANS;
    double KF_MIN_ROT;
    double STOP_VELOCITY_TRANS;
    double STOP_VELOCITY_ROT;
    size_t LOCAL_MAP_WINDOW_SIZE;
    int ICP_MAX_ITERATIONS;
    double ICP_TRANSFORMATION_EPSILON;
    double ICP_MAX_CORRESPONDENCE_DISTANCE;
    double NDT_RESOLUTION;
    int NDT_MAX_ITERATIONS;
    double NDT_TRANSFORMATION_EPSILON;
    double NDT_STEP_SIZE;
    int GICP_MAX_ITERATIONS;
    double GICP_TRANSFORMATION_EPSILON;
    double GICP_MAX_CORRESPONDENCE_DISTANCE;
    int SGICP_NUM_THREADS;
    int SGICP_CORRESPONDENCE_RANDOMNESS;
    double SGICP_MAX_CORRESPONDENCE_DISTANCE;
    double SGICP_VOXEL_RESOLUTION;
    std::string SGICP_REGISTRATION_TYPE;

    std::string map_frame_id_;
    std::string base_frame_id_;

    enum class ScanMatcherType { ICP, NDT, GICP, SMALL_GICP } scan_matcher_type_;

    std::shared_ptr<spdlog::logger> logger_;

    std::deque<typename pcl::PointCloud<PCL_POINT_TYPE>::Ptr> recent_keyframes_;

    // Poses
    Eigen::Matrix4d current_pose_       = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d previous_pose_      = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d last_keyframe_pose_ = Eigen::Matrix4d::Identity();

    // PointClouds
    typename pcl::PointCloud<PCL_POINT_TYPE>::Ptr local_map_{new pcl::PointCloud<PCL_POINT_TYPE>};

    // TF & ROS
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener listener_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_; // ★追加
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr out_cloud_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr keyframe_odom_pub_;

    // 2Dスキャン処理用のプロジェクターとモードフラグ
    laser_geometry::LaserProjection projector_;
    std::atomic<bool> is_2d_mode_{false};

    // 2D LiDAR トピックコールバック
    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
      is_2d_mode_ = true; // 2Dモードを有効化
      // LaserScan を PointCloud2 に変換
      sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
      projector_.projectLaser(*msg, *cloud_msg);
      // 共通の処理関数へ渡す
      process_cloud(cloud_msg);
    }

    // 3D LiDAR トピックコールバック
    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
      is_2d_mode_ = false; // 2Dモードを無効化
      process_cloud(msg);
    }

    // 実際の点群処理ロジック (統合版)
    void process_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
      typename pcl::PointCloud<PCL_POINT_TYPE>::Ptr current_cloud(new pcl::PointCloud<PCL_POINT_TYPE>);
      pcl::fromROSMsg(*msg, *current_cloud);

      // ダウンサンプリング
      typename pcl::PointCloud<PCL_POINT_TYPE>::Ptr downsampled_cloud(new pcl::PointCloud<PCL_POINT_TYPE>);
      pcl::VoxelGrid<PCL_POINT_TYPE> voxel_filter;
      voxel_filter.setLeafSize(VOXELGRID_SIZE, VOXELGRID_SIZE, VOXELGRID_SIZE);
      voxel_filter.setInputCloud(current_cloud);
      voxel_filter.filter(*downsampled_cloud);

      // 外れ値除去
      pcl::RadiusOutlierRemoval<PCL_POINT_TYPE> radius_outlier_removal;
      radius_outlier_removal.setInputCloud(downsampled_cloud);
      radius_outlier_removal.setRadiusSearch(RADIUS_OUTLIER_REMOVAL_RADIUS);
      radius_outlier_removal.setMinNeighborsInRadius(RADIUS_OUTLIER_REMOVAL_MIN_NEIGHBORS);
      radius_outlier_removal.filter(*downsampled_cloud);

      // 2Dモードの場合の厚み付け
      if (is_2d_mode_) {
        typename pcl::PointCloud<PCL_POINT_TYPE>::Ptr fake_3d_cloud(new pcl::PointCloud<PCL_POINT_TYPE>);
        for (const auto& pt : downsampled_cloud->points) {
          pcl::PointXYZ pt_top    = pt;
          pt_top.z                = FAKE_3D_POINT_Z;
          pcl::PointXYZ pt_bottom = pt;
          pt_bottom.z             = -FAKE_3D_POINT_Z;
          fake_3d_cloud->push_back(pt_top);
          fake_3d_cloud->push_back(pt);
          fake_3d_cloud->push_back(pt_bottom);
        }
        // ダウンサンプリングされた点群を、厚みを持たせた点群に差し替え
        downsampled_cloud = fake_3d_cloud;
      }

      // 初回フレームの処理
      if (local_map_->empty()) {
        *local_map_         = *downsampled_cloud;
        current_pose_       = Eigen::Matrix4d::Identity();
        previous_pose_      = Eigen::Matrix4d::Identity();
        last_keyframe_pose_ = Eigen::Matrix4d::Identity();
        RCLCPP_INFO(this->get_logger(), "Initialized local map with first frame. (2D Mode: %s)", is_2d_mode_ ? "True" : "False");
        publish_tf(msg->header.stamp, current_pose_);
        sensor_msgs::msg::PointCloud2 out_msg;
        pcl::toROSMsg(*downsampled_cloud, out_msg);
        out_msg.header.frame_id = map_frame_id_;
        out_msg.header.stamp    = msg->header.stamp;
        out_cloud_pub_->publish(out_msg);
        publish_keyframe_odom(msg->header.stamp, current_pose_);
        return;
      }

      // 等速直線運動モデルによる初期位置（Initial Guess）の予測
      Eigen::Matrix4d velocity = previous_pose_.inverse() * current_pose_;
      double v_trans           = velocity.block<3, 1>(0, 3).norm();
      double v_rot             = Eigen::AngleAxisd(velocity.block<3, 3>(0, 0)).angle();
      if (v_trans < STOP_VELOCITY_TRANS && v_rot < STOP_VELOCITY_ROT) {
        velocity = Eigen::Matrix4d::Identity();
      }

      // 2Dモード時は2D平面に限定
      if (is_2d_mode_) {
        velocity(2, 3)             = 0.0;
        double vyaw                = std::atan2(velocity(1, 0), velocity(0, 0));
        velocity.block<3, 3>(0, 0) = Eigen::AngleAxisd(vyaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
      }

      Eigen::Matrix4d initial_guess = current_pose_ * velocity;

      // スキャンマッチングの実行 (Target: local_map_, Source: current_cloud)
      auto result = scan_matching<PCL_POINT_TYPE>(local_map_, downsampled_cloud, initial_guess);
      if (result.has_value()) {
        auto [score, tmat, aligned_cloud] = result.value();
        RCLCPP_DEBUG(this->get_logger(), "Scan matching successful. Fitness score: %f", score);

        // 2Dモードの場合の強制クリッピング
        if (is_2d_mode_) {
          tmat(2, 3) = 0.0; // Z座標の浮きを0にリセット
          double yaw = std::atan2(tmat(1, 0), tmat(0, 0));
          Eigen::Matrix3d rot2d;
          rot2d                  = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
          tmat.block<3, 3>(0, 0) = rot2d; // ピッチとロールを捨てる

          // クリッピングした正しい姿勢で点群を再度アライメント
          pcl::transformPointCloud(*downsampled_cloud, aligned_cloud, tmat);
        }

        // 姿勢の更新
        previous_pose_ = current_pose_;
        current_pose_  = tmat; // マップ座標系における現在の位置・姿勢

        // TFの配信
        publish_tf(msg->header.stamp, current_pose_);

        // アラインメントされた点群のパブリッシュ
        sensor_msgs::msg::PointCloud2 out_msg;
        pcl::toROSMsg(aligned_cloud, out_msg);
        out_msg.header.frame_id = map_frame_id_; // マップ座標系に変換済み
        out_msg.header.stamp    = msg->header.stamp;
        out_cloud_pub_->publish(out_msg);

        // キーフレームの更新判定
        Eigen::Matrix4d delta_pose = last_keyframe_pose_.inverse() * current_pose_;
        double delta_trans         = delta_pose.block<3, 1>(0, 3).norm();
        double delta_rot           = Eigen::AngleAxisd(delta_pose.block<3, 3>(0, 0)).angle();
        if (delta_trans > KF_MIN_TRANS || delta_rot > KF_MIN_ROT) {
          typename pcl::PointCloud<PCL_POINT_TYPE>::Ptr cloud_ptr(new pcl::PointCloud<PCL_POINT_TYPE>(aligned_cloud));
          recent_keyframes_.push_back(cloud_ptr);
          if (recent_keyframes_.size() > LOCAL_MAP_WINDOW_SIZE) {
            recent_keyframes_.pop_front();
          }
          // ローカルマップ作成
          local_map_->clear();
          for (const auto& kf_cloud : recent_keyframes_) {
            *local_map_ += *kf_cloud;
          }
          // マップが肥大化しないようにVoxelGridをかける
          typename pcl::PointCloud<PCL_POINT_TYPE>::Ptr filtered_map(new pcl::PointCloud<PCL_POINT_TYPE>);
          voxel_filter.setInputCloud(local_map_);
          voxel_filter.filter(*filtered_map);
          local_map_ = filtered_map;

          last_keyframe_pose_ = current_pose_;
          publish_keyframe_odom(msg->header.stamp, current_pose_);
        }
      } else {
        RCLCPP_WARN(this->get_logger(), "Scan matching failed. Using previous pose as current pose.");
        current_pose_ = previous_pose_;
      }
    }

    void publish_tf(const rclcpp::Time& stamp, const Eigen::Matrix4d& pose) {
      geometry_msgs::msg::TransformStamped t;
      t.header.stamp    = stamp;
      t.header.frame_id = map_frame_id_;
      t.child_frame_id  = base_frame_id_;

      t.transform.translation.x = pose(0, 3);
      t.transform.translation.y = pose(1, 3);
      t.transform.translation.z = pose(2, 3);

      Eigen::Matrix3d R = pose.block<3, 3>(0, 0);
      Eigen::Quaterniond q(R);
      t.transform.rotation.x = q.x();
      t.transform.rotation.y = q.y();
      t.transform.rotation.z = q.z();
      t.transform.rotation.w = q.w();

      tf_broadcaster_->sendTransform(t);
    }

    void publish_keyframe_odom(const rclcpp::Time& stamp, const Eigen::Matrix4d& pose) {
      nav_msgs::msg::Odometry odom_msg;
      odom_msg.header.stamp    = stamp;
      odom_msg.header.frame_id = map_frame_id_;
      odom_msg.child_frame_id  = base_frame_id_;

      odom_msg.pose.pose.position.x = pose(0, 3);
      odom_msg.pose.pose.position.y = pose(1, 3);
      odom_msg.pose.pose.position.z = pose(2, 3);

      Eigen::Matrix3d R = pose.block<3, 3>(0, 0);
      Eigen::Quaterniond q(R);
      odom_msg.pose.pose.orientation.x = q.x();
      odom_msg.pose.pose.orientation.y = q.y();
      odom_msg.pose.pose.orientation.z = q.z();
      odom_msg.pose.pose.orientation.w = q.w();

      keyframe_odom_pub_->publish(odom_msg);
    }

    template <typename POINT_TYPE = pcl::PointXYZ>
    std::optional<std::tuple<double, Eigen::Matrix4d, pcl::PointCloud<POINT_TYPE>>>
    scan_matching(const typename pcl::PointCloud<POINT_TYPE>::ConstPtr& target, const typename pcl::PointCloud<POINT_TYPE>::ConstPtr& source,
                  const Eigen::Matrix4d& initial_guess) {

      if (target->empty() || source->empty()) {
        return std::nullopt;
      }

      pcl::PointCloud<POINT_TYPE> aligned_cloud;
      double fitness_score           = 0.0;
      Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
      Eigen::Matrix4f guess_f        = initial_guess.cast<float>();

      switch (scan_matcher_type_) {
        case ScanMatcherType::ICP: {
          pcl::IterativeClosestPoint<POINT_TYPE, POINT_TYPE> icp;
          icp.setInputTarget(target);
          icp.setInputSource(source);
          icp.setMaximumIterations(ICP_MAX_ITERATIONS);
          icp.setTransformationEpsilon(ICP_TRANSFORMATION_EPSILON);
          icp.setMaxCorrespondenceDistance(ICP_MAX_CORRESPONDENCE_DISTANCE);

          icp.align(aligned_cloud, guess_f);

          if (icp.hasConverged()) {
            fitness_score  = icp.getFitnessScore();
            transformation = icp.getFinalTransformation().template cast<double>();
            return std::make_tuple(fitness_score, transformation, aligned_cloud);
          }
          break;
        }
        case ScanMatcherType::NDT: {
          pcl::NormalDistributionsTransform<POINT_TYPE, POINT_TYPE> ndt;
          ndt.setInputTarget(target);
          ndt.setInputSource(source);
          ndt.setResolution(NDT_RESOLUTION);
          ndt.setMaximumIterations(NDT_MAX_ITERATIONS);
          ndt.setTransformationEpsilon(NDT_TRANSFORMATION_EPSILON);
          ndt.setStepSize(NDT_STEP_SIZE);

          ndt.align(aligned_cloud, guess_f);

          if (ndt.hasConverged()) {
            fitness_score  = ndt.getFitnessScore();
            transformation = ndt.getFinalTransformation().template cast<double>();
            return std::make_tuple(fitness_score, transformation, aligned_cloud);
          }
          break;
        }

        case ScanMatcherType::GICP: {
          pcl::GeneralizedIterativeClosestPoint<POINT_TYPE, POINT_TYPE> gicp;
          gicp.setInputTarget(target);
          gicp.setInputSource(source);
          gicp.setMaximumIterations(GICP_MAX_ITERATIONS);
          gicp.setTransformationEpsilon(GICP_TRANSFORMATION_EPSILON);
          gicp.setMaxCorrespondenceDistance(GICP_MAX_CORRESPONDENCE_DISTANCE);

          gicp.align(aligned_cloud, guess_f);

          if (gicp.hasConverged()) {
            fitness_score  = gicp.getFitnessScore();
            transformation = gicp.getFinalTransformation().template cast<double>();
            return std::make_tuple(fitness_score, transformation, aligned_cloud);
          }
          break;
        }
        case ScanMatcherType::SMALL_GICP: {
          small_gicp::RegistrationPCL<POINT_TYPE, POINT_TYPE> reg;
          reg.setInputTarget(target);
          reg.setInputSource(source);
          reg.setNumThreads(SGICP_NUM_THREADS);
          reg.setCorrespondenceRandomness(SGICP_CORRESPONDENCE_RANDOMNESS);
          reg.setMaxCorrespondenceDistance(SGICP_MAX_CORRESPONDENCE_DISTANCE);
          reg.setVoxelResolution(SGICP_VOXEL_RESOLUTION);
          reg.setRegistrationType(SGICP_REGISTRATION_TYPE);

          reg.align(aligned_cloud, guess_f);
          if (reg.hasConverged()) {
            fitness_score  = reg.getFitnessScore();
            transformation = reg.getFinalTransformation().template cast<double>();
            return std::make_tuple(fitness_score, transformation, aligned_cloud);
          }
        }
        break;
      }

      return std::nullopt;
    }
  };
} // namespace simple_slam

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(simple_slam::ScanMatcher)