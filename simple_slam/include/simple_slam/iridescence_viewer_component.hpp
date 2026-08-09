#pragma once

#include <nav_msgs/msg/path.hpp>
#include <rcl_interfaces/msg/log.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

// TF2
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

// Eigen
#include <Eigen/Dense>

// Iridescence
#include "imgui_spdlog_sink.hpp"
#include <glk/pointcloud_buffer.hpp>
#include <glk/primitives/primitives.hpp>
#include <glk/thin_lines.hpp>
#include <guik/viewer/light_viewer.hpp>
#include <imgui.h>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/spdlog.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace simple_slam {

  class IridescenceViewer : public rclcpp::Node {
  public:
    IridescenceViewer(const rclcpp::NodeOptions& options) : IridescenceViewer("", options) {}
    IridescenceViewer(const std::string& name_space = "", const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : Node(name_space.empty() ? "iridescence_viewer_node" : name_space, options), kill_switch_(false) {

      RCLCPP_INFO(this->get_logger(), "Starting Iridescence Viewer Component...");

      // TF2 Listener の初期化
      tf_buffer_   = std::make_unique<tf2_ros::Buffer>(this->get_clock());
      tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

      // Subscribers
      global_map_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
          "map_builder/global_map", rclcpp::QoS(1).transient_local(), std::bind(&IridescenceViewer::globalMapCallback, this, std::placeholders::_1));

      local_map_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
          "scan_matcher/local_map", rclcpp::QoS(1), std::bind(&IridescenceViewer::localMapCallback, this, std::placeholders::_1));

      current_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
          "scan_matcher/out_points", rclcpp::QoS(10), std::bind(&IridescenceViewer::currentCloudCallback, this, std::placeholders::_1));

      path_sub_ = this->create_subscription<nav_msgs::msg::Path>("optimized_path", rclcpp::QoS(10),
                                                                 std::bind(&IridescenceViewer::pathCallback, this, std::placeholders::_1));
      rosout_sub_ =
          this->create_subscription<rcl_interfaces::msg::Log>("/rosout", rclcpp::QoS(10), [this](const rcl_interfaces::msg::Log::SharedPtr msg) {
            if (!imgui_log_sink_) return;
            imgui_log_sink_->add_ros_log(*msg);
          });

      imgui_log_sink_ = std::make_shared<ImGuiSpdlogSink>(1000);
      spdlog::default_logger()->sinks().push_back(imgui_log_sink_);
      spdlog::apply_all([this](std::shared_ptr<spdlog::logger> l) { l->sinks().push_back(imgui_log_sink_); });

      // OpenGLのコンテキスト作成と描画ループを専用の別スレッドで起動
      viewer_thread_ = std::thread(&IridescenceViewer::viewer_loop, this);
    }

    ~IridescenceViewer() {
      RCLCPP_INFO(this->get_logger(), "Shutting down Iridescence Viewer Component...");

      kill_switch_ = true;
      if (viewer_thread_.joinable()) {
        viewer_thread_.join();
      }

      RCLCPP_INFO(this->get_logger(), "Iridescence Viewer Component successfully shutdown.");
      std::_Exit(0);
    }

  private:
    std::mutex mtx_;
    std::thread viewer_thread_;
    std::atomic<bool> kill_switch_;
    std::shared_ptr<ImGuiSpdlogSink> imgui_log_sink_;

    // TF2 members
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // ROS 2 Subscribers
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr global_map_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr local_map_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr current_cloud_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::Subscription<rcl_interfaces::msg::Log>::SharedPtr rosout_sub_;

    // 描画用のデータバッファ
    std::vector<Eigen::Vector3f> current_points_;
    std::vector<Eigen::Vector3f> global_points_;
    std::vector<Eigen::Vector3f> local_points_;
    nav_msgs::msg::Path::SharedPtr path_msg_;

    bool current_cloud_updated_ = false;
    bool global_map_updated_    = false;
    bool local_map_updated_     = false;
    bool path_updated_          = false;

    std::vector<Eigen::Vector3f> extract_points(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) {
      std::vector<Eigen::Vector3f> points;
      points.reserve(msg->width * msg->height);

      sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
      sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
      sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

      for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
        points.emplace_back(*iter_x, *iter_y, *iter_z);
      }
      return points;
    }

    void globalMapCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
      auto points = extract_points(msg);
      std::lock_guard<std::mutex> lock(mtx_);
      global_points_      = std::move(points);
      global_map_updated_ = true;
    }

    void localMapCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
      auto points = extract_points(msg);
      std::lock_guard<std::mutex> lock(mtx_);
      local_points_      = std::move(points);
      local_map_updated_ = true;
    }

    void currentCloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
      auto points = extract_points(msg);
      std::lock_guard<std::mutex> lock(mtx_);
      current_points_        = std::move(points);
      current_cloud_updated_ = true;
    }

    void pathCallback(const nav_msgs::msg::Path::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(mtx_);
      path_msg_     = msg;
      path_updated_ = true;
    }

    void viewer_loop() {
      auto viewer           = guik::LightViewer::instance();
      auto local_map_viewer = viewer->sub_viewer("Local Map");

      float point_scale = 0.3f;
      viewer->register_ui_callback("ui_callback", [&]() {
        ImGui::DragFloat("Point Scale", &point_scale, 0.5f, 0.01f, 1.0f);
        if (ImGui::Button("Close")) {
          viewer->close();
        }
      });

      viewer->register_ui_callback("spdlog_console", [this]() {
        if (imgui_log_sink_) {
          imgui_log_sink_->draw_ui();
        }
      });

      while (rclcpp::ok() && !kill_switch_ && viewer->spin_once()) {
        viewer->update_coord("map", guik::VertexColor());
        local_map_viewer->update_coord("map", guik::VertexColor());

        std::vector<Eigen::Vector3f> render_global_points;
        std::vector<Eigen::Vector3f> render_local_points;
        std::vector<Eigen::Vector3f> render_current_points;
        nav_msgs::msg::Path::SharedPtr render_path;

        bool do_current = false;
        bool do_global  = false;
        bool do_local   = false;
        bool do_path    = false;

        {
          std::lock_guard<std::mutex> lock(mtx_);
          if (global_map_updated_) {
            std::swap(render_global_points, global_points_);
            global_map_updated_ = false;
            do_global           = true;
          }
          if (local_map_updated_) {
            std::swap(render_local_points, local_points_);
            local_map_updated_ = false;
            do_local           = true;
          }
          if (current_cloud_updated_) {
            std::swap(render_current_points, current_points_);
            current_cloud_updated_ = false;
            do_current             = true;
          }
          if (path_updated_) {
            render_path   = path_msg_;
            path_updated_ = false;
            do_path       = true;
          }
        }

        if (do_global && !render_global_points.empty()) {
          auto buffer = std::make_shared<glk::PointCloudBuffer>(render_global_points);
          viewer->update_drawable("global_map", buffer, guik::Rainbow().set_point_scale(point_scale));
        }

        if (do_local && !render_local_points.empty()) {
          auto buffer = std::make_shared<glk::PointCloudBuffer>(render_local_points);
          local_map_viewer->update_drawable("local_map", buffer, guik::FlatColor(0.1f, 0.4f, 1.0f, 1.0f).set_point_scale(point_scale));
        }

        if (do_current && !render_current_points.empty()) {
          auto buffer = std::make_shared<glk::PointCloudBuffer>(render_current_points);
          viewer->update_drawable("current_cloud", buffer, guik::FlatColor(1.0f, 1.0f, 1.0f, 1.0f).set_point_scale(point_scale));
        }

        // 1. TFから最新のロボット姿勢（例: map -> base_link）を取得して座標軸を描画
        try {
          // ※ お使いの環境に合わせて "map" や "base_link" の名前を書き換えてください
          geometry_msgs::msg::TransformStamped transform_stamped = tf_buffer_->lookupTransform("map", "base_link", tf2::TimePointZero);

          Eigen::Matrix4f latest_pose = tf2::transformToEigen(transform_stamped.transform).matrix().cast<float>();
          viewer->update_drawable("current_pose", glk::Primitives::coordinate_system(), guik::VertexColor().transform(latest_pose));
        } catch (const tf2::TransformException& ex) {
          // TFがまだ配信されていない等のエラーは無視してループ継続
        }

        // 2. 軌跡（Path）の描画
        if (do_path && render_path) {
          std::vector<Eigen::Vector3f> path_points;
          path_points.reserve(render_path->poses.size());

          for (const auto& pose_stamped : render_path->poses) {
            const auto& p = pose_stamped.pose.position;
            path_points.emplace_back(p.x, p.y, p.z);
          }

          if (path_points.size() >= 2) {
            auto lines = std::make_shared<glk::ThinLines>(path_points, true);
            viewer->update_drawable("path", lines, guik::FlatColor(0.0f, 1.0f, 0.0f, 1.0f));
          }
        }
      }
    }
  };

} // namespace simple_slam

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(simple_slam::IridescenceViewer)