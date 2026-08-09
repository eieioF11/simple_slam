#pragma once

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rcl_interfaces/msg/log.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_srvs/srv/trigger.hpp>

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
#include <glk/texture.hpp>
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

  struct MapImageData {
    int width  = 0;
    int height = 0;
    std::vector<uint8_t> pixels; // RGBAピクセルデータ
  };

  class IridescenceViewer : public rclcpp::Node {
  public:
    IridescenceViewer(const rclcpp::NodeOptions& options) : IridescenceViewer("", options) {}
    IridescenceViewer(const std::string& name_space = "", const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : Node(name_space.empty() ? "iridescence_viewer_node" : name_space, options), kill_switch_(false) {

      RCLCPP_INFO(this->get_logger(), "Starting Iridescence Viewer Component...");

      tf_buffer_   = std::make_unique<tf2_ros::Buffer>(this->get_clock());
      tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

      // サービスクライアントの作成
      save_map_client_ = this->create_client<std_srvs::srv::Trigger>("save_map");

      // Subscribers
      global_map_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
          "map_builder/global_map", rclcpp::QoS(1).transient_local(), std::bind(&IridescenceViewer::globalMapCallback, this, std::placeholders::_1));

      local_map_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
          "scan_matcher/local_map", rclcpp::QoS(1), std::bind(&IridescenceViewer::localMapCallback, this, std::placeholders::_1));

      current_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
          "scan_matcher/out_points", rclcpp::QoS(10), std::bind(&IridescenceViewer::currentCloudCallback, this, std::placeholders::_1));

      path_sub_ = this->create_subscription<nav_msgs::msg::Path>("optimized_path", rclcpp::QoS(10),
                                                                 std::bind(&IridescenceViewer::pathCallback, this, std::placeholders::_1));

      // グリッドマップのサブスクライバー追加
      grid_map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
          "map", rclcpp::QoS(1).transient_local(), std::bind(&IridescenceViewer::gridMapCallback, this, std::placeholders::_1));

      rosout_sub_ =
          this->create_subscription<rcl_interfaces::msg::Log>("/rosout", rclcpp::QoS(10), [this](const rcl_interfaces::msg::Log::SharedPtr msg) {
            if (!imgui_log_sink_) return;
            imgui_log_sink_->add_ros_log(*msg);
          });

      imgui_log_sink_ = std::make_shared<ImGuiSpdlogSink>(1000);
      spdlog::default_logger()->sinks().push_back(imgui_log_sink_);
      spdlog::apply_all([this](std::shared_ptr<spdlog::logger> l) { l->sinks().push_back(imgui_log_sink_); });

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

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr save_map_client_; // ★ サービスクライアント

    MapImageData grid_image_data_;
    bool grid_image_updated_ = false;
    std::shared_ptr<glk::Texture> grid_texture_;

    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr global_map_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr local_map_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr current_cloud_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_map_sub_;
    rclcpp::Subscription<rcl_interfaces::msg::Log>::SharedPtr rosout_sub_;

    std::vector<Eigen::Vector3f> current_points_;
    std::vector<Eigen::Vector3f> global_points_;
    std::vector<Eigen::Vector3f> local_points_;
    std::vector<Eigen::Vector3f> grid_points_;
    nav_msgs::msg::Path::SharedPtr path_msg_;

    bool current_cloud_updated_ = false;
    bool global_map_updated_    = false;
    bool local_map_updated_     = false;
    bool grid_map_updated_      = false;
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

    // グリッドマップを受信して占有セルを点群化する
    void gridMapCallback(const nav_msgs::msg::OccupancyGrid::ConstSharedPtr msg) {
      std::vector<Eigen::Vector3f> points;
      points.reserve(msg->data.size() / 10);

      MapImageData img_data;
      img_data.width  = msg->info.width;
      img_data.height = msg->info.height;
      img_data.pixels.resize(msg->info.width * msg->info.height * 4); // RGBA (4bytes per pixel)

      float res = msg->info.resolution;
      float ox  = msg->info.origin.position.x;
      float oy  = msg->info.origin.position.y;
      float oz  = -0.1f;

      for (unsigned int y = 0; y < msg->info.height; ++y) {
        for (unsigned int x = 0; x < msg->info.width; ++x) {
          int idx    = y * msg->info.width + x;
          int8_t val = msg->data[idx];

          // --- 1. 3D点群用の処理 ---
          if (val > 50) {
            float px = ox + (x + 0.5f) * res;
            float py = oy + (y + 0.5f) * res;
            points.emplace_back(px, py, oz);
          }

          // --- 2. 2D画像用の処理 ---
          // ROSのマップは原点が左下ですが、画像は左上なのでY軸を反転させます
          int img_y   = (msg->info.height - 1 - y);
          int img_idx = (img_y * msg->info.width + x) * 4;

          uint8_t r, g, b, a = 255;
          if (val == -1) {
            // 未知領域 (Unknown) = グレー
            r = g = b = 127;
          } else if (val == 0) {
            // 自由空間 (Free) = 白
            r = g = b = 255;
          } else if (val == 100) {
            // 障害物 (Obstacle) = 黒
            r = g = b = 0;
          } else {
            // 中間の確率 (0〜100) = グラデーション
            uint8_t color = 255 - static_cast<uint8_t>((val * 255) / 100);
            r = g = b = color;
          }

          img_data.pixels[img_idx + 0] = r;
          img_data.pixels[img_idx + 1] = g;
          img_data.pixels[img_idx + 2] = b;
          img_data.pixels[img_idx + 3] = a;
        }
      }

      std::lock_guard<std::mutex> lock(mtx_);
      grid_points_      = std::move(points);
      grid_map_updated_ = true;

      grid_image_data_    = std::move(img_data);
      grid_image_updated_ = true;
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

        // マップ保存サービスコールのボタンを追加
        ImGui::Separator();
        if (ImGui::Button("Save Map to PCD")) {
          if (save_map_client_->service_is_ready()) {
            auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
            save_map_client_->async_send_request(req, [this](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
              if (future.get()->success) {
                RCLCPP_INFO(this->get_logger(), "Map save success: %s", future.get()->message.c_str());
              } else {
                RCLCPP_ERROR(this->get_logger(), "Map save failed: %s", future.get()->message.c_str());
              }
            });
          } else {
            RCLCPP_WARN(this->get_logger(), "Save map service is not available yet.");
          }
        }
        ImGui::Separator();

        if (ImGui::Button("Close")) {
          viewer->close();
        }

        if (grid_texture_) {
          ImGui::Begin("2D Grid Map Image");
          // ウィンドウ幅に合わせて画像をリサイズ表示
          float window_width = ImGui::GetWindowWidth() - 16.0f;
          float aspect_ratio = static_cast<float>(grid_texture_->size().y()) / grid_texture_->size().x();

          ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(grid_texture_->id())), ImVec2(window_width, window_width * aspect_ratio));
          ImGui::End();
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
        std::vector<Eigen::Vector3f> render_grid_points; // ★
        nav_msgs::msg::Path::SharedPtr render_path;
        MapImageData render_grid_image;
        bool do_grid_image_update = false;
        bool do_current           = false;
        bool do_global            = false;
        bool do_local             = false;
        bool do_path              = false;

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
          if (grid_image_updated_) {
            std::swap(render_grid_image, grid_image_data_);
            grid_image_updated_  = false;
            do_grid_image_update = true;
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

        // グリッドマップの描画（黒に近い濃いグレーで描画）
        if (do_grid_image_update && !render_grid_image.pixels.empty()) {
          // テクスチャが未作成、またはマップサイズが変わった場合は再生成
          if (!grid_texture_ || grid_texture_->size().x() != render_grid_image.width || grid_texture_->size().y() != render_grid_image.height) {
            grid_texture_ = std::make_shared<glk::Texture>(Eigen::Vector2i(render_grid_image.width, render_grid_image.height), GL_RGBA, GL_RGBA,
                                                           GL_UNSIGNED_BYTE);
          }

          // GPUにピクセルデータを転送
          glBindTexture(GL_TEXTURE_2D, grid_texture_->id());
          // 拡大した時にドット絵のようにくっきり表示させる設定（ニアレストネイバー）
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

          glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, render_grid_image.width, render_grid_image.height, GL_RGBA, GL_UNSIGNED_BYTE,
                          render_grid_image.pixels.data());
          glBindTexture(GL_TEXTURE_2D, 0);
        }

        try {
          geometry_msgs::msg::TransformStamped transform_stamped = tf_buffer_->lookupTransform("map", "base_link", tf2::TimePointZero);
          Eigen::Matrix4f latest_pose                            = tf2::transformToEigen(transform_stamped.transform).matrix().cast<float>();
          viewer->update_drawable("current_pose", glk::Primitives::coordinate_system(), guik::VertexColor().transform(latest_pose));
        } catch (const tf2::TransformException& ex) {
        }

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