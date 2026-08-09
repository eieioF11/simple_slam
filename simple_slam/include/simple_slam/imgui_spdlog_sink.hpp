#pragma once
#include <deque>
#include <imgui.h>
#include <mutex>
#include <rcl_interfaces/msg/log.hpp>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/spdlog.h>
#include <string>

namespace simple_slam {
  struct LogEntry {
    spdlog::level::level_enum level;
    std::string text;

    // デフォルト引数を追加して、テキスト1つだけでも生成できるようにする
    LogEntry() = default;
    LogEntry(spdlog::level::level_enum l, std::string t) : level(l), text(std::move(t)) {}
    LogEntry(std::string t) : level(spdlog::level::info), text(std::move(t)) {} // ← これを追加
  };
  // 独自のspdlogシンク（出力先）を定義
  class ImGuiSpdlogSink : public spdlog::sinks::base_sink<std::mutex> {
  public:
    // max_lines で保持するログの最大行数を指定（メモリ溢れ防止）
    ImGuiSpdlogSink(size_t max_lines = 1000) : max_lines_(max_lines) {}

    // Iridescence (ImGui) のコールバックから毎フレーム呼ばれる描画関数
    void draw_ui() {
      std::lock_guard<std::mutex> lock(custom_mutex_);

      ImGui::Begin("System Log (spdlog)");
      ImGui::BeginChild("LogRegion", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

      for (const auto& entry : messages_) {
        // ログレベルに応じて文字色を動的に変更
        if (entry.level == spdlog::level::err || entry.level == spdlog::level::critical) {
          ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.3f, 0.3f, 1.0f)); // 赤 (ERROR/FATAL)
        } else if (entry.level == spdlog::level::warn) {
          ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.8f, 0.2f, 1.0f)); // 黄 (WARN)
        } else if (entry.level == spdlog::level::debug) {
          ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.6f, 1.0f)); // 灰色 (DEBUG)
        } else {
          ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.9f, 0.9f, 1.0f)); // 白/薄いグレー (INFO)
        }

        ImGui::TextUnformatted(entry.text.c_str());
        ImGui::PopStyleColor();
      }

      if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
        ImGui::SetScrollHereY(1.0f);
      }

      ImGui::EndChild();
      ImGui::End();
    }

    void add_ros_log(const rcl_interfaces::msg::Log& msg) {
      spdlog::level::level_enum spd_level = spdlog::level::info;
      std::string level_str;

      // ROS 2のログレベルを spdlog のレベルに変換
      switch (msg.level) {
        case rcl_interfaces::msg::Log::DEBUG: // DEBUG
          spd_level = spdlog::level::debug;
          level_str = "[DEBUG]";
          break;
        case rcl_interfaces::msg::Log::INFO: // INFO
          spd_level = spdlog::level::info;
          level_str = "[INFO] ";
          break;
        case rcl_interfaces::msg::Log::WARN: // WARN
          spd_level = spdlog::level::warn;
          level_str = "[WARN] ";
          break;
        case rcl_interfaces::msg::Log::ERROR: // ERROR
          spd_level = spdlog::level::err;
          level_str = "[ERROR]";
          break;
        case rcl_interfaces::msg::Log::FATAL: // FATAL
          spd_level = spdlog::level::critical;
          level_str = "[FATAL]";
          break;
        default:
          spd_level = spdlog::level::info;
          level_str = "[UNKNOWN]";
          break;
      }

      std::lock_guard<std::mutex> lock(custom_mutex_);
      std::string log_text = level_str + " [" + msg.name + "] " + msg.msg;
      messages_.push_back({spd_level, log_text});
      if (messages_.size() > max_lines_) {
        messages_.pop_front();
      }
    }

    // void add_raw_text(const std::string& text) {
    //   std::lock_guard<std::mutex> lock(custom_mutex_);
    //   messages_.emplace_back(text);
    //   if (messages_.size() > max_lines_) {
    //     messages_.pop_front();
    //   }
    // }

  protected:
    // spdlog からログが送られてくるたびに呼ばれる
    void sink_it_(const spdlog::details::log_msg& msg) override {
      spdlog::memory_buf_t formatted;
      this->formatter_->format(msg, formatted);

      std::lock_guard<std::mutex> lock(custom_mutex_);
      messages_.push_back({msg.level, std::string(formatted.data(), formatted.size())});
      if (messages_.size() > max_lines_) {
        messages_.pop_front();
      }
    }
    void flush_() override {}

  private:
    std::mutex custom_mutex_;
    std::deque<LogEntry> messages_;
    size_t max_lines_;
  };

} // namespace simple_slam