#pragma once
#include <deque>
#include <imgui.h>
#include <mutex>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/spdlog.h>
#include <string>

namespace simple_slam {

  // 独自のspdlogシンク（出力先）を定義
  class ImGuiSpdlogSink : public spdlog::sinks::base_sink<std::mutex> {
  public:
    // max_lines で保持するログの最大行数を指定（メモリ溢れ防止）
    ImGuiSpdlogSink(size_t max_lines = 1000) : max_lines_(max_lines) {}

    // Iridescence (ImGui) のコールバックから毎フレーム呼ばれる描画関数
    void draw_ui() {
      std::lock_guard<std::mutex> lock(custom_mutex_);

      ImGui::Begin("System Log (spdlog)"); // ウィンドウの名前

      // スクロール可能な領域を作成
      ImGui::BeginChild("LogRegion", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

      for (const auto& msg : messages_) {
        ImGui::TextUnformatted(msg.c_str());
      }

      // 常に一番下に自動スクロールさせる（最新ログを追従）
      if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
        ImGui::SetScrollHereY(1.0f);
      }

      ImGui::EndChild();
      ImGui::End();
    }

    void add_raw_text(const std::string& text) {
      std::lock_guard<std::mutex> lock(custom_mutex_);
      messages_.emplace_back(text);
      if (messages_.size() > max_lines_) {
        messages_.pop_front();
      }
    }

  protected:
    // spdlog からログが送られてくるたびに呼ばれる
    void sink_it_(const spdlog::details::log_msg& msg) override {
      spdlog::memory_buf_t formatted;
      this->formatter_->format(msg, formatted); // 時刻などの書式設定を適用

      std::lock_guard<std::mutex> lock(custom_mutex_);
      messages_.emplace_back(formatted.data(), formatted.size());

      // 最大行数を超えたら古いものから削除
      if (messages_.size() > max_lines_) {
        messages_.pop_front();
      }
    }

    void flush_() override {}

  private:
    std::mutex custom_mutex_;
    std::deque<std::string> messages_;
    size_t max_lines_;
  };

} // namespace simple_slam