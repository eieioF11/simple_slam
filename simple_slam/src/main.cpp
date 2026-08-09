#include "simple_slam/iridescence_viewer_component.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<simple_slam::IridescenceViewer>());
  rclcpp::shutdown();
  return 0;
}