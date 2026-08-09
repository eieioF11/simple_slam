#include "scan_matcher/scan_matcher_component.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<simple_slam::ScanMatcher>());
  rclcpp::shutdown();
  return 0;
}