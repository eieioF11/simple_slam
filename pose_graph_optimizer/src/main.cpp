#include "pose_graph_optimizer/pose_graph_optimizer_component.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<simple_slam::PoseGraphOptimizer>());
  rclcpp::shutdown();
  return 0;
}