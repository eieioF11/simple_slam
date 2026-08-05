#include "loop_closure/loop_closure_component.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<simple_slam::LoopClosure>());
  rclcpp::shutdown();
  return 0;
}