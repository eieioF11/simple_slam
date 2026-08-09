#include "map_builder/map_builder_component.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<simple_slam::MapBuilder>());
  rclcpp::shutdown();
  return 0;
}