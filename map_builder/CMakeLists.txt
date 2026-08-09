cmake_minimum_required(VERSION 3.8)
project(map_builder)

add_compile_options(-O3)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake_auto REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
find_package(Iridescence REQUIRED)
if(NOT TARGET spdlog)
  # Stand-alone build
  find_package(spdlog REQUIRED)
endif()
ament_auto_find_build_dependencies()

# create component
ament_auto_add_library(${PROJECT_NAME}_component SHARED
  src/main.cpp
)
target_include_directories(${PROJECT_NAME}_component PRIVATE
  include
  ${PCL_INCLUDE_DIRS}
  # ${small_gicp_INCLUDE_DIRS}
)

rclcpp_components_register_node(${PROJECT_NAME}_component
  PLUGIN      "simple_slam::MapBuilder"
  EXECUTABLE  ${PROJECT_NAME}
)
target_link_libraries(${PROJECT_NAME}_component
  tbb
  Iridescence::Iridescence
  spdlog::spdlog_header_only
  ${PCL_LIBRARIES}
  # small_gicp::small_gicp
)
target_compile_definitions(${PROJECT_NAME}_component PRIVATE ${PCL_DEFINITIONS})

# install config
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/
)

# build testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()
