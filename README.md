# simple_slam

simple_slam is a ROS 2 package for matching laser scans to a map and visualizing the result. It is designed for localization and mapping workflows and provides launch files for both normal and 2D modes.

## Overview

This package includes:

- scan matching and map-related components
- ROS 2 interfaces for sensor input and localization output
- launch files for running the node in different modes

## Dependencies

The package depends on the following libraries and ROS 2 packages:

### ROS 2 dependencies

- rclcpp
- rclcpp_components
- std_msgs
- nav_msgs
- std_srvs
- tf2_ros
- tf2
- tf2_geometry_msgs
- laser_geometry

### External libraries

- PCL (Point Cloud Library)
- TBB
- OpenMP
- Iridescence
- spdlog

### Internal workspace dependencies

The required `common_utils` and `extension_node` packages are bundled in the [ros2_common_tools](https://github.com/eieioF11/ros2_common_tools) repository.

To install them, please clone the repository and initialize the submodules:

```bash
cd ros2_ws/src
git clone https://github.com/eieioF11/ros2_common_tools.git
cd ros2_common_tools
git submodule update --init --recursive
```

### Ubuntu / system packages

You may need to install the following packages before building:

```bash
sudo apt update
sudo apt install -y build-essential cmake libboost-all-dev libpcl-dev libtbb-dev libiridescence-dev
```

If you are using rosdep, you can also resolve the ROS dependencies with:

```bash
rosdep install --from-paths src/simple_slam --ignore-src -r -y
```

## Folder Structure

The repository is organized as follows:

```text
simple_slam/
├── simple_slam/        # main ROS 2 package
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── config/         # configuration files
│   ├── include/        # headers
│   ├── launch/         # launch files
│   └── src/            # source files
├── loop_closure/       # loop closure related components
├── map_builder/        # map building logic
├── pose_graph_optimizer/  # pose graph optimization
├── scan_matcher/       # scan matching implementation
```

## Build

From your ROS 2 workspace:

```bash
cd ~/ros2_ws
colcon build --packages-select simple_slam --symlink-install
source install/setup.bash
```

## Run

Launch the default node:

```bash
ros2 launch simple_slam simple_slam.launch.py
```

Launch the 2D mode variant:

```bash
ros2 launch simple_slam simple_slam_2D.launch.py
```

## Notes

- The package uses C++17 and is built with CMake through ament.
- The build is optimized with O3 and optional OpenMP support.
- If you encounter missing dependency errors, make sure your ROS 2 environment is sourced and that the required packages are installed.
