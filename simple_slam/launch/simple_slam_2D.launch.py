import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    pkg_share_dir = get_package_share_directory('simple_slam')
    scan_matcher_param = os.path.join(
        pkg_share_dir,
        # get_package_share_directory('scan_matcher'),
        'config',
        '2D',
        'scan_matcher_param.yaml'
    )

    pose_graph_optimizer_param = os.path.join(
        pkg_share_dir,
        # get_package_share_directory('pose_graph_optimizer'),
        'config',
        '2D',
        'pose_graph_optimizer_param.yaml'
    )

    map_builder_param = os.path.join(
        pkg_share_dir,
        # get_package_share_directory('map_builder'),
        'config',
        '2D',
        'map_builder_param.yaml'
    )

    loop_closure_param = os.path.join(
        pkg_share_dir,
        # get_package_share_directory('loop_closure'),
        'config',
        '2D',
        'loop_closure_param.yaml'
    )

    # コンポーザブルノードを格納するコンテナの定義
    container = ComposableNodeContainer(
        name='simple_slam_container',
        namespace='',
        package='rclcpp_components',
        # executable='component_container',
        executable='component_container_mt',
        arguments=['--ros-args', '--log-level', 'DEBUG'],
        composable_node_descriptions=[
            #Scan Matcher
            ComposableNode(
                package='scan_matcher',
                plugin='simple_slam::ScanMatcher',  # ※実際のクラス名に合わせて修正してください
                name='scan_matcher',
                parameters=[scan_matcher_param],
                remappings=[
                    ('in_scan', '/scan'),
                ],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
            #Pose Graph Optimizer
            ComposableNode(
                package='pose_graph_optimizer',
                plugin='simple_slam::PoseGraphOptimizer',
                name='pose_graph_optimizer',
                parameters=[pose_graph_optimizer_param],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
            #Map Builder
            ComposableNode(
                package='map_builder',
                plugin='simple_slam::MapBuilder',
                name='map_builder',
                parameters=[map_builder_param],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
            #Loop Closure
            ComposableNode(
                package='loop_closure',
                plugin='simple_slam::LoopClosure',
                name='loop_closure',
                parameters=[loop_closure_param],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
            #Iridescence Viewer
            ComposableNode(
                package='simple_slam',
                plugin='simple_slam::IridescenceViewer',
                name='iridescence_viewer',
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
        ],
        output='screen',
    )

    return LaunchDescription([
        container
    ])