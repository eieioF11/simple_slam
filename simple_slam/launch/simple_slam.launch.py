import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    # scan_matcher_param = os.path.join(
    #     get_package_share_directory('scan_matcher'),
    #     'config',
    #     'scan_matcher_param.yaml'
    # )

    # pose_graph_optimizer_param = os.path.join(
    #     get_package_share_directory('pose_graph_optimizer'),
    #     'config',
    #     'pose_graph_optimizer_param.yaml'
    # )

    # map_builder_param = os.path.join(
    #     get_package_share_directory('map_builder'),
    #     'config',
    #     'map_builder_param.yaml'
    # )

    # loop_closure_param = os.path.join(
    #     get_package_share_directory('loop_closure'),
    #     'config',
    #     'loop_closure_param.yaml'
    # )
    pkg_share_dir = get_package_share_directory('simple_slam')
    scan_matcher_param = os.path.join(
        pkg_share_dir,
        'config',
        'scan_matcher_param.yaml'
    )

    pose_graph_optimizer_param = os.path.join(
        pkg_share_dir,
        'config',
        'pose_graph_optimizer_param.yaml'
    )

    map_builder_param = os.path.join(
        pkg_share_dir,
        'config',
        'map_builder_param.yaml'
    )

    loop_closure_param = os.path.join(
        pkg_share_dir,
        'config',
        'loop_closure_param.yaml'
    )

    # コンポーザブルノードを格納するコンテナの定義
    container = ComposableNodeContainer(
        name='simple_slam_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        # arguments=['--ros-args', '--log-level', 'DEBUG'],
        composable_node_descriptions=[
            #Scan Matcher
            ComposableNode(
                package='scan_matcher',
                plugin='simple_slam::ScanMatcher',  # ※実際のクラス名に合わせて修正してください
                name='scan_matcher',
                parameters=[scan_matcher_param],
                remappings=[
                    ('in_points', '/velodyne_points')
                ]
            ),
            #Pose Graph Optimizer
            ComposableNode(
                package='pose_graph_optimizer',
                plugin='simple_slam::PoseGraphOptimizer',
                name='pose_graph_optimizer',
                parameters=[pose_graph_optimizer_param]
            ),
            #Map Builder
            ComposableNode(
                package='map_builder',
                plugin='simple_slam::MapBuilder',
                name='map_builder',
                parameters=[map_builder_param]
            ),
            #Loop Closure
            ComposableNode(
                package='loop_closure',
                plugin='simple_slam::LoopClosure',
                name='loop_closure',
                parameters=[loop_closure_param]
            ),
            #Iridescence Viewer
            ComposableNode(
                package='simple_slam',
                plugin='simple_slam::IridescenceViewer',
                name='iridescence_viewer',
                # parameters=[loop_closure_param]
            ),
        ],
        output='screen',
    )

    return LaunchDescription([
        container
    ])