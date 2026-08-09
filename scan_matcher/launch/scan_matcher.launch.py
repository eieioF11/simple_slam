from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='simple_slam',
            executable='scan_matcher_node', # CMakeLists.txtで指定した実行ファイル名
            name='scan_matcher_node',
            remappings=[
                ('in_points', '/velodyne_points')
            ],
            parameters=[
                {'scan_matcher.type': 0},
                {'scan_matcher.voxel_grid.size': 0.2}
            ]
        )
    ])