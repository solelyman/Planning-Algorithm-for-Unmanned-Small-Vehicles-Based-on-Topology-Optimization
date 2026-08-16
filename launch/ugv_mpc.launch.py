import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    launch_dir = os.path.dirname(os.path.realpath(__file__))
    pkg_dir = os.path.dirname(launch_dir)          # 仓库根目录
    scripts_dir = os.path.join(pkg_dir, 'multi_ugv_planner')
    params = os.path.join(pkg_dir, 'config', 'ugv_params.yaml')

    # ① 传感器 -> V-PRM 全局规划: /scan + /odom -> /reference_path (世界坐标)
    vprm = ExecuteProcess(
        cmd=['python3', os.path.join(scripts_dir, 'ugv_vprm_node.py'),
             '--goal-x', '2.0', '--goal-y', '0.0'],
        output='screen',
    )

    # ② paper2 C++ unicycle MPC (单体): /odom + /reference_path + /scan -> /cmd_vel
    mpc = Node(
        package='multi_ugv_planner',
        executable='ugv_planner_node_exe',
        name='ugv_planner_node',
        output='screen',
        parameters=[params],
        remappings=[
            ('odom', '/odom'),
            ('reference_path', '/reference_path'),
            ('final_goal', '/final_goal'),
            ('cmd_vel', '/cmd_vel'),
            ('smooth_trajectory', '/smooth_trajectory'),
        ],
    )

    return LaunchDescription([
        SetEnvironmentVariable(
            'LD_LIBRARY_PATH',
            '/home/lu/.local/acados/lib:' +
            '/home/lu/.local/share/acados/lib:' +
            os.environ.get('LD_LIBRARY_PATH', '')),
        vprm,
        mpc,
    ])
