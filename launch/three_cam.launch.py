# HUNK 6: fan-out launch for AIC's three wrist cameras.
#
# Each instance subscribes to one RGB stream + the matching per-camera depth
# republished by aic_vision/scripts/depth_adapter_gazebo_gt.py on
# /aic_isaacros/depth_{left,center,right}.
#
# Mesh order is fixed: argv index 0 = nic_assembly, 1 = sc_port. The AIC
# adapter routes the mesh subset by trial task (see
# aic_vision/aic_vision/estimators/isaacros_foundationpose.py).
#
# Usage (from inside the aic_isaacros container):
#   ros2 launch foundationpose_ros2 three_cam.launch.py \
#       meshes:="/aic/data/benchmark/cad_obj/nic_assembly/textured_simple.obj \
#                /aic/data/benchmark/cad_obj/sc_port/textured_simple.obj" \
#       assign_strategy:=largest \
#       reset_each_frame:=true

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

CAMERAS = ('left', 'center', 'right')


def _one_cam(cam, meshes_arg, strategy_arg, reset_arg):
    # We use ExecuteProcess instead of launch_ros.actions.Node because the
    # patched script is invoked as a plain python module, not a ros2 entry
    # point (the upstream repo ships no setup.py / package.xml). The CLI
    # `--meshes` and `--assign-strategy` flags are forwarded via argv; the
    # ROS param overrides are forwarded via `--ros-args -p`. Both paths land
    # in the same self._assign_strategy field at __init__ time.
    return ExecuteProcess(
        cmd=[
            'python3', '-u', 'foundationpose_ros_multi.py',
            '--meshes', meshes_arg,
            '--assign-strategy', strategy_arg,
            '--ros-args',
            '-r', f'__node:=foundationpose_{cam}',
            '-p', 'use_sim_time:=true',
            '-p', f"image_topic:=/wrist_{cam}/image_raw",
            '-p', f"depth_topic:=/aic_isaacros/depth_{cam}",
            '-p', f"camera_info_topic:=/wrist_{cam}/camera_info",
            '-p', f"pose_topic_prefix:=/aic_isaacros/pose_{cam}",
            '-p', f"frame_id_prefix:=wrist_{cam}_optical",
            '-p', 'sync_slop_s:=0.05',
            '-p', f"reset_each_frame:={reset_arg}",
        ],
        output='screen',
        name=f'foundationpose_{cam}',
    )


def generate_launch_description():
    decls = [
        DeclareLaunchArgument(
            'meshes',
            description='Space-separated list of OBJ paths (argv order = '
                        'assignment order).',
        ),
        DeclareLaunchArgument(
            'assign_strategy',
            default_value='largest',
            description='largest | gt_bbox_iou | class_filter (HUNK 3).',
        ),
        DeclareLaunchArgument(
            'reset_each_frame',
            default_value='false',
            description='HUNK 9: per-frame register() reset for sparse-frame '
                        'benchmark mode.',
        ),
    ]
    meshes_arg   = LaunchConfiguration('meshes')
    strategy_arg = LaunchConfiguration('assign_strategy')
    reset_arg    = LaunchConfiguration('reset_each_frame')
    return LaunchDescription(
        decls + [_one_cam(c, meshes_arg, strategy_arg, reset_arg) for c in CAMERAS]
    )
