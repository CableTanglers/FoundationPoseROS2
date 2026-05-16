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
from launch.actions import DeclareLaunchArgument, OpaqueFunction, ExecuteProcess
from launch.substitutions import LaunchConfiguration

CAMERAS = ('left', 'center', 'right')


def _one_cam(cam, mesh_paths, strategy_arg, reset_arg, sim_time_arg):
    # We use ExecuteProcess instead of launch_ros.actions.Node because the
    # patched script is invoked as a plain python module, not a ros2 entry
    # point (the upstream repo ships no setup.py / package.xml). The CLI
    # `--meshes` and `--assign-strategy` flags are forwarded via argv; the
    # ROS param overrides are forwarded via `--ros-args -p`. Both paths land
    # in the same self._assign_strategy field at __init__ time.
    # mesh_paths is the already-split list[str] of mesh files; --meshes
    # expects nargs='+' so we splat it into the cmd array.
    return ExecuteProcess(
        cmd=[
            'python3', '-u', 'foundationpose_ros_multi.py',
            '--meshes', *mesh_paths,
            '--assign-strategy', strategy_arg,
            '--ros-args',
            '-r', f'__node:=foundationpose_{cam}',
            '-p', ['use_sim_time:=', sim_time_arg],
            '-p', f"image_topic:=/wrist_{cam}/image_raw",
            '-p', f"depth_topic:=/aic_isaacros/depth_{cam}",
            '-p', f"camera_info_topic:=/wrist_{cam}/camera_info",
            '-p', f"pose_topic_prefix:=/aic_isaacros/pose_{cam}",
            '-p', f"frame_id_prefix:=wrist_{cam}_optical",
            '-p', 'sync_slop_s:=0.05',
            '-p', ['reset_each_frame:=', reset_arg],
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
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            # PRD v4.2 Phase 1: the daemon must run in either sim-time
            # mode (paired with /clock from a Gazebo run) OR wall-time
            # mode (paired with the host-side benchmark feeder which
            # publishes its own stamps via rclpy). Default false because
            # the offline benchmark is the more common bring-up mode;
            # the live eval harness flips this to true when /clock is
            # published.
            description='ROS use_sim_time — true for /clock-driven eval, '
                        'false for wall-stamp benchmark feeder.',
        ),
    ]
    strategy_arg = LaunchConfiguration('assign_strategy')
    reset_arg    = LaunchConfiguration('reset_each_frame')
    sim_time_arg = LaunchConfiguration('use_sim_time')

    # Resolve meshes at launch time so we can split the
    # space-separated string into separate argv entries (--meshes uses
    # nargs='+' inside foundationpose_ros_multi.py).
    def _spawn(context, *_):
        meshes_str = LaunchConfiguration('meshes').perform(context)
        mesh_paths = meshes_str.split()
        return [
            _one_cam(c, mesh_paths, strategy_arg, reset_arg, sim_time_arg)
            for c in CAMERAS
        ]

    return LaunchDescription(decls + [OpaqueFunction(function=_spawn)])
