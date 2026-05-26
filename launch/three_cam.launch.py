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
import os

# Memory constraint on 20 GiB GPU: each FP node uses ~6-8 GiB; with Gazebo
# rendering also competing for VRAM (~2-3 GiB), spawning all 3 OOMs during
# refine_network.forward. AIC_FP_CAMERAS env var overrides; default to
# center-only for smoke validation. Set AIC_FP_CAMERAS=left,center,right
# in production once we have a 24+ GiB GPU OR reduce per-node footprint.
CAMERAS = tuple(
    cam for cam in os.environ.get('AIC_FP_CAMERAS', 'center').split(',') if cam
)

# Agent CC (2026-05-18): per-cam CUDA device pinning. Each FP node
# uses ~13 GiB on an A4500 (with est_refine_iter=1 + reset_each_frame
# + kornia.warp_perspective transient) so 2 nodes do not fit on one
# 20 GiB GPU. AIC_FP_CUDA_VISIBLE_<CAM> lets the chain script pin
# each FP process to a different physical GPU. If unset, the FP
# container's compose-level CUDA_VISIBLE_DEVICES is inherited (legacy
# single-cam flow). The docker container must expose BOTH host GPUs
# (CUDA_VISIBLE_DEVICES=0,1 inside container) for this to take effect.
_PER_CAM_CUDA_ENV = {
    cam: os.environ.get(f'AIC_FP_CUDA_VISIBLE_{cam.upper()}', '')
    for cam in CAMERAS
}


def _one_cam(cam, mesh_paths, strategy_arg, reset_arg, sim_time_arg,
             pose_mode_arg, seed_topic_arg, seed_max_age_ms_arg,
             mask_in_sync_arg, mask_sync_slop_s_arg,
             external_mask_max_age_ms_arg):
    # We use ExecuteProcess instead of launch_ros.actions.Node because the
    # patched script is invoked as a plain python module, not a ros2 entry
    # point (the upstream repo ships no setup.py / package.xml). The CLI
    # `--meshes` and `--assign-strategy` flags are forwarded via argv; the
    # ROS param overrides are forwarded via `--ros-args -p`. Both paths land
    # in the same self._assign_strategy field at __init__ time.
    # mesh_paths is the already-split list[str] of mesh files; --meshes
    # expects nargs='+' so we splat it into the cmd array.
    # Agent CC (2026-05-18): per-cam CUDA pinning. Only override the
    # container's default CUDA_VISIBLE_DEVICES when the caller provided
    # one — otherwise let docker-compose's CUDA_VISIBLE_DEVICES rule.
    # Used on 2-GPU hardware to spread FP across host GPUs for multi-cam.
    additional_env = {}
    pinned_gpu = _PER_CAM_CUDA_ENV.get(cam, '')
    if pinned_gpu:
        additional_env['CUDA_VISIBLE_DEVICES'] = pinned_gpu
        print(f'[three_cam.launch] pinning {cam} → CUDA_VISIBLE_DEVICES={pinned_gpu}')

    return ExecuteProcess(
        cmd=[
            'python3', '-u', 'foundationpose_ros_multi.py',
            '--meshes', *mesh_paths,
            '--assign-strategy', strategy_arg,
            '--ros-args',
            '-r', f'__node:=foundationpose_{cam}',
            '-p', ['use_sim_time:=', sim_time_arg],
            # Topic names default to the live AIC eval pipeline (aic_adapter
            # publishes /<cam>_camera/image + /<cam>_camera/camera_info).
            # Offline benchmark feeder uses /wrist_<cam>/image_raw + matching
            # camera_info; that mode would need overrides via
            # `ros2 launch ... image_topic:=...` or a parallel launch file.
            '-p', f"image_topic:=/{cam}_camera/image",
            '-p', f"depth_topic:=/aic_isaacros/depth_{cam}",
            '-p', f"camera_info_topic:=/{cam}_camera/camera_info",
            '-p', f"pose_topic_prefix:=/aic_isaacros/pose_{cam}",
            # AIC PATCH (2026-05-17): the AIC eval URDF publishes the
            # wrist camera frames as `<cam>_camera/optical` (per
            # basler_camera_macro). The prior `wrist_{cam}_optical` was
            # from the upstream demo bag and doesn't exist in our TF tree.
            # fp_daemon_server's TF lookup needs this to match.
            '-p', f"frame_id_prefix:={cam}_camera/optical",
            '-p', 'sync_slop_s:=0.05',
            # HUNK 20/22: sync_queue remains for the legacy 3-leg
            # ApproximateTimeSynchronizer. The live single-mesh mask path
            # uses event_sync_queue below so late source-stamped depth/mask
            # streams can find their source RGB without widening sync slop.
            '-p', 'sync_queue:=100',
            '-p', 'event_sync_queue:=250',
            '-p', ['reset_each_frame:=', reset_arg],
            # HUNK 17 (Agent R, 2026-05-18): seeded_track mode + seed
            # topic wiring. Default 'register' preserves legacy
            # behaviour. Live NIC chain uses 'seeded_track' to consume
            # the nic_pose_daemon's broadcast prior.
            '-p', ['pose_mode:=', pose_mode_arg],
            '-p', ['seed_topic:=', seed_topic_arg],
            '-p', ['seed_max_age_ms:=', seed_max_age_ms_arg],
            # HUNK 20/22: mask-in-sync wiring. When mask_in_sync=true AND
            # argv has exactly one mesh, the daemon uses the HUNK 22
            # event-keyed synchronizer (tight mask_sync_slop_s stamp
            # tolerance). For the multi-mesh case the param silently
            # degrades to the legacy cached path
            # (controlled by external_mask_max_age_ms).
            '-p', ['mask_in_sync:=', mask_in_sync_arg],
            '-p', ['mask_sync_slop_s:=', mask_sync_slop_s_arg],
            '-p', ['external_mask_max_age_ms:=', external_mask_max_age_ms_arg],
        ],
        output='screen',
        name=f'foundationpose_{cam}',
        additional_env=additional_env,
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
        # HUNK 17 (Agent R, 2026-05-18): seeded_track wiring.
        DeclareLaunchArgument(
            'pose_mode',
            default_value='register',
            description="HUNK 17: 'register' (default — multimodal hypothesis "
                        "search per frame), 'seeded_track' (refine from "
                        "broadcast /aic_vision/coarse_pose_seed prior), or "
                        "'track' (register on bootstrap, track_one after).",
        ),
        DeclareLaunchArgument(
            'seed_topic',
            default_value='/aic_vision/coarse_pose_seed',
            description="HUNK 17: PoseStamped seed topic (T_cam_obj in "
                        "<cam>_camera/optical) consumed in seeded_track mode.",
        ),
        DeclareLaunchArgument(
            'seed_max_age_ms',
            default_value='500.0',
            description="HUNK 17: max age (ms) of seed vs current RGB stamp "
                        "before falling back to register-mode.",
        ),
        # HUNK 20 (Agent Z, 2026-05-18): mask-in-sync wiring.
        DeclareLaunchArgument(
            'mask_in_sync',
            default_value='true',
            description="HUNK 20: include external mask topic as the 4th leg "
                        "of ApproximateTimeSynchronizer. Only takes effect "
                        "when exactly one mesh is passed (single-obj setup). "
                        "Multi-mesh launches silently fall back to the legacy "
                        "cached path.",
        ),
        DeclareLaunchArgument(
            'mask_sync_slop_s',
            default_value='0.1',
            description="HUNK 20: ApproximateTimeSynchronizer slop (seconds) "
                        "for the 4-leg (RGB, depth, info, mask) sync — must "
                        "be tight so SAM lag relative to RGB does not bias "
                        "the pose.",
        ),
        DeclareLaunchArgument(
            'external_mask_max_age_ms',
            default_value='200.0',
            description="HUNK 20: max age (ms) of a cached external mask in "
                        "the LEGACY non-synced path. Was 30000 (30s) historically; "
                        "200ms matches live-chain SAM latency expectation.",
        ),
    ]
    strategy_arg = LaunchConfiguration('assign_strategy')
    reset_arg    = LaunchConfiguration('reset_each_frame')
    sim_time_arg = LaunchConfiguration('use_sim_time')
    pose_mode_arg       = LaunchConfiguration('pose_mode')
    seed_topic_arg      = LaunchConfiguration('seed_topic')
    seed_max_age_ms_arg = LaunchConfiguration('seed_max_age_ms')
    mask_in_sync_arg               = LaunchConfiguration('mask_in_sync')
    mask_sync_slop_s_arg           = LaunchConfiguration('mask_sync_slop_s')
    external_mask_max_age_ms_arg   = LaunchConfiguration('external_mask_max_age_ms')

    # Resolve meshes at launch time so we can split the
    # space-separated string into separate argv entries (--meshes uses
    # nargs='+' inside foundationpose_ros_multi.py).
    def _spawn(context, *_):
        meshes_str = LaunchConfiguration('meshes').perform(context)
        mesh_paths = meshes_str.split()
        return [
            _one_cam(c, mesh_paths, strategy_arg, reset_arg, sim_time_arg,
                     pose_mode_arg, seed_topic_arg, seed_max_age_ms_arg,
                     mask_in_sync_arg, mask_sync_slop_s_arg,
                     external_mask_max_age_ms_arg)
            for c in CAMERAS
        ]

    return LaunchDescription(decls + [OpaqueFunction(function=_spawn)])
