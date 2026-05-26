import sys
sys.path.append('./FoundationPose')
sys.path.append('./FoundationPose/nvdiffrast')

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from estimater import *
import cv2
import numpy as np
import trimesh
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseStamped
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
import argparse
import os
import threading
import time
import tf2_ros
from tf2_ros import TransformException
from rclpy.time import Time as _RclpyTime
from typing import Optional
from collections import OrderedDict
from scipy.spatial.transform import Rotation as R
from ultralytics import SAM
from cam_2_base_transform import *
import os
import tkinter as tk
from tkinter import Listbox, END, Button
import glob

# HUNK 17 (Agent R, 2026-05-18): torch import for seeded_track mode. The
# adapter must push a 4x4 prior into pose_est.pose_last as a CUDA tensor
# before calling track_one(). Defer the import — if torch isn't on the
# Python path (e.g. headless smoke tests), the node still constructs and
# falls back to register-mode.
try:
    import torch as _torch  # noqa: F401
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False

# Save the original `__init__` and `register` methods
original_init = FoundationPose.__init__
original_register = FoundationPose.register

# Modify `__init__` to add `is_register` attribute
def modified_init(self, model_pts, model_normals, symmetry_tfs=None, mesh=None, scorer=None, refiner=None, glctx=None, debug=0, debug_dir='./FoundationPose'):
    original_init(self, model_pts, model_normals, symmetry_tfs, mesh, scorer, refiner, glctx, debug, debug_dir)
    self.is_register = False  # Initialize as False

# Modify `register` to set `is_register` to True when a pose is registered
def modified_register(self, K, rgb, depth, ob_mask, iteration):
    pose = original_register(self, K, rgb, depth, ob_mask, iteration)
    self.is_register = True  # Set to True after registration
    return pose

# Apply the modifications
FoundationPose.__init__ = modified_init
FoundationPose.register = modified_register

class FileSelectorGUI:
    def __init__(self, master, file_paths):
        self.master = master
        self.master.title("Library: Sequence Selector")
        self.file_paths = file_paths
        self.reordered_paths = None  # Store the reordered paths here

        # Create a listbox to display the file names
        self.listbox = Listbox(master, selectmode="extended", width=50, height=10)
        self.listbox.pack()

        # Populate the listbox with file names without extensions
        for file_path in self.file_paths:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            self.listbox.insert(END, file_name)

        # Buttons for rearranging the order
        self.up_button = Button(master, text="Move Up", command=self.move_up)
        self.up_button.pack(side="left", padx=5, pady=5)

        self.down_button = Button(master, text="Move Down", command=self.move_down)
        self.down_button.pack(side="left", padx=5, pady=5)

        self.done_button = Button(master, text="Done", command=self.done)
        self.done_button.pack(side="left", padx=5, pady=5)

    def move_up(self):
        """Move selected items up in the listbox."""
        selected_indices = list(self.listbox.curselection())
        for index in selected_indices:
            if index > 0:
                # Swap with the previous item
                file_name = self.listbox.get(index)
                self.listbox.delete(index)
                self.listbox.insert(index - 1, file_name)
                self.listbox.selection_set(index - 1)

    def move_down(self):
        """Move selected items down in the listbox."""
        selected_indices = list(self.listbox.curselection())
        for index in reversed(selected_indices):
            if index < self.listbox.size() - 1:
                # Swap with the next item
                file_name = self.listbox.get(index)
                self.listbox.delete(index)
                self.listbox.insert(index + 1, file_name)
                self.listbox.selection_set(index + 1)

    def done(self):
        """Save the reordered paths and close the GUI."""
        reordered_file_names = self.listbox.get(0, END)

        # Recreate the full file paths based on the reordered file names (without extensions)
        file_name_to_full_path = {
            os.path.splitext(os.path.basename(file))[0]: file for file in self.file_paths
        }
        self.reordered_paths = [file_name_to_full_path[file_name] for file_name in reordered_file_names]

        # Close the GUI
        self.master.quit()

    def get_reordered_paths(self):
        """Return the reordered file paths after the GUI has closed."""
        return self.reordered_paths

# Example usage
def rearrange_files(file_paths):
    root = tk.Tk()
    app = FileSelectorGUI(root, file_paths)
    root.mainloop()  # Start the GUI event loop
    return app.get_reordered_paths()  # Return the reordered paths after GUI closes

# Argument Parser
# HUNK 2: add --meshes for headless mesh selection (kills Tkinter GUI in Docker/CI).
# HUNK 3: add --assign-strategy for deterministic mask-to-mesh assignment.
parser = argparse.ArgumentParser()
code_dir = os.path.dirname(os.path.realpath(__file__))
parser.add_argument('--est_refine_iter', type=int, default=1)  # was 4; matches isaacros_foundationpose_runner.py baseline
# HUNK 19 (Agent W diagnosis, 2026-05-18): tried bumping
# track_refine_iter default 2 -> 5 to close a ~25 mm seeded_track
# residual. Empirically MADE IT WORSE — 25 mm -> 52 mm with a TIGHTER
# cluster. Hypothesis: 5 iters lets the refiner converge confidently
# onto the SAM mask, but the SAM mask itself is mis-aligned vs the
# image timestamp (wrist moves ~7.5 cm in the ~250 ms between seed
# publish and FP processing), so converging harder onto a wrong-pose
# mask produces a worse latched pose. Reverted to default=2. Real fix
# is to (a) align seed/mask to image timestamp, OR (b) skip the seed
# entirely via pose_mode=register (cold-start each frame). See
# AIC_FP_POSE_MODE=register in start_fp_daemon_chain_legal.sh.
parser.add_argument('--track_refine_iter', type=int, default=2)
parser.add_argument(
    '--meshes', nargs='+', default=None,
    help='HUNK 2: explicit mesh paths in display/assignment order. '
         'Required for headless runs; falls back to Tk picker only if DISPLAY is set.'
)
parser.add_argument(
    '--assign-strategy', dest='assign_strategy',
    choices=['largest', 'gt_bbox_iou', 'class_filter'],
    default='largest',
    help='HUNK 3: deterministic mask-to-mesh assignment strategy.'
)
args, _unknown = parser.parse_known_args()

class PoseEstimationNode(Node):
    def __init__(self, new_file_paths):
        super().__init__('pose_estimation_node')

        # HUNK 5: declare use_sim_time so /clock from `ros2 bag play` is respected.
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)

        # HUNK 1: topic names + frame/topic prefixes are now ROS parameters so
        # multi-instance launch (HUNK 6) can configure per-camera without forks.
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('pose_topic_prefix', '/Current_OBJ_position')
        self.declare_parameter('frame_id_prefix', 'object')

        # HUNK 3: assignment strategy can be set by either CLI flag (single-process
        # mode) or ROS param (launch fan-out). Param takes precedence so the launch
        # file in HUNK 6 wins over a stale default.
        self.declare_parameter('assign_strategy', args.assign_strategy)

        # HUNK 8: synchronizer slop + queue depth.
        self.declare_parameter('sync_slop_s', 0.05)
        self.declare_parameter('sync_queue', 10)

        # HUNK 9: sparse-frame benchmark mode. When True, every frame re-runs
        # register() (no temporal tracking). Required for the frames_v2 manifest
        # where successive frames may be wider than the tracker convergence basin.
        self.declare_parameter('reset_each_frame', False)

        # HUNK 17 (Agent R, 2026-05-18): seeded_track mode. When pose_mode is
        # 'seeded_track' AND a recent PoseStamped seed is available on
        # ``seed_topic`` whose ``header.frame_id`` matches this node's
        # ``frame_id_prefix``, the daemon skips register()'s multimodal
        # hypothesis search and runs FP.track_one() refined from the seed
        # prior — i.e. the offline Phase 3b 3.88mm pattern, but live. Falls
        # back to register-mode when no fresh seed exists (cold start).
        # Default 'register' preserves legacy behavior for callers that
        # haven't wired the seed topic.
        self.declare_parameter('pose_mode', 'register')
        self.declare_parameter('seed_topic', '/aic_vision/coarse_pose_seed')
        self.declare_parameter('seed_max_age_ms', 500.0)
        # HUNK 21 (2026-05-19): world-frame seed. The daemon publishes
        # T_world_obj in ``base_frame``; this node composes
        # ``T_cam_obj = inv(T_world_cam(rgb_stamp)) @ T_world_obj`` at
        # the synced RGB frame's stamp via the TF buffer, eliminating
        # the wrist-motion bias that the prior per-cam composition
        # introduced (daemon's "latest" TF was at a different sim-time
        # than the FP frame, biasing the seeded_track prior).
        self.declare_parameter('base_frame', 'base_link')

        # HUNK 20 (Agent Z, 2026-05-18): mask-in-sync mode. When enabled and
        # the daemon is configured for a single mesh (single obj_id), the
        # external mask topic is added as a 4th leg of the
        # ApproximateTimeSynchronizer alongside (RGB, depth, camera_info).
        # This guarantees the mask processed for a given RGB frame was
        # produced from an RGB image with a stamp within ``sync_slop_s``
        # of the current frame — closing the SAM-vs-RGB temporal mismatch
        # that drove the 28mm TE in the live NIC chain (SAM at ~3.7s/frame
        # was lagging RGB, and FP was pulling stale cached masks via the
        # legacy 30s ``_external_mask_max_age_ns`` window).
        #
        # ``mask_topic`` is the explicit topic name; when empty (default)
        # we derive it from ``pose_topic_prefix`` the same way the legacy
        # cached-mask path does (s/pose_/mask_/ + ``_{obj_id}`` suffix).
        # For the multi-mesh case we fall back to the legacy cached path
        # (mask_in_sync silently disables).
        self.declare_parameter('mask_in_sync', True)
        self.declare_parameter('mask_topic', '')
        self.declare_parameter('mask_sync_slop_s', 0.1)
        # HUNK 22 (2026-05-26): event-keyed sync buffer for delayed derived
        # streams. FoundationStereo depth and the downstream mask carry the
        # SOURCE RGB stamp but arrive ~4s later, so a generic 4-leg
        # ApproximateTimeSynchronizer can evict the matching RGB first.
        # Buffer >=10s at 20Hz; exact/tight stamp matching still uses
        # mask_sync_slop_s.
        self.declare_parameter('event_sync_queue', 250)
        # Tighter fallback for the legacy cached path (only relevant when
        # mask_in_sync is False or disabled by multi-mesh): 200ms cap so
        # we skip frames where SAM lagged RGB rather than running FP on a
        # stale mask. Was 30s historically (matched register-per-frame
        # mode's ~18s budget); the new 200ms is tied to live-chain SAM
        # latency.
        self.declare_parameter('external_mask_max_age_ms', 200.0)

        image_topic = self.get_parameter('image_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        info_topic  = self.get_parameter('camera_info_topic').value
        self._pose_topic_prefix = self.get_parameter('pose_topic_prefix').value
        self._frame_id_prefix   = self.get_parameter('frame_id_prefix').value
        self._assign_strategy   = self.get_parameter('assign_strategy').value
        self._reset_each_frame  = bool(self.get_parameter('reset_each_frame').value)
        self._pose_mode         = str(self.get_parameter('pose_mode').value).strip().lower()
        self._seed_topic        = str(self.get_parameter('seed_topic').value)
        self._seed_max_age_ns   = int(float(self.get_parameter('seed_max_age_ms').value) * 1e6)
        self._base_frame        = str(self.get_parameter('base_frame').value)
        slop = float(self.get_parameter('sync_slop_s').value)
        qlen = int(self.get_parameter('sync_queue').value)
        self._mask_in_sync_param = bool(self.get_parameter('mask_in_sync').value)
        mask_topic_param         = str(self.get_parameter('mask_topic').value).strip()
        mask_sync_slop           = float(self.get_parameter('mask_sync_slop_s').value)
        event_sync_queue         = int(self.get_parameter('event_sync_queue').value)

        if self._pose_mode not in ('register', 'seeded_track', 'track'):
            self.get_logger().warning(
                f"HUNK 17: unknown pose_mode='{self._pose_mode}'; falling back to 'register'"
            )
            self._pose_mode = 'register'

        self.get_logger().info(
            f"FoundationPoseROS2 [PATCHED]: rgb='{image_topic}' depth='{depth_topic}' "
            f"info='{info_topic}' strat='{self._assign_strategy}' "
            f"reset_each_frame={self._reset_each_frame} "
            f"pose_mode='{self._pose_mode}' seed_topic='{self._seed_topic}' "
            f"seed_max_age_ms={self._seed_max_age_ns/1e6:.0f}"
        )

        # HUNK 8: synchronized entry point for the (RGB, depth, camera_info)
        # trio. Slop = 0.05 s = the PRD's timestamp_tolerance_ms (50 ms).
        # HUNK 22 creates subscriptions after mesh loading so the single-mesh
        # live path can use event-keyed sync without also subscribing via
        # message_filters.
        #
        # HUNK 16: RELIABLE QoS. ros_gz_bridge publishes the AIC eval
        # container's RGB / depth / camera_info topics with
        # Reliability: RELIABLE (verified via `ros2 topic info -v`
        # against my-eval:v1). BEST_EFFORT here silently fails the
        # QoS-compatibility check under rmw_zenoh_cpp — the daemon
        # shows matched publishers but never receives a frame. Same
        # root cause hit the chain's depth_adapter + mask_publisher.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=qlen,
        )
        # HUNK 20 (Agent Z): defer the synchronizer construction until we
        # know how many meshes were passed (mask_in_sync only kicks in
        # for the single-mesh case). Subscriptions are built below, after
        # ``self.meshes`` is populated.
        self._sync_qlen = qlen
        self._sync_slop = slop
        self._mask_sync_slop = mask_sync_slop
        self._event_sync_qlen = max(1, event_sync_queue)
        self._event_sync_tol_ns = int(mask_sync_slop * 1e9)
        self._mask_topic_param = mask_topic_param
        self._sync_qos = qos

        self.bridge = CvBridge()
        self.depth_image = None
        self.color_image = None
        # HUNK 11 (post-v4.2 Phase 1.F): external mask input. When the
        # operator publishes per-mesh binary masks on
        # `/aic_isaacros/mask_<cam>_<obj_id>`, the daemon SKIPS SAM2 +
        # _assign_masks_to_meshes and uses the supplied mask directly.
        # Falls back to SAM2+largest when no fresh mask is available.
        # Required for the SC path because SAM2's "largest blob"
        # heuristic picks the cable instead of the SC port at typical
        # wrist-camera working distance.
        self._external_masks: dict[int, tuple[int, np.ndarray]] = {}  # obj_id → (stamp_ns, mask)
        # HUNK 20 (Agent Z): externally-tunable freshness window. Historic
        # value was 30s (matched register_per_frame=true budget of ~18s
        # per frame). Live-chain default flipped to 200ms so the legacy
        # cached path (when mask_in_sync is False or disabled by
        # multi-mesh) skips frames where SAM lags RGB rather than
        # processing on a stale mask. Live mask_in_sync=true bypasses
        # this entirely via the HUNK 22 event-keyed synchronizer.
        self._external_mask_max_age_ns = int(
            float(self.get_parameter('external_mask_max_age_ms').value) * 1e6
        )
        self._n_synced_frames = 0
        self._n_skipped_stale_mask = 0
        self._n_skipped_no_mask = 0
        self._event_rgb_msgs = OrderedDict()
        self._event_depth_msgs = OrderedDict()
        self._event_mask_msgs = OrderedDict()
        self._event_info_msg: Optional[CameraInfo] = None
        self._event_n_matched = 0
        self._event_n_no_rgb = 0
        self._event_n_no_pair = 0
        self._event_n_no_info = 0
        self._event_n_info_changed = 0
        self.cam_K = None  # Initialize cam_K as None until we receive the camera info

        # HUNK 5: track the input RGB stamp so publish_pose_stamped can echo it.
        self._last_rgb_stamp = None
        # HUNK 7: one-shot log marker for depth-unit auto-detection.
        self._logged_depth_units = False

        # Load meshes
        self.mesh_files = new_file_paths
        self.meshes = [trimesh.load(mesh) for mesh in self.mesh_files]

        self.bounds = [trimesh.bounds.oriented_bounds(mesh) for mesh in self.meshes]
        self.bboxes = [np.stack([-extents/2, extents/2], axis=0).reshape(2, 3) for _, extents in self.bounds]

        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()

        # Initialize SAM2 model
        self.seg_model = SAM("sam2.1_b.pt")

        self.pose_estimations = {}  # Dictionary to track multiple pose estimations
        self.pose_publishers = {}  # Dictionary to store publishers for each object
        self.tracked_objects = []  # Initialize to store selected objects' masks
        self.i = 0
        # HUNK 9: dedicated frame counter (upstream's self.i counts per-object,
        # not per-frame — using it as a "have we ever processed a frame?" gate
        # is wrong as soon as more than one mesh is registered).
        self._frame_count = 0
        # HUNK 3 + HUNK 9: bootstrap flag. True on first frame and every frame
        # in reset mode. Set to False after a successful pose_estimations build.
        self._needs_reregister = True

        # HUNK 11 / HUNK 20 / HUNK 22: external mask wiring. Two paths:
        #
        #   (A) mask_in_sync=True AND single mesh — RECOMMENDED for the
        #       live NIC chain. HUNK 22 syncs by source event stamp:
        #       RGB is cached, CameraInfo K is cached/static, and depth+mask
        #       trigger dispatch once both source-stamped derived streams
        #       arrive within ``mask_sync_slop_s`` (default 100ms).
        #
        #   (B) Legacy cached path — mask_in_sync=False OR multi-mesh.
        #       The mask topic is a bare subscription; each callback caches
        #       the latest (stamp, mask) and process_images() consumes it
        #       via _external_masks lookup with a configurable max-age cap
        #       (``external_mask_max_age_ms``, default 200ms).
        #
        # Topic naming: ``mask_topic`` param overrides; otherwise we
        # derive from ``pose_topic_prefix`` via the legacy
        # s/pose_/mask_/ + ``_{obj_id}`` suffix scheme.
        mask_prefix = self._pose_topic_prefix.replace("/pose_", "/mask_", 1)
        n_meshes = len(self.meshes)
        self._mask_in_sync_active = (
            self._mask_in_sync_param and n_meshes == 1
        )

        if self._mask_in_sync_active:
            # HUNK 22: Single-mesh live path uses an event-keyed
            # synchronizer instead of a 4-leg ApproximateTimeSynchronizer.
            # Depth and mask are source-stamped but arrive seconds after
            # RGB; we cache RGB by stamp and trigger when the slow
            # source-stamped streams have both arrived. CameraInfo is
            # static K, so cache/validate it once instead of making it a
            # delayed sync leg.
            obj_id = 1  # single mesh ⇒ obj_id = mesh_idx + 1 = 1
            mask_topic = (
                self._mask_topic_param
                if self._mask_topic_param
                else f"{mask_prefix}_{obj_id}"
            )
            self.create_subscription(Image, image_topic, self._event_on_rgb, qos)
            self.create_subscription(Image, depth_topic, self._event_on_depth, qos)
            self.create_subscription(CameraInfo, info_topic, self._event_on_info, qos)
            self.create_subscription(Image, mask_topic, self._event_on_mask, qos)
            self.get_logger().info(
                f"HUNK 22: event-sync ENABLED. mask_topic='{mask_topic}' "
                f"tol={self._mask_sync_slop:.3f}s queue={self._event_sync_qlen}; "
                f"CameraInfo cached/static"
            )
        else:
            # Legacy path: 3-leg synchronizer for (RGB, depth, info)
            # plus per-obj bare subscription for cached masks.
            self.image_mf = Subscriber(self, Image, image_topic, qos_profile=qos)
            self.depth_mf = Subscriber(self, Image, depth_topic, qos_profile=qos)
            self.info_mf = Subscriber(self, CameraInfo, info_topic, qos_profile=qos)
            self._sync = ApproximateTimeSynchronizer(
                [self.image_mf, self.depth_mf, self.info_mf],
                queue_size=self._sync_qlen, slop=self._sync_slop,
            )
            self._sync.registerCallback(self._synced_callback)
            for mesh_idx in range(n_meshes):
                obj_id = mesh_idx + 1
                topic = (
                    self._mask_topic_param
                    if (self._mask_topic_param and n_meshes == 1)
                    else f"{mask_prefix}_{obj_id}"
                )
                self.create_subscription(
                    Image, topic,
                    lambda m, o=obj_id: self._on_external_mask(m, o),
                    qos,
                )
                self.get_logger().info(
                    f"HUNK 11: subscribed external mask topic {topic} → obj_id={obj_id}"
                )
            self.get_logger().info(
                f"HUNK 20: mask-in-sync DISABLED "
                f"(param={self._mask_in_sync_param} n_meshes={n_meshes}); "
                f"using cached path with max_age={self._external_mask_max_age_ns/1e6:.0f}ms"
            )

        # HUNK 21 (2026-05-19): coarse-pose seed in world frame. The daemon
        # publishes ``T_world_obj`` as a PoseStamped with
        # ``header.frame_id == base_frame`` (e.g. "base_link"). This node
        # composes ``T_cam_obj = inv(T_world_cam(rgb_stamp)) @ T_world_obj``
        # at the synced RGB frame's stamp using its own TF buffer, so the
        # prior is temporally aligned with the FP frame regardless of how
        # out-of-phase the daemon's publish tick is. Seed QoS matches the
        # daemon publisher (RELIABLE + TRANSIENT_LOCAL).
        #
        # Supersedes HUNK 17's per-cam ``<cam>_camera/optical`` seeds:
        # those required the daemon to perform composition at its own
        # tick using ``Time()`` (latest TF), which under bag-replay drove
        # a ~20mm wrist-motion bias into the seeded_track prior.
        self._latest_seed_T_world_obj: Optional[np.ndarray] = None
        self._seed_receipt_ns: int = 0
        self._seed_lock = threading.Lock()
        if self._pose_mode == 'seeded_track':
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
            seed_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=5,
            )
            self.create_subscription(
                PoseStamped, self._seed_topic,
                self._on_pose_seed, seed_qos,
            )
            self.get_logger().info(
                f"HUNK 21: subscribed seed topic {self._seed_topic} "
                f"(world-frame seeds in '{self._base_frame}' composed at "
                f"'{self._frame_id_prefix}' via TF)"
            )

    def _on_pose_seed(self, msg):
        """HUNK 21 (2026-05-19): cache the latest world-frame coarse-pose
        seed. The daemon publishes ``T_world_obj`` once per tick; this node
        defers composition into ``T_cam_obj`` to the synced RGB callback
        where ``_last_rgb_stamp`` is available, so the TF lookup matches
        the frame FP is about to process.
        """
        if msg.header.frame_id and msg.header.frame_id != self._base_frame:
            # Defensive: reject anything that isn't in our base frame.
            # (The daemon always publishes in base_frame after HUNK 21.)
            return
        px = float(msg.pose.position.x)
        py = float(msg.pose.position.y)
        pz = float(msg.pose.position.z)
        qx = float(msg.pose.orientation.x)
        qy = float(msg.pose.orientation.y)
        qz = float(msg.pose.orientation.z)
        qw = float(msg.pose.orientation.w)
        try:
            Rm = R.from_quat([qx, qy, qz, qw]).as_matrix()
        except Exception:
            return
        T_wo = np.eye(4, dtype=np.float32)
        T_wo[:3, :3] = Rm.astype(np.float32)
        T_wo[:3, 3] = (px, py, pz)
        receipt_ns = time.monotonic_ns()
        with self._seed_lock:
            self._latest_seed_T_world_obj = T_wo
            self._seed_receipt_ns = receipt_ns
        if not getattr(self, "_seed_first_logged", False):
            self.get_logger().info(
                f"HUNK 21: first world-frame seed received frame_id='{msg.header.frame_id}' "
                f"t_world_obj=({px:.3f}, {py:.3f}, {pz:.3f})"
            )
            self._seed_first_logged = True

    def _fresh_seed_T_cam_obj(self, obj_id: int):
        """Return ``(T_cam_obj, age_ns)`` composed at the synced RGB frame
        stamp, or None if no seed is fresh / TF is unavailable.

        HUNK 21 (2026-05-19): the world seed is composed just-in-time using
        the TF tree at ``_last_rgb_stamp`` (the synchronized RGB image's
        header stamp). This eliminates the daemon-tick vs FP-frame
        wrist-motion lag that biased the prior under bag replay.

        Age is measured from local receipt time (monotonic_ns) so we don't
        compare ROS clock domains when use_sim_time differs between the
        daemon and this node.
        """
        with self._seed_lock:
            T_wo = self._latest_seed_T_world_obj
            receipt_ns = self._seed_receipt_ns
        if T_wo is None:
            self.get_logger().warning(
                "HUNK 21 fallthrough: no world seed cached",
                throttle_duration_sec=2.0,
            )
            return None
        age = time.monotonic_ns() - receipt_ns
        if age > self._seed_max_age_ns:
            self.get_logger().warning(
                f"HUNK 21 fallthrough: seed stale age={age/1e6:.0f}ms "
                f"> max={self._seed_max_age_ns/1e6:.0f}ms",
                throttle_duration_sec=2.0,
            )
            return None
        if self._last_rgb_stamp is None:
            self.get_logger().warning(
                "HUNK 21 fallthrough: no RGB stamp yet",
                throttle_duration_sec=2.0,
            )
            return None
        # Compose T_cam_obj = inv(T_world_cam(rgb_stamp)) @ T_world_obj.
        try:
            tfs = self._tf_buffer.lookup_transform(
                self._frame_id_prefix,   # cam optical frame
                self._base_frame,        # world / base_link
                _RclpyTime.from_msg(self._last_rgb_stamp),
            )
        except TransformException as e:
            self.get_logger().warning(
                f"HUNK 21 fallthrough: TF lookup failed "
                f"'{self._frame_id_prefix}'<-'{self._base_frame}'@{self._last_rgb_stamp.sec}.{self._last_rgb_stamp.nanosec}: {e}",
                throttle_duration_sec=2.0,
            )
            return None
        t = tfs.transform.translation
        q = tfs.transform.rotation
        try:
            Rcw_mat = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        except Exception:
            return None
        T_cw = np.eye(4, dtype=np.float32)
        T_cw[:3, :3] = Rcw_mat.astype(np.float32)
        T_cw[:3, 3] = (float(t.x), float(t.y), float(t.z))
        T_co = (T_cw @ T_wo).astype(np.float32)
        return T_co, age

    def _on_external_mask(self, msg, obj_id: int):
        """HUNK 11: cache the latest external mask for this obj_id."""
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as e:
            self.get_logger().warning(
                f"HUNK 11: external mask decode failed for obj_id={obj_id}: {e}"
            )
            return
        stamp_ns = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
        coverage = int((mask > 0).sum())
        is_new = obj_id not in self._external_masks
        self._external_masks[obj_id] = (stamp_ns, (mask > 0).astype(np.uint8) * 255)
        if is_new:
            self.get_logger().info(
                f"HUNK 11: first external mask received for obj_id={obj_id} "
                f"coverage={coverage}px stamp={stamp_ns}"
            )

    @staticmethod
    def _stamp_ns(msg):
        return int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)

    def _event_put(self, buf, msg):
        stamp_ns = self._stamp_ns(msg)
        buf[stamp_ns] = msg
        buf.move_to_end(stamp_ns)
        while len(buf) > self._event_sync_qlen:
            buf.popitem(last=False)
        return stamp_ns

    def _event_nearest(self, buf, stamp_ns):
        if stamp_ns in buf:
            return stamp_ns, buf[stamp_ns], 0
        if not buf:
            return None, None, None
        best_ns = min(buf.keys(), key=lambda k: abs(k - stamp_ns))
        delta_ns = abs(best_ns - stamp_ns)
        if delta_ns <= self._event_sync_tol_ns:
            return best_ns, buf[best_ns], delta_ns
        return None, None, delta_ns

    def _event_on_rgb(self, msg):
        self._event_put(self._event_rgb_msgs, msg)

    def _event_on_info(self, msg):
        K = np.array(msg.k).reshape((3, 3))
        if self.cam_K is None:
            self._event_info_msg = msg
            self.camera_info_callback(msg)
            return
        if not np.allclose(K, self.cam_K, rtol=0.0, atol=1e-6):
            self._event_n_info_changed += 1
            self.get_logger().error(
                f"HUNK 22: event-sync CameraInfo K changed; keeping initial K "
                f"(n_changed={self._event_n_info_changed}). old={self.cam_K} new={K}"
            )
            return
        self._event_info_msg = msg

    def _event_on_depth(self, msg):
        stamp_ns = self._event_put(self._event_depth_msgs, msg)
        self._event_try_dispatch(stamp_ns, "depth")

    def _event_on_mask(self, msg):
        stamp_ns = self._event_put(self._event_mask_msgs, msg)
        self._event_try_dispatch(stamp_ns, "mask")

    def _event_try_dispatch(self, stamp_ns, source):
        if self._event_info_msg is None or self.cam_K is None:
            self._event_n_no_info += 1
            if self._event_n_no_info == 1 or self._event_n_no_info % 10 == 0:
                self.get_logger().warning(
                    f"HUNK 22: event-sync has {source} stamp={stamp_ns} but no "
                    f"CameraInfo/K yet (n_no_info={self._event_n_no_info})"
                )
            return

        depth_ns, depth_msg, depth_delta_ns = self._event_nearest(
            self._event_depth_msgs, stamp_ns
        )
        mask_ns, mask_msg, mask_delta_ns = self._event_nearest(
            self._event_mask_msgs, stamp_ns
        )
        if depth_msg is None or mask_msg is None:
            self._event_n_no_pair += 1
            if self._event_n_no_pair == 1 or self._event_n_no_pair % 10 == 0:
                nearest_ms = None
                if depth_msg is None and depth_delta_ns is not None:
                    nearest_ms = depth_delta_ns / 1e6
                if mask_msg is None and mask_delta_ns is not None:
                    nearest_ms = mask_delta_ns / 1e6
                self.get_logger().warning(
                    f"HUNK 22: event-sync no depth/mask pair for {source} "
                    f"stamp={stamp_ns}; tol_ms={self._event_sync_tol_ns/1e6:.1f} "
                    f"nearest_delta_ms={nearest_ms} "
                    f"buffers rgb={len(self._event_rgb_msgs)} "
                    f"depth={len(self._event_depth_msgs)} mask={len(self._event_mask_msgs)} "
                    f"(n_no_pair={self._event_n_no_pair})"
                )
            return

        # Prefer the depth source stamp as the canonical frame stamp; the
        # mask publisher should propagate the same RGB stamp.
        rgb_ns, rgb_msg, rgb_delta_ns = self._event_nearest(
            self._event_rgb_msgs, depth_ns
        )
        if rgb_msg is None:
            self._event_n_no_rgb += 1
            nearest_rgb_delta_ms = (
                "None" if rgb_delta_ns is None
                else f"{rgb_delta_ns/1e6:.1f}"
            )
            self.get_logger().warning(
                f"HUNK 22: event-sync no matching RGB for stamp={depth_ns}; "
                f"source={source} mask_delta_ms={abs(mask_ns-depth_ns)/1e6:.1f} "
                f"nearest_rgb_delta_ms={nearest_rgb_delta_ms} "
                f"buffers rgb={len(self._event_rgb_msgs)} "
                f"depth={len(self._event_depth_msgs)} mask={len(self._event_mask_msgs)} "
                f"queue={self._event_sync_qlen} tol_ms={self._event_sync_tol_ns/1e6:.1f} "
                f"(n_no_rgb={self._event_n_no_rgb})"
            )
            return

        self._event_depth_msgs.pop(depth_ns, None)
        self._event_mask_msgs.pop(mask_ns, None)
        self._event_rgb_msgs.pop(rgb_ns, None)
        self._event_n_matched += 1
        now_ns = self.get_clock().now().nanoseconds
        depth_lag_s = (now_ns - depth_ns) / 1e9 if now_ns > depth_ns else 0.0
        if self._event_n_matched == 1 or self._event_n_matched % 20 == 0:
            self.get_logger().info(
                f"HUNK 22 event-sync: matched (rgb,depth,mask) at stamp={depth_ns} "
                f"depth_lag={depth_lag_s:.3f}s "
                f"rgb_delta_ms={abs(rgb_ns-depth_ns)/1e6:.1f} "
                f"mask_delta_ms={abs(mask_ns-depth_ns)/1e6:.1f} "
                f"(n={self._event_n_matched})"
            )

        self._synced_callback_with_mask(
            rgb_msg, depth_msg, self._event_info_msg, mask_msg
        )

    def _synced_callback(self, img_msg, depth_msg, info_msg):
        """HUNK 8: single entry point for the synced (RGB, depth, info) triple.

        Order matters: camera_info_callback must run before process_images so
        cam_K is populated when register()/track_one() are dispatched. The bare
        upstream callbacks below are kept for backwards compatibility with
        anyone driving the node from a non-synced bag — but the synchronizer is
        the authoritative entry-point in this fork.
        """
        self._last_rgb_stamp = img_msg.header.stamp
        # Camera info first (sets cam_K if still None).
        self.camera_info_callback(info_msg)
        # Then RGB and depth.
        self.image_callback(img_msg)
        self.depth_callback(depth_msg)

    def _synced_callback_with_mask(self, img_msg, depth_msg, info_msg, mask_msg):
        """HUNK 20/22: synced entry point for RGB, depth, CameraInfo, mask.

        The mask is decoded and inserted into ``_external_masks`` with the
        mask msg's own stamp — close (within ``mask_sync_slop_s``) to the
        RGB stamp by construction of the synchronizer. HUNK 22's live path
        reaches this via the event-keyed synchronizer; the legacy wording
        still applies to older 4-leg ApproximateTimeSynchronizer callers.
        ``process_images`` then picks it up via the same legacy code path;
        the freshness gate there will pass trivially because we just inserted
        a fresh stamp.

        Counts ``self._n_synced_frames`` for end-of-run reporting. HUNK 22
        logs no-match cases explicitly before skipping a frame; older
        ApproximateTimeSynchronizer callers implicitly dropped them.
        """
        self._last_rgb_stamp = img_msg.header.stamp
        # Decode mask first so it's cached before process_images() runs.
        try:
            mask_np = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding="mono8")
        except Exception as e:
            self.get_logger().warning(
                f"HUNK 20: synced mask decode failed: {e}; falling back to "
                f"SAM2 for this frame."
            )
            mask_np = None
        if mask_np is not None:
            mask_stamp_ns = (
                int(mask_msg.header.stamp.sec) * 10**9
                + int(mask_msg.header.stamp.nanosec)
            )
            self._external_masks[1] = (
                mask_stamp_ns, (mask_np > 0).astype(np.uint8) * 255,
            )
            self._n_synced_frames += 1
            if self._n_synced_frames == 1:
                rgb_ns = (
                    int(img_msg.header.stamp.sec) * 10**9
                    + int(img_msg.header.stamp.nanosec)
                )
                self.get_logger().info(
                    f"HUNK 20/22: first synced mask frame rgb_stamp_ns={rgb_ns} "
                    f"mask_stamp_ns={mask_stamp_ns} "
                    f"delta_ms={(rgb_ns-mask_stamp_ns)/1e6:.1f}"
                )
            elif (self._n_synced_frames % 20) == 0:
                self.get_logger().info(
                    f"HUNK 20/22: n_synced={self._n_synced_frames}"
                )
        self.camera_info_callback(info_msg)
        self.image_callback(img_msg)
        self.depth_callback(depth_msg)

    def camera_info_callback(self, msg):
        if self.cam_K is None:  # Update cam_K only once to avoid redundant updates
            self.cam_K = np.array(msg.k).reshape((3, 3))
            self.get_logger().info(f"Camera intrinsic matrix initialized: {self.cam_K}")

    def image_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")

    def depth_callback(self, msg):
        # HUNK 7: auto-detect depth units. Sim publishes 32FC1 metres; RealSense
        # publishes 16UC1 millimetres. Upstream unconditionally divided by 1000,
        # which silently destroys sim depth. Detection rule below.
        enc = msg.encoding
        if enc == '32FC1':
            depth_m = self.bridge.imgmsg_to_cv2(msg, "32FC1").astype(np.float32)
            scale = 1.0
        elif enc in ('16UC1', 'mono16'):
            raw = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            depth_m = raw.astype(np.float32) / 1000.0
            scale = 1.0 / 1000.0
        elif enc == '32SC1':
            raw = self.bridge.imgmsg_to_cv2(msg, "32SC1")
            depth_m = raw.astype(np.float32) / 1000.0
            scale = 1.0 / 1000.0
        else:
            self.get_logger().warning(
                f"HUNK 7: unrecognized depth encoding '{enc}'; assuming metres (1.0 scale)."
            )
            depth_m = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').astype(np.float32)
            scale = 1.0

        if not self._logged_depth_units:
            finite = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
            med = float(np.median(finite)) if finite.size else float('nan')
            self.get_logger().info(
                f"HUNK 7: depth encoding='{enc}' scale={scale} median_m={med:.4f}"
            )
            self._logged_depth_units = True

        self.depth_image = depth_m
        self.process_images()

    def process_images(self):
        if self.color_image is None or self.depth_image is None or self.cam_K is None:
            return

        H, W = self.color_image.shape[:2]
        color = cv2.resize(self.color_image, (W, H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(self.depth_image, (W, H), interpolation=cv2.INTER_NEAREST)
        # HUNK 12 (Phase 1.G parity): the upstream daemon clamped
        # depth<0.1m → 0 to mute the close-range gripper. The batch
        # runner.py baseline that produced 87/90 NIC + 87/90 SC just
        # zeros out non-finite depth — clamping creates a hole at the
        # cable plug that perturbs FP.register's convergence.
        depth[~np.isfinite(depth)] = 0.0

        # HUNK 9: in reset mode, tear down the per-frame pose_estimations and
        # re-bootstrap each frame. Do this BEFORE the bootstrap check so the
        # `_needs_reregister` flag flips True for the bootstrap block.
        if self._reset_each_frame and self._frame_count > 0:
            self.pose_estimations = {}
            self._needs_reregister = True

        # HUNK 3: bootstrap path replaced with a deterministic strategy.
        # Re-entered on every frame in reset mode (HUNK 9).
        if self._needs_reregister:
            # HUNK 11: prefer external masks when fresh. The mask topic
            # is the operator-driven path that bypasses SAM2 + the
            # ``largest`` heuristic — required for SC where SAM2 picks
            # the cable plug instead of the port at wrist-camera
            # working distance.
            now_ns = int(self._last_rgb_stamp.sec) * 10**9 + int(self._last_rgb_stamp.nanosec) \
                     if self._last_rgb_stamp is not None else 0
            external_assigned: dict[int, dict] = {}
            for mesh_idx in range(len(self.meshes)):
                obj_id = mesh_idx + 1
                cached = self._external_masks.get(obj_id)
                if cached is None:
                    # HUNK 20 (Agent Z): track when we have no mask at all.
                    if not self._mask_in_sync_active:
                        self._n_skipped_no_mask += 1
                    continue
                stamp_ns, mask = cached
                if now_ns > 0 and abs(now_ns - stamp_ns) > self._external_mask_max_age_ns:
                    # HUNK 20 (Agent Z): instrument stale-mask skips so we
                    # can see the temporal-mismatch failure rate without
                    # ssh-ing into the daemon. Only counted in the legacy
                    # path; mask_in_sync drops stale frames implicitly via
                    # the synchronizer.
                    if not self._mask_in_sync_active:
                        self._n_skipped_stale_mask += 1
                        if (self._n_skipped_stale_mask % 10) == 1:
                            self.get_logger().warning(
                                f"HUNK 20: skipping frame — cached mask stale "
                                f"({(now_ns - stamp_ns)/1e6:.0f}ms vs cap "
                                f"{self._external_mask_max_age_ns/1e6:.0f}ms) "
                                f"obj_id={obj_id} "
                                f"(n_skipped_stale={self._n_skipped_stale_mask})"
                            )
                    continue
                # Resize to match color image if shapes differ.
                if mask.shape[:2] != (H, W):
                    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                external_assigned[mesh_idx] = {'mask': mask, 'box': None, 'contour': None}

            if external_assigned:
                # Skip SAM2 entirely; assignment is direct.
                assigned = {i: external_assigned.get(i) for i in range(len(self.meshes))}
                self.get_logger().info(
                    f"HUNK 11: using external masks for "
                    f"{sorted(external_assigned.keys())}"
                )
            elif self._mask_in_sync_active:
                # HUNK 20 (Agent Z): in synced mode we expect every triggered
                # frame to carry a fresh external mask by construction. If
                # we land here, something violated the assumption (e.g.
                # multi-mesh path that the param-gate was supposed to
                # exclude). Bail explicitly rather than silently fall back
                # to SAM2.
                self.get_logger().warning(
                    "HUNK 20: mask_in_sync active but external_assigned empty; "
                    "skipping frame (do NOT cold-start SAM2 in sync mode)."
                )
                return
            else:
                res = self.seg_model.predict(color, verbose=False)[0]
                if not res or len(res) == 0:
                    self.get_logger().warning("HUNK 3: SAM2 produced no masks; retry next frame.")
                    return

                objects_to_track = []
                for r in res:
                    for c in r:
                        if c.masks is None or len(c.masks.xy) == 0:
                            continue
                        mask = np.zeros((H, W), np.uint8)
                        contour = c.masks.xy[-1].astype(np.int32).reshape(-1, 1, 2)
                        cv2.drawContours(mask, [contour], -1, (255, 255, 255), cv2.FILLED)
                        objects_to_track.append({
                            'mask': mask,
                            'box': c.boxes.xyxy.tolist()[-1] if len(c.boxes.xyxy) else None,
                            'contour': contour,
                        })

                if not objects_to_track:
                    self.get_logger().warning("HUNK 3: no usable mask candidates; retry next frame.")
                    return

                assigned = self._assign_masks_to_meshes(objects_to_track, color)
            temporary_pose_estimations = {}
            for mesh_idx, obj in assigned.items():
                if obj is None:
                    continue
                temp_mesh = self.meshes[mesh_idx]
                temp_to_origin, _ = self.bounds[mesh_idx]
                # HUNK 14 (Phase 1.G further parity):
                #   - Cast pts/normals to float32 (batch baseline does;
                #     trimesh defaults float64 which can perturb FP's
                #     internal hypothesis ranking).
                #   - glctx=None lets FP create its own context per
                #     instance, matching the batch baseline.
                #   - symmetry_tfs=None explicit (matches batch).
                pose_est = FoundationPose(
                    model_pts=np.asarray(temp_mesh.vertices, dtype=np.float32),
                    model_normals=np.asarray(temp_mesh.vertex_normals, dtype=np.float32),
                    symmetry_tfs=None,
                    mesh=temp_mesh,
                    scorer=self.scorer,
                    refiner=self.refiner,
                    glctx=None,
                )
                temporary_pose_estimations[mesh_idx] = {
                    'pose_est': pose_est,
                    'mask': obj['mask'],
                    'to_origin': temp_to_origin,
                }

            if not temporary_pose_estimations:
                # No mesh was assigned a usable mask. Leave _needs_reregister
                # TRUE so the next frame retries. Do NOT advance _frame_count.
                self.get_logger().warning(
                    "HUNK 3: assignment yielded zero pose estimators; retry next frame."
                )
                return

            self.pose_estimations = temporary_pose_estimations
            self._needs_reregister = False

        visualization_image = np.copy(color)

        for idx, data in self.pose_estimations.items():
            pose_est = data['pose_est']
            obj_mask = data['mask']
            to_origin = data['to_origin']
            obj_id_local = idx + 1
            # HUNK 17 (Agent R, 2026-05-18): seeded_track branch. If a
            # fresh per-frame seed is available, skip register()'s
            # multimodal hypothesis search and refine via track_one()
            # from the seed. Matches the offline Phase 3b seeded_track
            # pattern that reaches 3.88mm. Falls through to register
            # when no fresh seed is in cache.
            seeded = None
            seed_pending_tf = False
            if self._pose_mode == 'seeded_track' and _HAVE_TORCH:
                seeded = self._fresh_seed_T_cam_obj(obj_id_local)
                # HUNK 21 (2026-05-19): if a world seed is cached but
                # composition failed (TF not yet covering this RGB stamp),
                # skip this frame rather than fall through to register(),
                # which OOMs at 252-candidate hypothesis search on 20 GB
                # GPUs. The TF buffer fills as bag-replay progresses;
                # the next frame whose stamp is within /tf coverage
                # succeeds. Skipping a few early frames is preferable
                # to an OOM cascade.
                if seeded is None and self._latest_seed_T_world_obj is not None:
                    seed_pending_tf = True
            if seed_pending_tf:
                # Don't run register() — daemon will eventually publish a
                # seed whose composition succeeds.
                continue
            if seeded is not None:
                seed_T, seed_age_ns = seeded
                # HUNK 18 (Agent S, 2026-05-18): convert seed_T from the
                # link-anchored T_cam_obj that Agent Q's daemon publishes
                # into the AABB-centered frame that FoundationPose's
                # `pose_last` lives in internally. FP.reset_object recenters
                # the mesh by model_center=(min+max)/2 (estimater.py:44-51)
                # and stores poses in centered coords; register/track_one
                # OUTPUTS get converted back via `@ get_tf_to_centered_mesh()`
                # (estimater.py:233,268) on the way out, but the INPUT
                # pose_last must already be centered. Without this inverse
                # conversion, the seed is offset by model_center → ~80mm Z
                # bias on NIC (model_center=(-7.95,-12.50,+86.50)mm). HUNK
                # 17 missed this on the way in; the refiner can only close
                # a few mm per iter so track_refine_iter=2 left ~75mm
                # residual.
                seed_T_t = _torch.as_tensor(seed_T, device="cuda", dtype=_torch.float)
                tf_centered_to_link = pose_est.get_tf_to_centered_mesh()
                # right-multiplying by inv() converts a link-frame pose into
                # the centered-mesh frame that FP expects in pose_last.
                pose_est.pose_last = seed_T_t @ _torch.linalg.inv(tf_centered_to_link)
                pose = pose_est.track_one(
                    rgb=color, depth=depth, K=self.cam_K,
                    iteration=args.track_refine_iter,
                )
                pose_est.is_register = True
                center_pose = pose
                if not getattr(self, "_seeded_pub_logged", False):
                    self.get_logger().info(
                        f"HUNK 17: seeded_track first pose published "
                        f"obj_id={obj_id_local} seed_age_ms={seed_age_ns/1e6:.1f}"
                    )
                    self._seeded_pub_logged = True
                self.publish_pose_stamped(
                    center_pose,
                    f"{self._frame_id_prefix}_{idx}",
                    f"{self._pose_topic_prefix}_{idx+1}",
                    stamp=self._last_rgb_stamp,
                )
                visualization_image = self.visualize_pose(visualization_image, center_pose, idx)
                self.i += 1
                continue
            if pose_est.is_register and not self._reset_each_frame:
                pose = pose_est.track_one(rgb=color, depth=depth, K=self.cam_K, iteration=args.track_refine_iter)
                # HUNK 15 (track_one parity with HUNK 13): drop the
                # inv(to_origin) multiply here too. FP.register and
                # FP.track_one both return poses in the same frame
                # convention — mesh-native, since mesh=mesh was passed
                # at construction. The OBB-to-mesh re-shift introduces
                # the same ~88mm error that HUNK 13 fixed on register.
                # Today masked by reset_each_frame=true (which never
                # takes this branch), but Phase 4 tracker mode would
                # exercise this path and resurrect the bug.
                center_pose = pose

                self.publish_pose_stamped(
                    center_pose,
                    f"{self._frame_id_prefix}_{idx}",
                    f"{self._pose_topic_prefix}_{idx+1}",
                    stamp=self._last_rgb_stamp,
                )

                visualization_image = self.visualize_pose(visualization_image, center_pose, idx)
            else:
                pose = pose_est.register(K=self.cam_K, rgb=color, depth=depth, ob_mask=obj_mask, iteration=args.est_refine_iter)
                # HUNK 9: in reset mode (or first frame in normal mode), also
                # publish the register() result — we'd otherwise drop the
                # bootstrap pose entirely in sparse-frame mode.
                # HUNK 13 (Phase 1.G parity): drop the inv(to_origin)
                # multiply. The batch runner.py (which produced the
                # 87/90 NIC + 87/90 SC baselines) publishes pose
                # directly from FP.register, NOT pose @ inv(to_origin).
                # FP's register already returns the pose in the
                # mesh-frame convention since `mesh=mesh` was passed
                # at construction; the extra OBB-to-mesh transform
                # double-centers and offsets the result by ~88mm on
                # NIC (mesh centroid at 128mm Z).
                center_pose = pose
                self.publish_pose_stamped(
                    center_pose,
                    f"{self._frame_id_prefix}_{idx}",
                    f"{self._pose_topic_prefix}_{idx+1}",
                    stamp=self._last_rgb_stamp,
                )
                visualization_image = self.visualize_pose(visualization_image, center_pose, idx)
            self.i += 1

        # HUNK 9: bump per-frame counter (distinct from upstream's per-object
        # self.i). Frame-bounded operations use this counter.
        self._frame_count += 1

        # Headless-safe: skip cv2.imshow when no display is available. The
        # conda-vendored OpenCV has only the xcb Qt plugin (no offscreen),
        # so attempting imshow without a display crashes the process with
        # `qt.qpa.plugin: Could not find the Qt platform plugin "offscreen"`.
        # The visualization is human-debug only; the pose pub above is the
        # functional output.
        # (Surfaced & patched by Agent P, 2026-05-18, during NIC live-chain validation.)
        if os.environ.get("DISPLAY"):
            cv2.imshow('Pose Estimation & Tracking', visualization_image[..., ::-1])
            cv2.waitKey(1)

    # HUNK 3: assignment helpers — methods on PoseEstimationNode (NOT module
    # globals). Each returns {mesh_idx: candidate or None}.
    def _assign_masks_to_meshes(self, candidates, color):
        """Deterministic mask-to-mesh assignment. Strategy semantics:

          'largest'      — for each mesh i (argv order), bind to the i-th
                           largest still-unbound candidate mask.
          'gt_bbox_iou'  — bind each mesh to the candidate with highest IoU
                           against the projected GT bbox for that mesh's
                           object (training/benchmark only; uses /scoring/tf).
                           Falls back to 'class_filter' on failure.
          'class_filter' — assumes SAM2 candidates were filtered upstream by
                           a class-aware detector (RT-DETR feed); binds by
                           the order the candidates arrived.
        """
        strat = self._assign_strategy
        n_meshes = len(self.meshes)
        if strat == 'largest':
            ranked = sorted(candidates, key=lambda o: int(o['mask'].sum()), reverse=True)
            return {i: ranked[i] if i < len(ranked) else None for i in range(n_meshes)}
        if strat == 'gt_bbox_iou':
            try:
                return self._assign_by_gt_iou(candidates)
            except Exception as e:
                self.get_logger().warning(
                    f"HUNK 3: gt_bbox_iou unavailable ({e}); falling back to class_filter."
                )
                return self._assign_by_class(candidates)
        if strat == 'class_filter':
            return self._assign_by_class(candidates)
        raise ValueError(f"unknown assign strategy {strat}")

    def _assign_by_gt_iou(self, candidates):
        """Concrete implementation lives in the AIC adapter
        (aic_vision/aic_vision/estimators/isaacros_foundationpose.py). The
        adapter republishes the projected GT bbox on /aic_isaacros/gt_bbox_*
        — we keep the upstream tree decoupled here. Raising NotImplementedError
        triggers _assign_masks_to_meshes' fallback path.
        """
        raise NotImplementedError(
            "HUNK 3: gt_bbox_iou requires the AIC adapter to publish /aic_isaacros/gt_bbox_*."
        )

    def _assign_by_class(self, candidates):
        ranked = sorted(candidates, key=lambda o: int(o['mask'].sum()), reverse=True)
        n = len(self.meshes)
        return {i: ranked[i] if i < len(ranked) else None for i in range(n)}

    def visualize_pose(self, image, center_pose, idx):
        # AIC PATCH (2026-05-17): opencv-python 4.13 in our kilted conda
        # env is stricter about tuple types passed to arrowedLine —
        # numpy.int64 tuples no longer auto-cast. Wrap in try/except so
        # a viz crash doesn't take down register + publish. Our own
        # annotated_image_publisher in the chain handles operator viz
        # via /aic_vision/annotated_image.
        try:
            bbox = self.bboxes[idx % len(self.bboxes)]
            vis = draw_posed_3d_box(self.cam_K, img=image, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=self.cam_K, thickness=3, transparency=0, is_input_rgb=True)
            return vis
        except Exception as e:
            if not getattr(self, "_viz_warned", False):
                self.get_logger().warning(f"visualize_pose disabled (cv2 type-check): {e}")
                self._viz_warned = True
            return image

    def publish_pose_stamped(self, center_pose, frame_id, topic_name, stamp=None):
        if topic_name not in self.pose_publishers:
            self.pose_publishers[topic_name] = self.create_publisher(PoseStamped, topic_name, 10)

        # Convert the center_pose matrix to a PoseStamped message
        pose_stamped_msg = PoseStamped()
        # HUNK 5: prefer the input RGB header stamp; fall back to clock.now()
        # only if the caller didn't pass one through (defensive).
        pose_stamped_msg.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        pose_stamped_msg.header.frame_id = frame_id

        # Convert center_pose to the pose format
        position = center_pose[:3, 3]
        rotation_matrix = center_pose[:3, :3]
        quaternion = R.from_matrix(rotation_matrix).as_quat()

        # Combine position and quaternion into a single array
        pose_array = np.concatenate((position, quaternion))

        # Apply transformation to convert from camera to base frame
        # HUNK 4: transformation() is now identity (see cam_2_base_transform.py).
        # The AIC adapter does the link-frame composition downstream.
        transformed_pose = transformation(pose_array)

        # Populate PoseStamped message with transformed pose
        pose_stamped_msg.pose.position.x = transformed_pose[0]
        pose_stamped_msg.pose.position.y = transformed_pose[1]
        pose_stamped_msg.pose.position.z = transformed_pose[2]

        pose_stamped_msg.pose.orientation.w = transformed_pose[3]
        pose_stamped_msg.pose.orientation.x = transformed_pose[4]
        pose_stamped_msg.pose.orientation.y = transformed_pose[5]
        pose_stamped_msg.pose.orientation.z = transformed_pose[6]

        # Publish the transformed pose
        self.pose_publishers[topic_name].publish(pose_stamped_msg)

def main(args=None):
    # HUNK 2: prefer the --meshes CLI list; fall back to the Tk picker only if
    # a human user explicitly skipped the flag AND $DISPLAY is set. Refuses to
    # spawn the Tk dialog in headless contexts (Docker / CI).
    cli_meshes = globals().get('args')
    if cli_meshes is not None and getattr(cli_meshes, 'meshes', None):
        new_file_paths = list(cli_meshes.meshes)
    else:
        source_directory = "demo_data"
        file_paths = glob.glob(os.path.join(source_directory, '**', '*.obj'), recursive=True) + \
                     glob.glob(os.path.join(source_directory, '**', '*.stl'), recursive=True) + \
                     glob.glob(os.path.join(source_directory, '**', '*.STL'), recursive=True)
        if not os.environ.get('DISPLAY'):
            raise RuntimeError(
                "HUNK 2: no --meshes passed and DISPLAY is unset; refusing to spawn "
                "Tkinter FileSelectorGUI in a headless context."
            )
        new_file_paths = rearrange_files(file_paths)

    rclpy.init(args=args)
    node = PoseEstimationNode(new_file_paths)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
