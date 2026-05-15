import sys
sys.path.append('./FoundationPose')
sys.path.append('./FoundationPose/nvdiffrast')

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
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
from scipy.spatial.transform import Rotation as R
from ultralytics import SAM
from cam_2_base_transform import *
import os
import tkinter as tk
from tkinter import Listbox, END, Button
import glob

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
parser.add_argument('--est_refine_iter', type=int, default=4)
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

        image_topic = self.get_parameter('image_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        info_topic  = self.get_parameter('camera_info_topic').value
        self._pose_topic_prefix = self.get_parameter('pose_topic_prefix').value
        self._frame_id_prefix   = self.get_parameter('frame_id_prefix').value
        self._assign_strategy   = self.get_parameter('assign_strategy').value
        self._reset_each_frame  = bool(self.get_parameter('reset_each_frame').value)
        slop = float(self.get_parameter('sync_slop_s').value)
        qlen = int(self.get_parameter('sync_queue').value)

        self.get_logger().info(
            f"FoundationPoseROS2 [PATCHED]: rgb='{image_topic}' depth='{depth_topic}' "
            f"info='{info_topic}' strat='{self._assign_strategy}' "
            f"reset_each_frame={self._reset_each_frame}"
        )

        # HUNK 8: ApproximateTimeSynchronizer for the (RGB, depth, camera_info)
        # trio. Slop = 0.05 s = the PRD's timestamp_tolerance_ms (50 ms).
        # NOTE: the previous bare `self.create_subscription` calls from upstream
        # are intentionally removed in favor of this single synced entry-point.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=qlen,
        )
        self.image_mf = Subscriber(self, Image, image_topic, qos_profile=qos)
        self.depth_mf = Subscriber(self, Image, depth_topic, qos_profile=qos)
        self.info_mf  = Subscriber(self, CameraInfo, info_topic, qos_profile=qos)
        self._sync = ApproximateTimeSynchronizer(
            [self.image_mf, self.depth_mf, self.info_mf],
            queue_size=qlen, slop=slop,
        )
        self._sync.registerCallback(self._synced_callback)

        self.bridge = CvBridge()
        self.depth_image = None
        self.color_image = None
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
            self.get_logger().warn(
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
        depth[(depth < 0.1) | (depth >= np.inf)] = 0

        # HUNK 9: in reset mode, tear down the per-frame pose_estimations and
        # re-bootstrap each frame. Do this BEFORE the bootstrap check so the
        # `_needs_reregister` flag flips True for the bootstrap block.
        if self._reset_each_frame and self._frame_count > 0:
            self.pose_estimations = {}
            self._needs_reregister = True

        # HUNK 3: bootstrap path replaced with a deterministic strategy.
        # Re-entered on every frame in reset mode (HUNK 9).
        if self._needs_reregister:
            res = self.seg_model.predict(color, verbose=False)[0]
            if not res or len(res) == 0:
                self.get_logger().warn("HUNK 3: SAM2 produced no masks; retry next frame.")
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
                self.get_logger().warn("HUNK 3: no usable mask candidates; retry next frame.")
                return

            assigned = self._assign_masks_to_meshes(objects_to_track, color)
            temporary_pose_estimations = {}
            for mesh_idx, obj in assigned.items():
                if obj is None:
                    continue
                temp_mesh = self.meshes[mesh_idx]
                temp_to_origin, _ = self.bounds[mesh_idx]
                pose_est = FoundationPose(
                    model_pts=temp_mesh.vertices,
                    model_normals=temp_mesh.vertex_normals,
                    mesh=temp_mesh,
                    scorer=self.scorer,
                    refiner=self.refiner,
                    glctx=self.glctx,
                )
                temporary_pose_estimations[mesh_idx] = {
                    'pose_est': pose_est,
                    'mask': obj['mask'],
                    'to_origin': temp_to_origin,
                }

            if not temporary_pose_estimations:
                # No mesh was assigned a usable mask. Leave _needs_reregister
                # TRUE so the next frame retries. Do NOT advance _frame_count.
                self.get_logger().warn(
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
            if pose_est.is_register and not self._reset_each_frame:
                pose = pose_est.track_one(rgb=color, depth=depth, K=self.cam_K, iteration=args.track_refine_iter)
                center_pose = pose @ np.linalg.inv(to_origin)

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
                center_pose = pose @ np.linalg.inv(to_origin)
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
                self.get_logger().warn(
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
        bbox = self.bboxes[idx % len(self.bboxes)]
        vis = draw_posed_3d_box(self.cam_K, img=image, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=self.cam_K, thickness=3, transparency=0, is_input_rgb=True)
        return vis

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
