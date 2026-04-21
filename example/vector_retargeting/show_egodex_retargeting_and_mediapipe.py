import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import tyro
import yaml
from scipy.spatial.transform import Rotation

try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False

# ── sys.path 配置 ──────────────────────────────────────────────────────────────
_ROOT = str(Path(__file__).absolute().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_EGODEX = str(Path(__file__).absolute().parent.parent.parent / "egodex")
if _EGODEX not in sys.path:
    sys.path.insert(1, _EGODEX)

from dex_retargeting.constants import (
    RobotName, RetargetingType, HandType, get_default_config_path
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.robot_wrapper import RobotWrapper
from dex_retargeting.seq_retarget import SeqRetargeting
from egodex.simple_dataset import SimpleDataset
from egodex.utils.skeleton_tfs import NEW_LEFT_FINGERS, NEW_RIGHT_FINGERS

NEW_QUERY_TFS = (
    ['leftHand'] + NEW_LEFT_FINGERS + ['leftForearm'] +
    ['rightHand'] + NEW_RIGHT_FINGERS + ['rightForearm']
)

T_robot_avp = np.array([[0, 0, -1, 0], # rot z -90° and rot y -90° fixed axis: Roty(-90d)*Rotz(-90d)
                        [-1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])

# rot x 90° and rot z 90° fixed axis of Wrist(Robot Convention): Rotz(90°)*Rotx(90°). 
# For corresponding to the oritention from avp wrist to robot model default orientation that defined in the urdf and mesh.
T_wrist_left_2world = np.eye(4)
T_wrist_left_2world[:3, :3] = (Rotation.from_euler('z', 90, degrees=True) * Rotation.from_euler('x', 90, degrees=True)).as_matrix()  
print("T_wrist_left_2world:\n", T_wrist_left_2world)


# rot x -90° and rot y 180° fixed axis of Wrist(Robot Convention): Roty(180°)*Rotx(-90°)
T_wrist_right_2world = np.eye(4)
T_wrist_right_2world[:3, :3] = (Rotation.from_euler('z', -90, degrees=True) * Rotation.from_euler('x', 90, degrees=True)).as_matrix()  
print("T_wrist_right_2world:\n", T_wrist_right_2world)
# ──────────────────────────────────────────────────────────────────────────────
# 人手骨骼连接定义（对应 tfs 的前/后半部分，各 26 个节点）
# idx 0: Hand, 1-4: Thumb, 5-9: Index, 10-14: Middle, 15-19: Ring, 20-24: Little, 25: Forearm
# ──────────────────────────────────────────────────────────────────────────────
HAND_BONE_PAIRS = [
    (25, 0),
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8), (8, 9),
    (0, 10), (10, 11), (11, 12), (12, 13), (13, 14),
    (0, 15), (15, 16), (16, 17), (17, 18), (18, 19),
    (0, 20), (20, 21), (21, 22), (22, 23), (23, 24),
]


def bone_strips(positions: np.ndarray) -> list:
    return [[positions[i], positions[j]] for i, j in HAND_BONE_PAIRS]


# ──────────────────────────────────────────────────────────────────────────────
# URDF 解析：提取每个 link 的 visual mesh 路径
# ──────────────────────────────────────────────────────────────────────────────

def parse_urdf_link_meshes(urdf_path: str) -> dict:
    """
    解析 URDF，返回 {link_name: [mesh_abs_path, ...]} 的字典。
    只处理 <visual><geometry><mesh> 节点。
    """
    urdf_dir = Path(urdf_path).parent
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    link_meshes: dict[str, list[str]] = {}
    for link in root.findall("link"):
        name = link.get("name")
        meshes = []
        for visual in link.findall("visual"):
            geom = visual.find("geometry")
            if geom is not None:
                mesh_el = geom.find("mesh")
                if mesh_el is not None:
                    filename = mesh_el.get("filename", "")
                    # 处理 package:// 前缀
                    filename = filename.replace("package://", "")
                    abs_path = str((urdf_dir / filename).resolve())
                    meshes.append(abs_path)
        if meshes:
            link_meshes[name] = meshes
    return link_meshes


# ──────────────────────────────────────────────────────────────────────────────
# Rerun 工具
# ──────────────────────────────────────────────────────────────────────────────

def mat4_to_rr_transform(mat: np.ndarray) -> rr.Transform3D:
    """4×4 齐次矩阵 → rr.Transform3D（xyzw 四元数）。"""
    pos = mat[:3, 3]
    q_xyzw = Rotation.from_matrix(mat[:3, :3]).as_quat()
    return rr.Transform3D(
        translation=pos,
        rotation=rr.Quaternion(xyzw=q_xyzw),
    )


def log_axes(entity: str, mat: np.ndarray, size: float = 0.05):
    """
    在 mat 定义的坐标系下记录三根坐标轴（X=红, Y=绿, Z=蓝），便于调试旋转偏差。
    """
    origin = mat[:3, 3]
    x_axis = origin + mat[:3, 0] * size
    y_axis = origin + mat[:3, 1] * size
    z_axis = origin + mat[:3, 2] * size
    rr.log(f"{entity}/axes",
           rr.LineStrips3D(
               [[origin, x_axis], [origin, y_axis], [origin, z_axis]],
               colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
           ))


def log_robot_meshes_once(entity_prefix: str, link_meshes: dict):
    """
    将机器人各 link 的 STL mesh 一次性记录到 Rerun（geometry 不随时间变化）。
    entity 路径：{entity_prefix}/{link_name}/mesh_{i}
    """
    for link_name, paths in link_meshes.items():
        for i, mesh_path in enumerate(paths):
            rr.log(
                f"{entity_prefix}/{link_name}/mesh_{i}",
                rr.Asset3D(path=mesh_path),
                static=True,
            )


def log_robot_transforms(
    entity_prefix: str,
    robot: RobotWrapper,
    q_full: np.ndarray,
    wrist_world: np.ndarray,
):
    """
    计算 FK，为每个 pinocchio frame 记录世界系变换。
    entity 路径与 log_robot_meshes_once 保持一致：{entity_prefix}/{link_name}
    """
    robot.compute_forward_kinematics(q_full)
    for link_id, link_name in enumerate(robot.link_names):
        local_pose = robot.get_link_pose(link_id)    # robot base frame → link
        world_pose = wrist_world @ local_pose        # world frame → link
        rr.log(
            f"{entity_prefix}/{link_name}",
            mat4_to_rr_transform(world_pose),
        )


# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe 双手检测器
# ──────────────────────────────────────────────────────────────────────────────

_OPERATOR2MANO_RIGHT = np.array([[0, 0, -1], [-1, 0, 0], [0,  1, 0]])
_OPERATOR2MANO_LEFT  = np.array([[0, 0, -1], [ 1, 0, 0], [0, -1, 0]])


class DualHandDetector:
    """
    基于 MediaPipe 的双手检测器（非自拍模式，selfie=False）。
    MediaPipe 在非镜像图像中：
        "Left"  标签 → 图像左侧 → 实际右手 → 使用 OPERATOR2MANO_RIGHT
        "Right" 标签 → 图像右侧 → 实际左手 → 使用 OPERATOR2MANO_LEFT
    """

    def __init__(self, min_detection_confidence: float = 0.6,
                 min_tracking_confidence: float = 0.6):
        if not _MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe 未安装，请运行：pip install mediapipe")
        self.detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, rgb: np.ndarray):
        """
        Args:
            rgb: (H, W, 3) uint8 numpy array
        Returns:
            left_result:  (joint_pos (21,3), kp2d) 或 None
            right_result: (joint_pos (21,3), kp2d) 或 None
        """
        results = self.detector.process(rgb)
        left_result, right_result = None, None
        if not results.multi_hand_landmarks:
            return left_result, right_result
        for i, handedness in enumerate(results.multi_handedness):
            label = handedness.classification[0].label  # "Left" or "Right"
            kp3d  = results.multi_hand_world_landmarks[i]
            kp2d  = results.multi_hand_landmarks[i]
            arr = np.array([[kp3d.landmark[j].x,
                             kp3d.landmark[j].y,
                             kp3d.landmark[j].z] for j in range(21)])
            arr -= arr[0:1]  # 以手腕为原点
            rot = self._estimate_frame(arr)
            if label == "Left":  # → 实际右手
                right_result = (arr @ rot @ _OPERATOR2MANO_RIGHT, kp2d)
            else:               # → 实际左手
                left_result  = (arr @ rot @ _OPERATOR2MANO_LEFT,  kp2d)
        return left_result, right_result

    @staticmethod
    def draw_on_image(image: np.ndarray, kp2d) -> np.ndarray:
        """在 image 上就地绘制 MediaPipe 骨骼，返回 image。"""
        mp.solutions.drawing_utils.draw_landmarks(
            image, kp2d,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style(),
        )
        return image

    @staticmethod
    def _estimate_frame(kp: np.ndarray) -> np.ndarray:
        """与 SingleHandDetector.estimate_frame_from_hand_points 等价。"""
        pts   = kp[[0, 5, 9], :]
        x_vec = pts[0] - pts[2]
        pts_c = pts - np.mean(pts, axis=0, keepdims=True)
        _, _, v = np.linalg.svd(pts_c)
        normal = v[2]
        x = x_vec - np.sum(x_vec * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)
        if np.sum(z * (pts[1] - pts[2])) < 0:
            normal *= -1
            z      *= -1
        return np.stack([x, normal, z], axis=1)


# ──────────────────────────────────────────────────────────────────────────────
# 核心流程
# ──────────────────────────────────────────────────────────────────────────────

def retarget_egodex(dataset: SimpleDataset, robot_name: RobotName, num_episodes: int, use_mediapipe: bool = False):
    tf2idx = {k: i for i, k in enumerate(NEW_QUERY_TFS)}
    num_transitions = dataset.cumulative_len[num_episodes - 1]
    print(f"Total transitions to process: {num_transitions}")

    cfg_file = Path(__file__).parent.parent.parent / "src/dex_retargeting/configs/teleop" / f"{robot_name.name}_hand_dexpilot.yml"
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    left_config  = RetargetingConfig.from_dict(cfg['left'])
    right_config = RetargetingConfig.from_dict(cfg['right'])
    left_retargeting:  SeqRetargeting = left_config.build()
    right_retargeting: SeqRetargeting = right_config.build()

    left_kp_idx  = left_retargeting.optimizer.target_link_human_indices
    right_kp_idx = right_retargeting.optimizer.target_link_human_indices
    left_robot   = left_retargeting.optimizer.robot
    right_robot  = right_retargeting.optimizer.robot
    left_wrist_link_name = left_config.wrist_link_name
    right_wrist_link_name = right_config.wrist_link_name

    # ── 解析 URDF mesh，一次性写入 Rerun（static=True） ────────────────────────
    left_link_meshes  = parse_urdf_link_meshes(left_config.urdf_path)
    right_link_meshes = parse_urdf_link_meshes(right_config.urdf_path)

    # ── 机器人 mesh 静态写入（几何体不变，只有 Transform 随时间变化）──────────────
    log_robot_meshes_once("robot/left",  left_link_meshes)
    log_robot_meshes_once("robot/right", right_link_meshes)

    # ── MediaPipe 初始化（可选）──────────────────────────────────────────────────
    if use_mediapipe:
        mp_detector = DualHandDetector()
        # 使用不含自定义 human_indices 的单手配置（DexPilot 默认走 21 关节 MediaPipe 格式）
        mp_left_cfg  = RetargetingConfig.load_from_file(
            get_default_config_path(robot_name, RetargetingType.dexpilot, HandType.left))
        mp_right_cfg = RetargetingConfig.load_from_file(
            get_default_config_path(robot_name, RetargetingType.dexpilot, HandType.right))
        mp_left_retargeting  = mp_left_cfg.build()
        mp_right_retargeting = mp_right_cfg.build()
        mp_left_kp_idx  = mp_left_retargeting.optimizer.target_link_human_indices
        mp_right_kp_idx = mp_right_retargeting.optimizer.target_link_human_indices
        mp_left_robot   = mp_left_retargeting.optimizer.robot
        mp_right_robot  = mp_right_retargeting.optimizer.robot
        # URDF 与 EgoDex 相同，复用已解析的 mesh 路径，挂载到不同 entity 前缀下
        log_robot_meshes_once("mediapipe/robot/left",  left_link_meshes)
        log_robot_meshes_once("mediapipe/robot/right", right_link_meshes)

    # ── 逐帧循环 ──────────────────────────────────────────────────────────────
    # wrist_init_poses = np.tile(np.eye(4, dtype=np.float64), (2, 1, 1))

    for i in range(num_transitions):
        rr.set_time_sequence("frame", i)

        tfs_avp, cam_ext, cam_int, cam_img, _, _ = dataset[i]
        episodes_idx, start_ts = dataset._locate_transition(i)
        tfs = T_robot_avp @ tfs_avp

        # if i == start_ts:
        print(f"Episode {episodes_idx} starts at transition {i}.start_ts is {start_ts}")
        wrist_init_poses = tfs[[tf2idx['leftHand'], tf2idx['rightHand']], :, :]

        left_wrist_pose  = tfs[tf2idx['leftHand']]
        right_wrist_pose = tfs[tf2idx['rightHand']]

        # ── Retargeting ───────────────────────────────────────────────────────
        tfs_left, tfs_right = np.split(tfs, 2, axis=0)
        pos_left_world  = tfs_left[:, :3, 3]    # 世界系，用于骨骼可视化
        pos_right_world = tfs_right[:, :3, 3]

        # DexPilot 优化器的 robot FK 在 base_link (wrist)局部坐标系下计算，
        # 输入向量必须与之一致，即转到 robot base_link 坐标系。
        #
        # 坐标系链（右乘约定）：
        #   world  ← R_wrist ← T_wrist_*_2world ← base_link
        # 逆变换（world → base_link）：
        #   v_base = T_wrist_2world[:3,:3].T  @  R_wrist.T  @  v_world
        #
        # 对比 MediaPipe 通道：
        #   joint_pos = raw_kp @ mediapipe_wrist_rot @ operator2mano
        #   其中 mediapipe_wrist_rot @ operator2mano 就扮演了同样的角色，
        #   把关键点从摄像机空间转到 base_link 等价坐标系。
        R_wrist_left  = tfs_left[0, :3, :3]    # AVP 左手腕在 robot-world 系的旋转
        R_wrist_right = tfs_right[0, :3, :3]
        # 完整变换：AVP wrist → robot base_link 
        # ** 对齐URDF定义的手腕姿态(z朝手指方向)和数据集中的手腕姿态，否则机器人的FK计算的基座(z朝手指方向)和数据集中的基座的姿态不一致，导致重定向结果不正确 **
        R_base_left  = R_wrist_left  @ T_wrist_left_2world[:3, :3] 
        R_base_right = R_wrist_right @ T_wrist_right_2world[:3, :3]

        pos_left_local  = (R_base_left.T  @ pos_left_world.T).T    # (26,3) base_link系
        pos_right_local = (R_base_right.T @ pos_right_world.T).T

        ref_left  = pos_left_local[left_kp_idx[1, :]]  - pos_left_local[left_kp_idx[0, :]]
        ref_right = pos_right_local[right_kp_idx[1, :]] - pos_right_local[right_kp_idx[0, :]]

        q_target_left  = left_retargeting.retarget(ref_left)   # shape = robot.dof
        q_target_right = right_retargeting.retarget(ref_right)


        # 相机图像：torchcodec 返回 (C, H, W) uint8 Tensor，转成 (H, W, C) numpy
        img_np = None
        if cam_img is not None:
            img_np = cam_img.detach().cpu().numpy().transpose(1, 2, 0)  # C×H×W → H×W×C
            rr.log("camera/image", rr.Image(img_np))

        # 世界坐标系旋绕
        world_origin = np.eye(4)
        log_axes("world", world_origin, size=0.1)

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  📷  人手数据  (来自 EgoDex/AVP 数据集，原始世界坐标系下)           ║
        # ╚══════════════════════════════════════════════════════════════════╝
       
        # 左手：关键点（蓝色）+ 骨骼连线 + 手腕坐标轴   ==基坐标系是世界坐标系==
        rr.log("human/left/keypoints",
               rr.Points3D(pos_left_world,  radii=0.005, colors=[0, 180, 255]))
        rr.log("human/left/bones",
               rr.LineStrips3D(bone_strips(pos_left_world), colors=[0, 120, 200]))
        log_axes("human/left/wrist", left_wrist_pose, size=0.05)

        # 右手：关键点（橙色）+ 骨骼连线 + 手腕坐标轴
        rr.log("human/right/keypoints",
               rr.Points3D(pos_right_world, radii=0.005, colors=[255, 140, 0]))
        rr.log("human/right/bones",
               rr.LineStrips3D(bone_strips(pos_right_world), colors=[200, 90, 0]))
        log_axes("human/right/wrist", right_wrist_pose, size=0.05)

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  🤖  机器人渲染  (Retargeting 结果，URDF STL mesh + FK)          ║
        # ╚══════════════════════════════════════════════════════════════════╝

        # 更新各 link 的世界系变换（mesh 已静态写入，只需更新 Transform） ==基坐标系是手腕==
        left_wrist_pose2render = left_wrist_pose @ T_wrist_left_2world
        right_wrist_pose2render = right_wrist_pose @ T_wrist_right_2world
        log_robot_transforms("robot/left",  left_robot,  q_target_left,  left_wrist_pose2render)
        log_robot_transforms("robot/right", right_robot, q_target_right, right_wrist_pose2render)

        # base_link 坐标轴（用于对比人手手腕坐标系是否对齐）
        left_robot.compute_forward_kinematics(q_target_left)
        right_robot.compute_forward_kinematics(q_target_right)
        left_base_world  = left_wrist_pose2render  @ left_robot.get_link_pose(
            left_robot.get_link_index(left_wrist_link_name))
        right_base_world = right_wrist_pose2render @ right_robot.get_link_pose(
            right_robot.get_link_index(right_wrist_link_name))
        log_axes(f"robot/left/{left_wrist_link_name}",  left_base_world,  size=0.05)
        log_axes(f"robot/right/{right_wrist_link_name}", right_base_world, size=0.05)

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  🖐  MediaPipe 实时检测 + Retargeting  (--use-mediapipe 时启用)  ║
        # ╚══════════════════════════════════════════════════════════════════╝
        if use_mediapipe and img_np is not None:
            img_annotated = img_np.copy()
            left_det, right_det = mp_detector.detect(img_np)

            # 在图像上绘制检测到的骨骼关键点
            if left_det is not None:
                DualHandDetector.draw_on_image(img_annotated, left_det[1])
            if right_det is not None:
                DualHandDetector.draw_on_image(img_annotated, right_det[1])
            rr.log("mediapipe/camera/image", rr.Image(img_annotated))

            # MediaPipe 关节位置 → 机器人 Retargeting → 渲染
            # 注意：检测失败时不更新 transform，Rerun 自动保持上一帧姿态，避免骨骼图闪烁消失
            if right_det is not None:
                joint_pos, _ = right_det
                ref = joint_pos[mp_right_kp_idx[1, :]] - joint_pos[mp_right_kp_idx[0, :]]
                q_mp_right = mp_right_retargeting.retarget(ref)
                log_robot_transforms(
                    "mediapipe/robot/right", mp_right_robot, q_mp_right, right_wrist_pose2render)
            if left_det is not None:
                joint_pos, _ = left_det
                ref = joint_pos[mp_left_kp_idx[1, :]] - joint_pos[mp_left_kp_idx[0, :]]
                q_mp_left = mp_left_retargeting.retarget(ref)
                log_robot_transforms(
                    "mediapipe/robot/left", mp_left_robot, q_mp_left, left_wrist_pose2render)


# ──────────────────────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────────────────────
"""
Rerun 可视化布局设计解释：
    /
    ├── world/                     ← 世界坐标轴
    ├── human/
    │   ├── left/keypoints
    │   ├── right/keypoints
    │   └── ...
    ├── robot/                     ← EgoDex retargeting 的机器人手  ← ①
    │   ├── left/left_base_link/mesh_0
    │   ├── right/...
    │   └── ...
    └── mediapipe/
        ├── camera/image           ← 带骨骼标注的图像
        └── robot/                 ← MediaPipe retargeting 的机器人手  ← ②
            ├── left/...
            └── right/...

    egodex_3d = rrb.Spatial3DView(
        name="EgoDex + Robot Retargeting",
        origin="/"         # ← 只是说"以根节点为原点"
                        #   但实际渲染 / 下面的所有东西：① + ②
    )

    加contents 白名单，过滤一些不想显示的 entity，例如：
    egodex_3d = rrb.Spatial3DView(
        name="EgoDex + Robot Retargeting",
        origin="/",
        contents=[
            "+ /human/**",    # ✅ 人手骨骼
            "+ /robot/**",    # ✅ EgoDex 机器人手（只有这个）
            "+ /world/**",    # ✅ 世界坐标轴
                            # ❌ /mediapipe/** 不在列表里，自动排除
        ]
    )        
"""

def make_blueprint(split_view: bool, use_mediapipe: bool) -> rrb.Blueprint:
    """根据 split_view / use_mediapipe 参数动态生成 Rerun 布局。"""
    # 主 3D 场景（EgoDex 数据 + EgoDex Retargeting）
    if split_view:
        egodex_3d = rrb.Horizontal(
            rrb.Spatial3DView(name="📷 Human Hand (EgoDex/AVP)", origin="/human"),
            rrb.Spatial3DView(name="🤖 EgoDex Retargeting Robot",  origin="/robot"),
        )
    else:
        egodex_3d = rrb.Spatial3DView(
            name="EgoDex + Robot Retargeting",
            origin="/",
            contents=[
                "+ /human/**",    # ✅ 人手骨骼
                "+ /robot/**",    # ✅ EgoDex 机器人手（只有这个）
                "+ /world/**",    # ✅ 世界坐标轴
                                # ❌ /mediapipe/** 不在列表里，自动排除
            ]
        )   

    camera_view = rrb.Spatial2DView(name="📷 Camera Image", origin="/camera")

    if use_mediapipe:
        # [EgoDex 3D | 原图 / MediaPipe骨骼图 | MediaPipe 机器人 3D]
        return rrb.Blueprint(
            rrb.Horizontal(
                egodex_3d,
                rrb.Vertical(
                    camera_view,
                    rrb.Spatial2DView(
                        name="📷 Camera + MediaPipe Skeleton",
                        origin="/mediapipe/camera",
                    ),
                ),
                rrb.Spatial3DView(
                    name="🤖 MediaPipe Retargeting Robot",
                    origin="/mediapipe/robot",
                ),
                column_shares=[3, 1, 2],
            )
        )
    else:
        # [EgoDex 3D | 相机原图]
        return rrb.Blueprint(
            rrb.Horizontal(
                egodex_3d,
                camera_view,
                column_shares=[3, 1],
            )
        )


def main(
    robot_name: RobotName,
    data_dir: str,
    num_episodes: int = 1,
    split_view: bool = False,
    use_mediapipe: bool = False,
):
    """
    从 EgoDex 数据集读取双手骨骼，retargeting 后用 Rerun 可视化：
    - 人手骨骼（蓝/橙关键点 + 连线）
    - 机器人 URDF mesh（真实 STL 模型，随关节角实时运动）

    Args:
        robot_name:    目标机械手型号。
        data_dir:      EgoDex 数据目录路径。
        num_episodes:  可视化的 episode 数量。
        split_view:    是否将 EgoDex 人手/机器人分图显示（默认同图）。
        use_mediapipe: 是否开启 MediaPipe 实时检测通道（需安装 mediapipe）。
    """
    robot_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))

    rr.init("dex_egodex_retargeting", spawn=True)
    rr.send_blueprint(make_blueprint(split_view, use_mediapipe))

    dataset = SimpleDataset(data_dir, query_tfs=NEW_QUERY_TFS)
    retarget_egodex(dataset, robot_name, num_episodes, use_mediapipe)


if __name__ == "__main__":
    tyro.cli(main)
