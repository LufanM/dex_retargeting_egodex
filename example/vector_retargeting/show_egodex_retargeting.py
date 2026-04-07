import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import tyro
import yaml
from scipy.spatial.transform import Rotation

# ── sys.path 配置 ──────────────────────────────────────────────────────────────
_ROOT = str(Path(__file__).absolute().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_EGODEX = str(Path(__file__).absolute().parent.parent.parent / "egodex")
if _EGODEX not in sys.path:
    sys.path.insert(1, _EGODEX)

from dex_retargeting.constants import RobotName
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
# 核心流程
# ──────────────────────────────────────────────────────────────────────────────

def retarget_egodex(dataset: SimpleDataset, robot_name: RobotName, num_episodes: int):
    tf2idx = {k: i for i, k in enumerate(NEW_QUERY_TFS)}
    num_transitions = dataset.cumulative_len[num_episodes - 1]

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

    # ── 解析 URDF mesh，一次性写入 Rerun（static=True） ────────────────────────
    left_link_meshes  = parse_urdf_link_meshes(left_config.urdf_path)
    right_link_meshes = parse_urdf_link_meshes(right_config.urdf_path)

    # ── 机器人 mesh 静态写入（几何体不变，只有 Transform 随时间变化）──────────────
    log_robot_meshes_once("robot/left",  left_link_meshes)
    log_robot_meshes_once("robot/right", right_link_meshes)

    # ── 逐帧循环 ──────────────────────────────────────────────────────────────
    # wrist_init_poses = np.tile(np.eye(4, dtype=np.float64), (2, 1, 1))

    for i in range(num_transitions):
        rr.set_time_sequence("frame", i)

        tfs_avp, cam_ext, cam_int, cam_img, _, _ = dataset[i]
        episodes_idx, start_ts = dataset._locate_transition(i)
        tfs = T_robot_avp @ tfs_avp

        if i == start_ts:
            print(f"Episode {episodes_idx} starts at transition {i}.start_ts is {start_ts}")
            wrist_init_poses = tfs[[tf2idx['leftHand'], tf2idx['rightHand']], :, :]

        left_wrist_pose  = tfs[tf2idx['leftHand']]
        right_wrist_pose = tfs[tf2idx['rightHand']]

        # ── Retargeting ───────────────────────────────────────────────────────
        tfs_left, tfs_right = np.split(tfs, 2, axis=0)
        pos_left  = tfs_left[:, :3, 3]
        pos_right = tfs_right[:, :3, 3]

        ref_left  = pos_left[left_kp_idx[1, :]]  - pos_left[left_kp_idx[0, :]]
        ref_right = pos_right[right_kp_idx[1, :]] - pos_right[right_kp_idx[0, :]]

        q_target_left  = left_retargeting.retarget(ref_left)   # shape = robot.dof
        q_target_right = right_retargeting.retarget(ref_right)


        # 相机图像：torchcodec 返回 (C, H, W) uint8 Tensor，转成 (H, W, C) numpy
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
               rr.Points3D(pos_left,  radii=0.005, colors=[0, 180, 255]))
        rr.log("human/left/bones",
               rr.LineStrips3D(bone_strips(pos_left), colors=[0, 120, 200]))
        log_axes("human/left/wrist", left_wrist_pose, size=0.05)

        # 右手：关键点（橙色）+ 骨骼连线 + 手腕坐标轴
        rr.log("human/right/keypoints",
               rr.Points3D(pos_right, radii=0.005, colors=[255, 140, 0]))
        rr.log("human/right/bones",
               rr.LineStrips3D(bone_strips(pos_right), colors=[200, 90, 0]))
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
            left_robot.get_link_index('left_base_link'))
        right_base_world = right_wrist_pose2render @ right_robot.get_link_pose(
            right_robot.get_link_index('right_base_link'))
        log_axes("robot/left/base_link",  left_base_world,  size=0.05)
        log_axes("robot/right/base_link", right_base_world, size=0.05)


# ──────────────────────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────────────────────


def main(
    robot_name: RobotName,
    data_dir: str,
    num_episodes: int = 1,
    split_view: bool = False,
):
    """
    从 EgoDex 数据集读取双手骨骼，retargeting 后用 Rerun 可视化：
    - 人手骨骼（蓝/橙关键点 + 连线）
    - 机器人 URDF mesh（真实 STL 模型，随关节角实时运动）
    - split_view: True=人手/机器人分两个 3D 视图，False=同一视图（默认）

    Args:
        robot_name:    目标机械手型号。
        data_dir:      EgoDex 数据目录路径。
        num_episodes:  可视化的 episode 数量。
        split_view:    是否分图渲染
    """
    robot_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))

    # Rerun blueprint: 单图/分图
    rr.init("dex_egodex_retargeting", spawn=True)
    if split_view:
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial3DView(
                        name="📷 Human Hand (EgoDex/AVP)",
                        origin="/human",
                    ),
                    rrb.Spatial3DView(
                        name="🤖 Robot Rendering (Retargeting)",
                        origin="/robot",
                    ),
                    rrb.Spatial2DView(
                        name="📷 Camera Image",
                        origin="/camera",
                    ),
                    column_shares=[2, 2, 1],
                )
            )
        )
    else:
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial3DView(
                        name="EgoDex+ Robot Retargeting",
                        origin="/",
                    ),
                    rrb.Spatial2DView(
                        name="📷 Camera Image",
                        origin="/camera",
                    ),
                    column_shares=[3, 1],
                )
            )
        )

    dataset = SimpleDataset(data_dir, query_tfs=NEW_QUERY_TFS)
    retarget_egodex(dataset, robot_name, num_episodes)


if __name__ == "__main__":
    tyro.cli(main)
