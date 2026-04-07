from typing import List

import numpy as np
import numpy.typing as npt
import sys
import os

# ====================== 【仅当前进程生效】动态清理路径 ======================
def clean_env_for_current_process():
    # 1. 处理 sys.path（Python 内部的模块搜索路径）：移除 /opt/openrobots
    original_sys_path = sys.path.copy()  # 可选：备份原始路径，万一需要恢复
    sys.path = [p for p in sys.path if "/opt/openrobots" not in p]

    # 2. 处理 os.environ（当前进程的环境变量）：移除 /opt/openrobots ，添加 cmeel.prefix/lib
    # 添加 cmeel.prefix/lib/python3.10/site-packages 到 sys.path（让 Python 能找到 pinocchio 包）
    cmeel_py_path = "/home/molufan/anaconda3/envs/dex_retargeting/lib/python3.10/site-packages/cmeel.prefix/lib/python3.10/site-packages"
    if cmeel_py_path not in sys.path:
        sys.path.insert(1, cmeel_py_path)

    if "LD_LIBRARY_PATH" in os.environ:
        original_ld = os.environ["LD_LIBRARY_PATH"]
        # 移除 /opt/openrobots
        clean_ld = [p for p in original_ld.split(os.pathsep) if p and "/opt/openrobots" not in p]
        # 添加 cmeel.prefix/lib（让 .so 共享库能被加载）
        cmeel_lib = "/home/molufan/anaconda3/envs/dex_retargeting/lib/python3.10/site-packages/cmeel.prefix/lib"
        if cmeel_lib not in clean_ld:
            clean_ld.insert(0, cmeel_lib)
        # 更新当前进程的环境变量
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(clean_ld)

# 执行清理（仅对当前 Python 进程生效）
# clean_env_for_current_process()

import pinocchio as pin


class RobotWrapper:
    """
    This class does not take mimic joint into consideration
    """

    def __init__(self, urdf_path: str, use_collision=False, use_visual=False):
        # Create robot model and data
        self.model: pin.Model = pin.buildModelFromUrdf(urdf_path)
        self.data: pin.Data = self.model.createData()

        if use_visual or use_collision:
            raise NotImplementedError

        self.q0 = pin.neutral(self.model)
        if self.model.nv != self.model.nq:
            raise NotImplementedError("Can not handle robot with special joint.")

    # -------------------------------------------------------------------------- #
    # Robot property
    # -------------------------------------------------------------------------- #
    @property
    def joint_names(self) -> List[str]:
        return list(self.model.names)

    @property
    def dof_joint_names(self) -> List[str]:
        nqs = self.model.nqs
        return [name for i, name in enumerate(self.model.names) if nqs[i] > 0]

    @property
    def dof(self) -> int:
        return self.model.nq

    @property
    def link_names(self) -> List[str]:
        link_names = []
        for i, frame in enumerate(self.model.frames):
            link_names.append(frame.name)
        return link_names

    @property
    def joint_limits(self):
        lower = self.model.lowerPositionLimit
        upper = self.model.upperPositionLimit
        return np.stack([lower, upper], axis=1)

    # -------------------------------------------------------------------------- #
    # Query function
    # -------------------------------------------------------------------------- #
    def get_joint_index(self, name: str):
        return self.dof_joint_names.index(name)

    def get_link_index(self, name: str):
        if name not in self.link_names:
            raise ValueError(
                f"{name} is not a link name. Valid link names: \n{self.link_names}"
            )
        return self.model.getFrameId(name, pin.BODY)

    def get_joint_parent_child_frames(self, joint_name: str):
        joint_id = self.model.getFrameId(joint_name)
        parent_id = self.model.frames[joint_id].parent
        child_id = -1
        for idx, frame in enumerate(self.model.frames):
            if frame.previousFrame == joint_id:
                child_id = idx
        if child_id == -1:
            raise ValueError(f"Can not find child link of {joint_name}")

        return parent_id, child_id

    # -------------------------------------------------------------------------- #
    # Kinematics function
    # -------------------------------------------------------------------------- #
    def compute_forward_kinematics(self, qpos: npt.NDArray):
        pin.forwardKinematics(self.model, self.data, qpos)

    def get_link_pose(self, link_id: int) -> npt.NDArray:
        pose: pin.SE3 = pin.updateFramePlacement(self.model, self.data, link_id)
        return pose.homogeneous

    def get_link_pose_inv(self, link_id: int) -> npt.NDArray:
        pose: pin.SE3 = pin.updateFramePlacement(self.model, self.data, link_id)
        return pose.inverse().homogeneous

    def compute_single_link_local_jacobian(self, qpos, link_id: int) -> npt.NDArray:
        J = pin.computeFrameJacobian(self.model, self.data, qpos, link_id)
        return J
