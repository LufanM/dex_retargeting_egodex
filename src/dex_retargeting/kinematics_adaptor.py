from abc import abstractmethod
from typing import List

import numpy as np

from dex_retargeting.robot_wrapper import RobotWrapper


class KinematicAdaptor:
    def __init__(self, robot: RobotWrapper, target_joint_names: List[str]):
        self.robot = robot
        self.target_joint_names = target_joint_names

        # Index mapping
        self.idx_pin2target = np.array(
            [robot.get_joint_index(n) for n in target_joint_names]
        )

    @abstractmethod
    def forward_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """
        Adapt the joint position for different kinematics constraints.
        Note that the joint order of this qpos is consistent with pinocchio
        Args:
            qpos: the pinocchio qpos

        Returns: the adapted qpos with the same shape as input

        """
        pass

    @abstractmethod
    def backward_jacobian(self, jacobian: np.ndarray) -> np.ndarray:
        """
        Adapt the jacobian for different kinematics applications.
        Note that the joint order of this Jacobian is consistent with pinocchio
        Args:
            jacobian: the original jacobian

        Returns: the adapted jacobian with the same shape as input

        """
        pass


class MimicJointKinematicAdaptor(KinematicAdaptor):
    """
    背景：
        有些关节的运动并不是独立的，而是由其他关节的运动通过固定的几何关系决定（例如平行四连杆、齿轮传动等）。这种关节被称为 mimic joint（模仿关节/从动关节）。
        它的位置（角度/位移）通常与某个源关节的位置成线性关系：
        q_mimic = multiplier * q_source + offset
        其中 multiplier 是一个乘数，offset 是一个偏移量。这些参数由
    在代码中的作用：
        - self.multipliers 就是上述公式中的 multiplier 列表，每个元素对应一个 mimic 关节的乘数。
        - 它与 self.offsets 一起，定义了从源关节到 mimic 关节的映射关系。
        - 在 forward_qpos 方法中，通过 pin_qpos[self.idx_pin2source] * self.multipliers + self.offsets 计算出所有 mimic 关节的目标位置，然后更新到完整的关节状态 pin_qpos 中。
        - 在 backward_jacobian 方法中，将 mimic 关节对雅可比矩阵的贡献乘以 multipliers 后累加到目标关节的雅可比上，以保证运动学计算的正确性。
    """
    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        source_joint_names: List[str],
        mimic_joint_names: List[str],
        multipliers: List[float],
        offsets: List[float],
    ):
        super().__init__(robot, target_joint_names)

        self.multipliers = np.array(multipliers)
        self.offsets = np.array(offsets)

        # Joint name check
        union_set = set(mimic_joint_names).intersection(set(target_joint_names))
        if len(union_set) > 0:
            raise ValueError(
                f"Mimic joint should not be one of the target joints.\n"
                f"Mimic joints: {mimic_joint_names}.\n"
                f"Target joints: {target_joint_names}\n"
                f"You need to specify the target joint names explicitly in your retargeting config"
                f" for robot with mimic joint constraints: {target_joint_names}"
            )

        # Indices in the pinocchio
        self.idx_pin2source = np.array(
            [robot.get_joint_index(name) for name in source_joint_names]
        )
        self.idx_pin2mimic = np.array(
            [robot.get_joint_index(name) for name in mimic_joint_names]
        )

        # Indices in the output results
        self.idx_target2source = np.array(
            [self.target_joint_names.index(n) for n in source_joint_names]
        )

        # Dimension check
        len_source, len_mimic = (
            self.idx_target2source.shape[0],
            self.idx_pin2mimic.shape[0],
        )
        len_mul, len_offset = self.multipliers.shape[0], self.offsets.shape[0]
        if not (len_mimic == len_source == len_mul == len_offset):
            raise ValueError(
                f"Mimic joints setting dimension mismatch.\n"
                f"Source joints: {len_source}, mimic joints: {len_mimic}, multiplier: {len_mul}, offset: {len_offset}"
            )
        self.num_active_joints = len(robot.dof_joint_names) - len_mimic

        # Uniqueness check
        if len(mimic_joint_names) != len(np.unique(mimic_joint_names)):
            raise ValueError(f"Redundant mimic joint names: {mimic_joint_names}")

    def forward_qpos(self, pin_qpos: np.ndarray) -> np.ndarray:
        mimic_qpos = pin_qpos[self.idx_pin2source] * self.multipliers + self.offsets
        pin_qpos[self.idx_pin2mimic] = mimic_qpos
        return pin_qpos

    def backward_jacobian(self, jacobian: np.ndarray) -> np.ndarray:
        """
        self.idx_pin2target 目标关节索引（需要优化的关节）在全关节中的位置
        self.idx_pin2mimic 镜像关节索引在全关节中的位置
        self.multipliers 镜像关节与源关节的乘数关系, 镜像关节角度 = 乘数 × 源关节角度
        self.idx_target2source 映射，表示每个镜像关节对应的源目标关节在目标关节数组中的索引。

        v = J_t * q_dot_target + J_m * q_dot_mimic
        由于 q_dot_mimic = multipliers * q_dot_source, 可以将其代入上式得到：
        v = J_t * q_dot_target + J_m * multipliers * q_dot_source
        因此，v = (J_t + J_m * multipliers) * q_dot_target，其中 J_t 是目标关节的雅可比矩阵，J_m 是镜像关节的雅可比矩阵。
        """
        target_jacobian = jacobian[..., self.idx_pin2target]
        mimic_joint_jacobian = jacobian[..., self.idx_pin2mimic] * self.multipliers

        for i, index in enumerate(self.idx_target2source):
            target_jacobian[..., index] += mimic_joint_jacobian[..., i]
        return target_jacobian
