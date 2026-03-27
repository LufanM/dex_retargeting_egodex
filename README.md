<div align="center">
  <h1 align="center"> Dex Retargeting </h1>
  <h3 align="center">
    Various retargeting optimizers to translate human hand motion to robot hand motion.
  </h3>
</div>
<p align="center">
  <!-- code check badges -->
  <a href='https://github.com/dexsuite/dex-retargeting/blob/main/.github/workflows/test.yml'>
      <img src='https://github.com/dexsuite/dex-retargeting/actions/workflows/test.yml/badge.svg' alt='Test Status' />
  </a>
  <!-- issue badge -->
  <a href="https://github.com/dexsuite/dex-retargeting/issues">
  <img src="https://img.shields.io/github/issues-closed/dexsuite/dex-retargeting.svg" alt="Issues Closed">
  </a>
  <a href="https://github.com/dexsuite/dex-retargeting/issues?q=is%3Aissue+is%3Aclosed">
  <img src="https://img.shields.io/github/issues/dexsuite/dex-retargeting.svg" alt="Issues">
  </a>
  <!-- release badge -->
  <a href="https://github.com/dexsuite/dex-retargeting/tags">
  <img src="https://img.shields.io/github/v/release/dexsuite/dex-retargeting.svg?include_prereleases&sort=semver" alt="Releases">
  </a>
  <!-- pypi badge -->
  <a href="https://github.com/dexsuite/dex-retargeting/tags">
  <img src="https://static.pepy.tech/badge/dex_retargeting/month" alt="pypi">
  </a>
  <!-- license badge -->
  <a href="https://github.com/dexsuite/dex-retargeting/blob/main/LICENSE">
      <img alt="License" src="https://img.shields.io/badge/license-MIT-blue">
  </a>
</p>
<div align="center">
  <h4>This repo originates from <a href="https://yzqin.github.io/anyteleop/">AnyTeleop Project</a></h4>
  <img src="example/vector_retargeting/teaser.webp" alt="Retargeting with different hands.">
</div>

## Installation

```shell
pip install dex_retargeting
```

To run the example, you may need additional dependencies for rendering and hand pose detection.

```shell
git clone https://github.com/dexsuite/dex-retargeting
cd dex-retargeting
pip install -e ".[example]" 
# 意思会把pyproject.toml文件里面的[project.optional-dependencies]下名为example的依赖随主依赖一起安装
```
torch 安装
```shell
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
```



## 关于pin本地与虚拟环境冲突问题

* 方法一：conda脚本更好环境路径

  因为本地环境使用了robotpkg安装了pinocchio，并且更改了PYTHONPATH和包路径，为了使conda环境使用pip3 install安装的那个包；进行如下操作(假设conda环境为dex_retargetting)：

  * `/home/molufan/anaconda3/envs/dex_retargeting/etc/conda/deactivate.d/restore_openrobots.sh`添加内容

    ```bash
    #!/bin/bash
    # 退出conda环境时恢复原始路径
    if [ -n "${_CONDA_OLD_PYTHONPATH+x}" ]; then
        export PYTHONPATH="${_CONDA_OLD_PYTHONPATH}"
        unset _CONDA_OLD_PYTHONPATH
    fi
    
    if [ -n "${_CONDA_OLD_LD_LIBRARY_PATH+x}" ]; then
        export LD_LIBRARY_PATH="${_CONDA_OLD_LD_LIBRARY_PATH}"
        unset _CONDA_OLD_LD_LIBRARY_PATH
    fi
    
    ```

  * `/home/molufan/anaconda3/envs/dex_retargeting/etc/conda/activate.d/remove_openrobots.sh`添加内容

    ```bash
    #!/bin/bash
    # 保存原始路径，并移除 /opt/openrobots 路径
    export _CONDA_OLD_PYTHONPATH="${PYTHONPATH}"
    export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v '/opt/openrobots' | tr '\n' ':' | sed 's/:$//')
    
    export _CONDA_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/opt/openrobots' | tr '\n' ':' | sed 's/:$//')
    
    # 将 cmeel.prefix/lib 加到最前面，确保优先加载 pip 安装的库
    CMEEL_LIB="/home/molufan/anaconda3/envs/dex_retargeting/lib/python3.9/site-packages/cmeel.prefix/lib"
    export LD_LIBRARY_PATH="${CMEEL_LIB}:${LD_LIBRARY_PATH}"
    
    ```


  

* 方法二： 使用代码更好环境，只作用于当前进程

  在导入pinocchio库前代码最先执行：

  ```python
  # ====================== 【仅当前进程生效】动态清理路径 ======================
  def clean_env_for_current_process():
      # 1. 处理 sys.path（Python 内部的模块搜索路径）：移除 /opt/openrobots
      original_sys_path = sys.path.copy()  # 可选：备份原始路径，万一需要恢复
      sys.path = [p for p in sys.path if "/opt/openrobots" not in p]
  
      # 2. 处理 os.environ（当前进程的环境变量）：移除 /opt/openrobots，添加 cmeel.prefix/lib
      if "LD_LIBRARY_PATH" in os.environ:
          original_ld = os.environ["LD_LIBRARY_PATH"]
          # 移除 /opt/openrobots
          clean_ld = [p for p in original_ld.split(os.pathsep) if p and "/opt/openrobots" not in p]
          # 添加 cmeel.prefix/lib（优先）
          cmeel_lib = "/home/molufan/anaconda3/envs/dex_retargeting/lib/python3.9/site-packages/cmeel.prefix/lib"
          if cmeel_lib not in clean_ld:
              clean_ld.insert(0, cmeel_lib)
          # 更新当前进程的环境变量
          os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(clean_ld)
  
  # 执行清理（仅对当前 Python 进程生效）
  clean_env_for_current_process()
  ```

  

  

## Changelog

### v0.5.0

- **Numpy Support Update**: Starting from this version, `dex-retargeting` supports `numpy >= 2.0.0`. If you need to use `numpy < 2.0.0`, you can install an earlier version of `dex-retargeting` using:
  ```bash
  pip install "dex-retargeting<0.5.0"
  ```

- **Mediapipe Compatibility**: Although `mediapipe` lists `numpy 1.x` as a dependency, it is compatible with `numpy >= 2.0.0`. You can safely ignore any warnings related to this and continue using `numpy 2.0.0` or higher.

- **Dependency Cleanup**: Removed `trimesh` as a dependency to simplify installation and reduce potential conflicts. The core functionality of `dex-retargeting` no longer requires mesh processing capabilities.

## Examples

### Retargeting from human hand video

This type of retargeting can be used for applications like teleoperation,
e.g. [AnyTeleop](https://yzqin.github.io/anyteleop/).

[Tutorial on retargeting from human hand video](example/vector_retargeting/README.md)

### Retarget from hand object pose dataset

![teaser](example/position_retargeting/hand_object.webp)

This type of retargeting can be used post-process human data for robot imitation,
e.g. [DexMV](https://yzqin.github.io/dexmv/).

[Tutorial on retargeting from hand-object pose dataset](example/position_retargeting/README.md)

## FAQ and Troubleshooting

### Joint Orders for Retargeting

URDF parsers, such as ROS, physical simulators, real robot driver, and this repository, may parse URDF files with
different joint orders. To use `dex-retargeting` results with other libraries, handle joint ordering explicitly **using
joint names**, which are unique within a URDF file.

Example: Using `dex-retargeting` with the SAPIEN simulator

```python
from dex_retargeting.seq_retarget import SeqRetargeting

retargeting: SeqRetargeting
sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
retargeting_joint_names = retargeting.joint_names
retargeting_to_sapien = np.array([retargeting_joint_names.index(name) for name in sapien_joint_names]).astype(int)

# Use the index map to handle joint order differences
sapien_robot.set_qpos(retarget_qpos[retargeting_to_sapien])
```

This example retrieves joint names from the SAPIEN robot and `SeqRetargeting` object, creates a mapping
array (`retargeting_to_sapien`) to map joint indices, and sets the SAPIEN robot's joint positions using the retargeted
joint positions.

## Citation

This repository is derived from the [AnyTeleop Project](https://yzqin.github.io/anyteleop/) and is subject to ongoing
enhancements. If you utilize this work, please cite it as follows:

```shell
@inproceedings{qin2023anyteleop,
  title     = {AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System},
  author    = {Qin, Yuzhe and Yang, Wei and Huang, Binghao and Van Wyk, Karl and Su, Hao and Wang, Xiaolong and Chao, Yu-Wei and Fox, Dieter},
  booktitle = {Robotics: Science and Systems},
  year      = {2023}
}
```

## Acknowledgments

The robot hand models in this repository are sourced directly from [dex-urdf](https://github.com/dexsuite/dex-urdf).
The robot kinematics in this repo are based on [pinocchio](https://github.com/stack-of-tasks/pinocchio).
Examples use [SAPIEN](https://github.com/haosulab/SAPIEN) for rendering and visualization.

The `PositionOptimizer` leverages methodologies from our earlier
project, [From One Hand to Multiple Hands](https://yzqin.github.io/dex-teleop-imitation/).
Additionally, the `DexPilotOptimizer`is crafted using insights from [DexPilot](https://sites.google.com/view/dex-pilot).
