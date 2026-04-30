# import sys
# import os

# # ====================== 【仅当前进程生效】动态清理路径 ======================
# def clean_env_for_current_process():
#     # 1. 处理 sys.path（Python 内部的模块搜索路径）：移除 /opt/openrobots
#     # original_sys_path = sys.path.copy()  # 可选：备份原始路径，万一需要恢复
#     sys.path = [p for p in sys.path if "/opt/openrobots" not in p]

#     # 2. 处理 os.environ（当前进程的环境变量）：移除 /opt/openrobots ，添加 cmeel.prefix/lib
#     # 添加 cmeel.prefix/lib/python3.10/site-packages 到 sys.path（让 Python 能找到 pinocchio 包）
#     cmeel_py_path = "/home/molufan/anaconda3/envs/dex_retargeting/lib/python3.10/site-packages/cmeel.prefix/lib/python3.10/site-packages"
#     if cmeel_py_path not in sys.path:
#         sys.path.insert(1, cmeel_py_path)

#     if "LD_LIBRARY_PATH" in os.environ:
#         original_ld = os.environ["LD_LIBRARY_PATH"]
#         # 移除 /opt/openrobots
#         clean_ld = [p for p in original_ld.split(os.pathsep) if p and "/opt/openrobots" not in p]
#         # 添加 cmeel.prefix/lib（让 .so 共享库能被加载）
#         cmeel_lib = "/home/molufan/anaconda3/envs/dex_retargeting/lib/python3.10/site-packages/cmeel.prefix/lib"
#         if cmeel_lib not in clean_ld:
#             clean_ld.insert(0, cmeel_lib)
#         # 更新当前进程的环境变量
#         os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(clean_ld)

# # 执行清理（仅对当前 Python 进程生效）
# # clean_env_for_current_process()

# print("Python 解释器路径：", sys.executable)
# print("\n所有库查找路径（按优先级排序）：")
# for p in sys.path:
#     print(p)

# import pinocchio as pin


# print("Pinocchio 版本：", pin.__version__)