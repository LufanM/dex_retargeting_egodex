import sys
print("Python 解释器路径：", sys.executable)
print("\n所有库查找路径（按优先级排序）：")
for p in sys.path:
    print(p)