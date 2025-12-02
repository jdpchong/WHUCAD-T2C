import h5py
import numpy as np
import sys

if len(sys.argv) < 1 + 1:
    print("Usage: python showh5.py path/to/file.h5")
    sys.exit(1)

path = sys.argv[1]

with h5py.File(path, "r") as f:
    # 1. 打印所有 dataset 名称
    print("keys in file:", list(f.keys()))

    # 2. 取出 vec
    vec_ds = f["vec"]
    print("vec shape:", vec_ds.shape)
    print("vec dtype:", vec_ds.dtype)

    # 3. 转成 numpy 数组
    vec = vec_ds[()]   # (N, 33)

# 设置显示格式（可选）
np.set_printoptions(precision=4, suppress=True)

# 4. 打印前几行内容，方便肉眼检查
print("\nFirst 5 rows of vec:")
print(vec)

# 5. 统计：每个出现的 token_id 以及参数不为 -1 的列号
type_ids = vec[:, 0].astype(int)   # 第 0 列是 token_id / type_id
unique_ids = np.unique(type_ids)

print("\nSummary by token_id (type_id):")
for tid in unique_ids:
    # 选出所有 type_id == tid 的行
    rows = vec[type_ids == tid]

    # 如果全是 padding（例如 tid==0 且整行=-1），可以自己决定要不要跳过
    # 这里只要行存在就统计
    params = rows[:, 1:]  # 去掉第 0 列，只看参数 (N_rows, 32)

    # 对每一列检查是否存在 != -1 的值
    # valid_mask: shape (32,), True 表示这一列至少有一个 != -1
    valid_mask = (params != -1).any(axis=0)

    # 参数列下标（0~31），对应原始矩阵的列 1~32
    param_indices = np.where(valid_mask)[0]          # 0-based for params
    global_indices = param_indices + 1               # 0-based in整行: 1~32

    print(f"\n  token_id = {tid}")
    print("    param columns (0-based in param block 0..31) with value != -1:")
    print("    ", param_indices.tolist())
    print("    param columns (global col index in vec 0..32) with value != -1:")
    print("    ", global_indices.tolist())
