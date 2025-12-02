#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert WHUCAD .h5 (dataset 'vec', shape (Nc, 33)) into a step-by-step JSON
instruction sequence, using the same command/argument layout as configAE.py.

Usage:
    python whucad_h5_to_json.py input.h5 output.json
"""

import argparse
import json
import os
from typing import Dict, Any

import h5py
import numpy as np

# ---------------- WHUCAD command & arg layout (from your config) ---------------- #

ALL_COMMANDS = [
    'Line', 'Arc', 'Circle', 'Spline', 'SCP',
    'EOS', 'SOL',
    'Ext', 'Rev', 'Pocket', 'Groove',
    'Shell', 'Chamfer', 'Fillet', 'Draft', 'Mirror', 'Hole',
    'Topo', 'Select', 'MirrorStart',
    'NoSharedIncluded', 'NoSharedIncludedEnd',
    'AllOrientedIncluded1', 'AllOrientedIncluded2', 'AllOrientedIncludedEnd',
    'AllPartiallySharedIncluded', 'AllPartiallySharedIncludedEnd'
]

# 32 参数位的含义顺序（对应 N_ARGS_SKETCH + N_ARGS_EXT + N_ARGS_FINISH_PARAM + N_ARGS_SELECT_PARAM）
# N_ARGS_SKETCH = 5  -> 0..4
# N_ARGS_PLANE  = 3  -> 5..7
# N_ARGS_TRANS  = 4  -> 8..11
# N_ARGS_BODY   = 7  -> 12..18
# N_ARGS_FINISH = 9  -> 19..27
# N_ARGS_SELECT = 4  -> 28..31
PARAM_ORDER = [
    # sketch
    "x", "y", "alpha", "f", "r",
    # plane orientation
    "theta", "phi", "gamma",
    # plane origin & sketch bbox
    "px", "py", "pz", "s",
    # body params (extrude/revolve/pocket/groove)
    "length1", "length2", "length1_type", "length2_type",
    "angle1", "angle2", "boolean",
    # finish params (shell/chamfer/fillet/draft/hole)
    "thickness1", "thickness2",
    "finish_length1", "finish_length2",
    "finish_radius", "finish_alpha",
    "hole_r", "hole_depth", "hole_type",
    # selection params
    "select_type", "body_type", "body_no", "no",
]

# 检查一下我们和 WHUCAD 的 N_ARGS 一致（安全起见）
assert len(PARAM_ORDER) == 32

N_ARGS_SKETCH = 5
N_ARGS_PLANE = 3
N_ARGS_TRANS = 4
N_ARGS_BODY_PARAM = 7
N_ARGS_FINISH_PARAM = 9
N_ARGS_SELECT_PARAM = 4
N_ARGS_EXT = N_ARGS_PLANE + N_ARGS_TRANS + N_ARGS_BODY_PARAM
N_ARGS = N_ARGS_SKETCH + N_ARGS_EXT + N_ARGS_FINISH_PARAM + N_ARGS_SELECT_PARAM
assert N_ARGS == 32

PAD_VAL = -1

# 一些常量下标（用不上也没关系，主要是保持一致）
LINE_IDX = ALL_COMMANDS.index('Line')        # 0
ARC_IDX = ALL_COMMANDS.index('Arc')          # 1
CIRCLE_IDX = ALL_COMMANDS.index('Circle')    # 2
SPLINE_IDX = ALL_COMMANDS.index('Spline')    # 3
SCP_IDX = ALL_COMMANDS.index('SCP')          # 4
EOS_IDX = ALL_COMMANDS.index('EOS')          # 5
SOL_IDX = ALL_COMMANDS.index('SOL')          # 6
EXT_IDX = ALL_COMMANDS.index('Ext')          # 7
REV_IDX = ALL_COMMANDS.index('Rev')          # 8
POCKET_IDX = ALL_COMMANDS.index('Pocket')    # 9
GROOVE_IDX = ALL_COMMANDS.index('Groove')    # 10
SHELL_IDX = ALL_COMMANDS.index('Shell')      # 11
CHAMFER_IDX = ALL_COMMANDS.index('Chamfer')  # 12
FILLET_IDX = ALL_COMMANDS.index('Fillet')    # 13
DRAFT_IDX = ALL_COMMANDS.index('Draft')      # 14
MIRROR_IDX = ALL_COMMANDS.index('Mirror')    # 15
HOLE_IDX = ALL_COMMANDS.index('Hole')        # 16
TOPO_IDX = ALL_COMMANDS.index('Topo')        # 17
SELECT_IDX = ALL_COMMANDS.index('Select')    # 18
MIRROR_START_IDX = ALL_COMMANDS.index('MirrorStart')  # 19
NO_SHARED_INCLUDED_IDX = ALL_COMMANDS.index('NoSharedIncluded')  # 20
NO_SHARED_INCLUDED_END_IDX = ALL_COMMANDS.index('NoSharedIncludedEnd')  # 21
ALL_ORIENTED_INCLUDED_1_IDX = ALL_COMMANDS.index('AllOrientedIncluded1')  # 22
ALL_ORIENTED_INCLUDED_2_IDX = ALL_COMMANDS.index('AllOrientedIncluded2')  # 23
ALL_ORIENTED_INCLUDED_END_IDX = ALL_COMMANDS.index('AllOrientedIncludedEnd')  # 24
ALL_PARTIALLY_INCLUDED_IDX = ALL_COMMANDS.index('AllPartiallySharedIncluded')  # 25
ALL_PARTIALLY_INCLUDED_END_IDX = ALL_COMMANDS.index('AllPartiallySharedIncludedEnd')  # 26

# ---- 来自你代码的 CMD_ARGS_MASK（直接照搬） ---- #
CMD_ARGS_MASK = np.array([
    # Line
    [1, 1, 0, 0, 0, *[0]*N_ARGS_EXT, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],
    # Arc
    [1, 1, 1, 1, 0, *[0]*N_ARGS_EXT, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],
    # Circle
    [1, 1, 0, 0, 1, *[0]*N_ARGS_EXT, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],
    # Spline
    [0, 0, 0, 0, 0, *[0]*N_ARGS_EXT, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],
    # SCP
    [1, 1, 0, 0, 0, *[0]*N_ARGS_EXT, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],
    # EOS
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],
    # SOL
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT, *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],
    # Ext
    [*[0]*N_ARGS_SKETCH,
     *[1]*(N_ARGS_PLANE + N_ARGS_TRANS), 1, 1, 1, 1, 0, 0, 1,
     *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],
    # Rev
    [*[0]*N_ARGS_SKETCH,
     *[1]*(N_ARGS_PLANE + N_ARGS_TRANS), 0, 0, 0, 0, 1, 1, 1,
     *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],
    # Pocket
    [*[0]*N_ARGS_SKETCH,
     *[1]*(N_ARGS_PLANE + N_ARGS_TRANS), 1, 1, 1, 1, 0, 0, 0,
     *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],
    # Groove
    [*[0]*N_ARGS_SKETCH,
     *[1]*(N_ARGS_PLANE + N_ARGS_TRANS), 0, 0, 0, 0, 1, 1, 0,
     *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],
    # Shell
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT,
     1, 1, 0, 0, 0, 0, 0, 0, 0, *[0]*N_ARGS_SELECT_PARAM],
    # Chamfer
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT,
     0, 0, 1, 1, 0, 0, 0, 0, 0, *[0]*N_ARGS_SELECT_PARAM],
    # Fillet
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT,
     0, 0, 0, 0, 1, 0, 0, 0, 0, *[0]*N_ARGS_SELECT_PARAM],
    # Draft
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT,
     0, 0, 0, 0, 0, 1, 0, 0, 0, *[0]*N_ARGS_SELECT_PARAM],
    # Mirror
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT,
     *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],
    # Hole
    [1, 1, 0, 0, 0, *[0]*N_ARGS_EXT,
     0, 0, 0, 0, 0, 0, 1, 1, 1,
     *[0]*N_ARGS_SELECT_PARAM],
    # Topo
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT,
     *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],
    # Select
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT,
     *[0]*N_ARGS_FINISH_PARAM, *[1]*N_ARGS_SELECT_PARAM],
    # MirrorStart
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT,
     *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],
    # NoSharedIncluded
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT,
     *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],
    # NoSharedIncludedEnd
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT,
     *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],
    # AllOrientedIncluded1
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT,
     *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],
    # AllOrientedIncluded2
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT,
     *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],
    # AllOrientedIncludedEnd
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT,
     *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],
    # AllPartiallySharedIncluded
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT,
     *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],
    # AllPartiallySharedIncludedEnd
    [*[0]*N_ARGS_SKETCH, *[0]*N_ARGS_EXT,
     *[0]*N_ARGS_FINISH_PARAM, *[0]*N_ARGS_SELECT_PARAM],
], dtype=np.int32)

assert CMD_ARGS_MASK.shape == (len(ALL_COMMANDS), N_ARGS)

# id -> name 映射
FEATURE_ID_TO_NAME: Dict[int, str] = {i: name for i, name in enumerate(ALL_COMMANDS)}

# ------------------------ 辅助函数：一行向量 → dict ------------------------ #

def build_param_dict(param_row: np.ndarray) -> Dict[str, float]:
    """把一行 32 维向量转成 {param_name: value} dict。"""
    assert param_row.shape[0] == len(PARAM_ORDER)
    return {name: float(param_row[i]) for i, name in enumerate(PARAM_ORDER)}


def filter_used_params(type_id: int, raw_params: Dict[str, float]) -> Dict[str, float]:
    """根据 type_id 和 CMD_ARGS_MASK，筛选真正有意义的参数。"""
    if type_id < 0 or type_id >= CMD_ARGS_MASK.shape[0]:
        # 未知类型：保留非 PAD 的值方便调试
        return {k: v for k, v in raw_params.items() if v != PAD_VAL}

    mask = CMD_ARGS_MASK[type_id]
    used = {}
    for i, flag in enumerate(mask):
        if flag == 1:
            name = PARAM_ORDER[i]
            v = raw_params[name]
            if v != PAD_VAL:
                used[name] = v
    return used


# ------------------------ 主转换逻辑：.h5 -> JSON ------------------------ #

def convert_whucad_h5_to_json(h5_path: str) -> Dict[str, Any]:
    with h5py.File(h5_path, "r") as f:
        vec = np.asarray(f["vec"][:], dtype=np.float32)  # (Nc, 33)

    if vec.ndim != 2 or vec.shape[1] != 33:
        raise RuntimeError(f"Expected dataset 'vec' with shape (Nc,33), got {vec.shape}")

    Nc = vec.shape[0]
    type_ids = vec[:, 0].astype(np.int32)
    params = vec[:, 1:]  # (Nc, 32)

    sequence = []

    for i in range(Nc):
        tid = int(type_ids[i])
        prow = params[i]

        # 有些序列会在后面 padding EOS / SOL / 全 PAD_VAL；这里只跳过“完全空”的行
        if tid < 0 and np.allclose(prow, PAD_VAL):
            continue

        type_name = FEATURE_ID_TO_NAME.get(tid, f"Unknown_{tid}")
        raw_param_dict = build_param_dict(prow)
        used_param_dict = filter_used_params(tid, raw_param_dict)

        step = {
            "step_idx": i,
            "type_id": tid,
            "type_name": type_name,
            "params": used_param_dict,  # 只保留用到的
        }
        sequence.append(step)

    return {
        "meta": {
            "source": "WHUCAD",
            "file_name": os.path.basename(h5_path),
            "num_steps": len(sequence),
        },
        "sequence": sequence,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_h5", type=str)
    ap.add_argument("output_json", type=str)
    args = ap.parse_args()

    result = convert_whucad_h5_to_json(args.input_h5)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Saved:", args.output_json)


if __name__ == "__main__":
    main()
