#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量将 WHUCAD 向量 .h5 文件（data/vec/...）转换为 JSON 指令流（data/json/...）

目录结构要求：
- 输入：data/vec/0000/*.h5, data/vec/0001/*.h5, ...
- 输出：data/json/0000/*.json, data/json/0001/*.json, ...
  即：在 data 目录下生成一个 json 文件夹，子目录结构与 vec 完全一致，只改后缀。

用法 1（符合你现在的目录约定，推荐）:
    python batch_whucad_h5_to_json.py

用法 2（自定义路径，可选）:
    python batch_whucad_h5_to_json.py --vec_root path/to/data/vec --json_root path/to/data/json
"""

import os
import argparse
import json

from whucad_h5_to_json import convert_whucad_h5_to_json


def convert_one_file(h5_path: str, vec_root: str, json_root: str):
    """
    将单个 h5 文件转换为 json，并写入到 json_root 中对应的位置。
    - vec_root : input 根目录 (data/vec)
    - json_root: output 根目录 (data/json)
    输出路径 = json_root / 相对路径(去掉 .h5，加上 .json)
    """
    # 计算相对于 vec_root 的相对路径，例如 "0000/00000001.h5"
    rel_path = os.path.relpath(h5_path, vec_root)
    rel_no_ext, _ = os.path.splitext(rel_path)
    out_path = os.path.join(json_root, rel_no_ext + ".json")

    # 创建输出目录
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)

    # 调用已有的转换函数
    result = convert_whucad_h5_to_json(h5_path)

    # 写 json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[OK] {h5_path} -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vec_root",
        type=str,
        default=os.path.join("data", "vec"),
        help="输入 h5 向量根目录（默认: data/vec）",
    )
    parser.add_argument(
        "--json_root",
        type=str,
        default=os.path.join("data", "json"),
        help="输出 json 根目录（默认: data/json）",
    )
    args = parser.parse_args()

    vec_root = os.path.abspath(args.vec_root)
    json_root = os.path.abspath(args.json_root)

    if not os.path.isdir(vec_root):
        raise RuntimeError(f"vec_root 不是目录: {vec_root}")

    print(f"Input  (vec_root) : {vec_root}")
    print(f"Output (json_root): {json_root}")

    n_total = 0
    for dirpath, dirnames, filenames in os.walk(vec_root):
        for fname in filenames:
            if not fname.lower().endswith(".h5"):
                continue
            h5_path = os.path.join(dirpath, fname)
            convert_one_file(h5_path, vec_root, json_root)
            n_total += 1

    print(f"Done. Converted {n_total} .h5 files.")


if __name__ == "__main__":
    main()
