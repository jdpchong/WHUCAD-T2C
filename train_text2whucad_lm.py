#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
推理脚本：给定一段英文描述，生成 WHUCAD 指令向量序列，用于验证训练效果。

使用方式（在项目根目录）：
    python infer_text2whucad.py --text "A rectangular plate with four corner holes" \
        --checkpoint checkpoints/text2whucad_lm_latest.pt --device cpu

如果不加 --text，会提示你在终端输入一段文本。
"""

import os
import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from whucad_tokenizer import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    VOCAB_SIZE,
    decode_sequence,
)
from model_text2whucad_lm import Text2WhuCADLM


def generate_cad_tokens(
    model,
    text_tokenizer,
    text: str,
    device: str = "cpu",
    max_len: int = 128,
):
    """
    给一段文本，使用训练好的模型自回归生成 CAD token 序列（不含 BOS）。
    使用贪心解码（greedy），直到生成 EOS 或达到 max_len。
    """
    model.eval()

    # 1) 文本编码
    enc = text_tokenizer(
        [text],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = enc["input_ids"].to(device)
    text_attention_mask = enc["attention_mask"].to(device)

    # 2) 初始化 CAD 序列：只含 BOS
    cad_ids = torch.tensor([[BOS_TOKEN_ID]], dtype=torch.long, device=device)  # [1, 1]

    # 我们训练时 max_cad_len = 128，pos_emb 也是这个长度，
    # 所以生成时总长度（含 BOS）不能超过 128
    max_total_len = max_len

    with torch.no_grad():
        for _ in range(max_total_len - 1):  # 已有 BOS，占 1 个位置
            out = model(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                cad_input_ids=cad_ids,
                cad_labels=None,
            )
            logits = out["logits"]  # [1, L, VOCAB_SIZE]
            next_logits = logits[:, -1, :]  # 取最后一个位置的预测 [1, V]

            # 贪心解码：取概率最大的 token
            next_id = next_logits.argmax(dim=-1, keepdim=True)  # [1, 1]

            cad_ids = torch.cat([cad_ids, next_id], dim=1)  # 拼接到序列末尾

            if next_id.item() == EOS_TOKEN_ID:
                break

    # 去掉 BOS，只保留真正 CAD token
    pred_tokens = cad_ids[0, 1:].cpu().numpy().tolist()
    return pred_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Input English description. If omitted, you will be prompted to type one.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/text2whucad_lm_latest.pt",
        help="Path to the trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to run inference on. Recommend 'cpu' first to avoid GPU issues.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=128,
        help="Maximum generated CAD sequence length (including BOS). "
             "Should not exceed the max_len used in training.",
    )
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU.")
        device = "cpu"

    # 1) 准备输入文本
    if args.text is None:
        print("Please input an English description (press Enter to finish):")
        text = input("> ").strip()
    else:
        text = args.text.strip()

    if not text:
        print("Empty text, abort.")
        return

    print(f"\n[Input text]\n{text}\n")

    # 2) 构建 BERT tokenizer 和 encoder（与训练一致）
    print("Loading BERT tokenizer and encoder...")
    text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text_encoder = AutoModel.from_pretrained("bert-base-uncased")
    text_encoder.to(device)

    # 3) 构建 Text2WhuCADLM 模型（配置需与训练时一致）
    print("Building Text2WhuCADLM model...")
    model = Text2WhuCADLM(
        text_encoder=text_encoder,
        d_model=256,   # 确保与 train_text2whucad_lm.py 中的设置一致
        n_layers=2,
        n_heads=4,
        dim_ff=1024,
        max_len=args.max_len,
    )
    model.to(device)

    # 4) 加载训练好的权重
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint from {ckpt_path} ...")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    print("Checkpoint loaded.\n")

    # 5) 生成 CAD token 序列
    print("Generating CAD tokens...")
    pred_tokens = generate_cad_tokens(
        model=model,
        text_tokenizer=text_tokenizer,
        text=text,
        device=device,
        max_len=args.max_len,
    )

    print(f"\n[Generated token ids] (without BOS, EOS may appear in the sequence)")
    print(pred_tokens)

    # 6) 还原为向量序列 [T, 1+N_ARGS]
    print("\nDecoding tokens to WHUCAD vector sequence...")
    vec_seq = decode_sequence(pred_tokens)  # numpy array: [T, 1+N_ARGS]

    print(f"Vector sequence shape: {vec_seq.shape}")  # (T, 1+N_ARGS)
    if vec_seq.shape[0] > 0:
        print("\nFirst few rows of [cmd_idx, arg0, arg1, ...]:")
        n_show = min(5, vec_seq.shape[0])
        for i in range(n_show):
            print(f"Row {i}: {vec_seq[i].tolist()}")
    else:
        print("Decoded sequence is empty.")

    # 你也可以把结果保存下来，方便丢回内核建模：
    # out_path = "debug_generated_vec.npy"
    # np.save(out_path, vec_seq)
    # print(f"\nSaved generated vector sequence to {out_path}")


if __name__ == "__main__":
    main()
