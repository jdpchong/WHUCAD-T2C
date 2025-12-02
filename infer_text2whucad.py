# infer_text2whucad_lm.py
"""
简单推理脚本：
- 加载训练好的 Text2WhuCADLM 模型和最新权重
- 输入一段英文描述，生成 CAD token 序列
- 用 decode_sequence 还原为 WHUCAD 向量 [T, 1+N_ARGS]
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
    decode_sequence,
)
from model_text2whucad_lm import Text2WhuCADLM

# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "."))
CKPT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "text2whucad_lm_latest.pt")


def generate_cad_tokens(model, text_tokenizer, text: str,
                        device: str = "cuda", max_len: int = 256):
    """
    自回归生成：
    - 给定一段文本，先用 BERT 编码
    - CAD 端从 BOS 开始逐 token 生成，直到 EOS 或 max_len
    """
    model.eval()

    # 文本编码
    enc = text_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
    )
    text_input_ids = enc["input_ids"].to(device)
    text_attention_mask = enc["attention_mask"].to(device)

    # 初始 CAD 序列：只包含 BOS
    cad_input_ids = torch.tensor([[BOS_TOKEN_ID]], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_len - 1):  # 减去 BOS 占的一个长度
            out = model(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                cad_input_ids=cad_input_ids,
                cad_labels=None,
            )
            logits = out["logits"]  # [1, L, V]
            next_logits = logits[:, -1, :]           # 取最后一个位置
            next_id = next_logits.argmax(dim=-1)     # greedy 解码 [1]

            # 拼到序列后面
            cad_input_ids = torch.cat(
                [cad_input_ids, next_id.unsqueeze(1)], dim=1
            )  # [1, L+1]

            # 遇到 EOS 提前停止
            if next_id.item() == EOS_TOKEN_ID:
                break

    # 去掉 BOS，自然包含 EOS（如果生成到了）
    pred_tokens = cad_input_ids[0, 1:].tolist()
    # 去掉中间可能出现的 PAD
    pred_tokens = [t for t in pred_tokens if t != PAD_TOKEN_ID]

    return pred_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Inference for Text2WhuCADLM: text -> WHUCAD vector sequence"
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input English description for the CAD part.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=256,
        help="Maximum generated CAD token length.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CKPT_PATH}. "
            "Please train the model first (train_text2whucad_lm.py)."
        )

    # 1) 文本 encoder / tokenizer
    text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text_encoder = AutoModel.from_pretrained("bert-base-uncased")
    text_encoder.to(device)

    # 2) 模型结构要和训练时一致
    model = Text2WhuCADLM(
        text_encoder=text_encoder,
        d_model=512,
        n_layers=4,
        n_heads=8,
        dim_ff=2048,
        max_len=args.max_len,
    )
    model.to(device)

    # 3) 加载权重
    state = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint from {CKPT_PATH}")

    # 4) 生成 CAD token 序列
    print("\nInput text:")
    print(args.text)
    print("\nGenerating CAD tokens...")
    pred_tokens = generate_cad_tokens(
        model, text_tokenizer, args.text, device=device, max_len=args.max_len
    )
    print(f"Generated {len(pred_tokens)} tokens.")

    # 5) 还原成 WHUCAD 向量
    vec_seq = decode_sequence(pred_tokens)  # [T, 1+N_ARGS]
    print("Decoded WHUCAD vector sequence shape:", vec_seq.shape)
    print("First few rows:")
    print(vec_seq[:5])

    # 如果你想保存成 .npy，顺手写一份：
    out_path = os.path.join(PROJECT_ROOT, "demo_pred_vec.npy")
    np.save(out_path, vec_seq)
    print(f"\nSaved predicted vector sequence to: {out_path}")


if __name__ == "__main__":
    main()
