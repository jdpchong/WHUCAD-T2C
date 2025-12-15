import os
import random
import argparse  # [NEW] 引入 argparse 以便从命令行控制参数
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd  # [NEW] 用于处理 Excel
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from whucad_tokenizer import encode_sequence, BOS_TOKEN_ID, PAD_TOKEN_ID
from model_text2whucad_lm import Text2WhuCADLM, IGNORE_INDEX

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

# ----------------- 路径配置 -----------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
TEXT_ROOT = os.path.join(DATA_ROOT, "text")
VEC_ROOT = os.path.join(DATA_ROOT, "vec")

# [NEW] 日志文件路径
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "training_metrics.xlsx")


# ----------------- 辅助函数：记录到 Excel -----------------

def log_to_excel(file_path, exp_name, epoch, loss, note=""):
    """
    [NEW] 将训练数据追加到 Excel 表中
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_data = {
        "Experiment": [exp_name],
        "Timestamp": [now_str],
        "Epoch": [epoch],
        "Loss": [round(loss, 6)],
        "Note": [note]
    }
    new_df = pd.DataFrame(new_data)

    if os.path.exists(file_path):
        try:
            # 读取现有数据并追加
            old_df = pd.read_excel(file_path)
            # 使用 concat 代替 append (pandas 新版推荐)
            df = pd.concat([old_df, new_df], ignore_index=True)
        except Exception as e:
            print(f"[WARN] 无法读取现有的 Excel 文件: {e}，将创建新文件。")
            df = new_df
    else:
        df = new_df

    # 保存文件
    try:
        df.to_excel(file_path, index=False)
        print(f" >> [Log] Metrics saved to {file_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save log to excel: {e}")


# ----------------- 数据处理与 Dataset (保持不变) -----------------

def collect_paired_ids(text_root: str, vec_root: str) -> List[str]:
    paired_ids: List[str] = []
    for dirpath, _, filenames in os.walk(text_root):
        for fname in filenames:
            if not fname.lower().endswith(".txt"):
                continue
            text_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(text_path, text_root)
            sample_id, _ = os.path.splitext(rel_path)
            base_vec = os.path.join(vec_root, sample_id)
            if os.path.exists(base_vec + ".npy") or os.path.exists(base_vec + ".h5"):
                paired_ids.append(sample_id)
    paired_ids = sorted(paired_ids)
    print(f"Found {len(paired_ids)} paired samples (text + vec).")
    return paired_ids


class TextWhuCADLMDataset(Dataset):
    def __init__(self, ids: List[str]):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def _load_vec(self, sample_id: str) -> np.ndarray:
        base = os.path.join(VEC_ROOT, sample_id)
        npy_path = base + ".npy"
        h5_path = base + ".h5"
        if os.path.exists(npy_path):
            return np.load(npy_path).astype(np.int32)
        if os.path.exists(h5_path):
            if not HAS_H5PY:
                raise RuntimeError("h5py not installed.")
            with h5py.File(h5_path, "r") as f:
                key = "vec" if "vec" in f else list(f.keys())[0]
                arr = f[key][()]
            return arr.astype(np.int32)
        raise FileNotFoundError(f"Vec not found: {sample_id}")

    def _load_text(self, sample_id: str) -> str:
        text_path = os.path.join(TEXT_ROOT, sample_id + ".txt")
        with open(text_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        text = self._load_text(sample_id)
        vec_seq = self._load_vec(sample_id)
        cad_tokens = encode_sequence(vec_seq, add_eos=True)
        return {"text": text, "cad_tokens": cad_tokens}


def build_lm_example(cad_tokens, max_len: int):
    tokens = [BOS_TOKEN_ID] + cad_tokens
    if len(tokens) > max_len: tokens = tokens[:max_len]
    if len(tokens) < max_len: tokens = tokens + [PAD_TOKEN_ID] * (max_len - len(tokens))
    input_ids = tokens
    labels = input_ids[1:] + [PAD_TOKEN_ID]
    labels = [(tok if tok != PAD_TOKEN_ID else IGNORE_INDEX) for tok in labels]
    return input_ids, labels


def collate_fn(batch, text_tokenizer, max_cad_len=256):
    texts = [b["text"] for b in batch]
    cad_token_lists = [b["cad_tokens"] for b in batch]
    text_enc = text_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    input_list, label_list = [], []
    for cad_tokens in cad_token_lists:
        inp, lab = build_lm_example(cad_tokens, max_len=max_cad_len)
        input_list.append(torch.tensor(inp, dtype=torch.long))
        label_list.append(torch.tensor(lab, dtype=torch.long))
    return {
        "text_input_ids": text_enc["input_ids"],
        "text_attention_mask": text_enc["attention_mask"],
        "cad_input_ids": torch.stack(input_list, dim=0),
        "cad_labels": torch.stack(label_list, dim=0),
    }


# ------------------- 训练主程序 (Updated) -------------------

def main():
    # [NEW] 增加命令行参数解析，方便你区分不同的实验
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="baseline_v1", help="实验名称，用于在 Excel 中区分不同模型")
    parser.add_argument("--log_interval", type=int, default=10, help="每多少个 epoch 记录一次 Excel")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch Size")
    parser.add_argument("--epochs", type=int, default=200, help="总 Epoch 数")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Experiment: {args.exp_name} | Log Interval: {args.log_interval}")

    # 1) 数据准备
    all_ids = collect_paired_ids(TEXT_ROOT, VEC_ROOT)
    if len(all_ids) == 0:
        raise RuntimeError("No paired samples found.")

    random.seed(42)
    random.shuffle(all_ids)
    n_val = int(len(all_ids) * 0.1)
    train_ids = all_ids[n_val:]

    train_dataset = TextWhuCADLMDataset(train_ids)
    text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    max_cad_len = 128

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # GPU上建议设为 4 或 8
        collate_fn=lambda b: collate_fn(b, text_tokenizer, max_cad_len=max_cad_len),
    )

    # 2) 模型构建
    text_encoder = AutoModel.from_pretrained("bert-base-uncased")
    text_encoder.to(device)

    model = Text2WhuCADLM(
        text_encoder=text_encoder,
        d_model=256,
        n_layers=2,
        n_heads=4,
        dim_ff=1024,
        max_len=max_cad_len,
    )
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

    # 3) 训练循环
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(
                text_input_ids=batch["text_input_ids"],
                text_attention_mask=batch["text_attention_mask"],
                cad_input_ids=batch["cad_input_ids"],
                cad_labels=batch["cad_labels"],
            )
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

            if steps % 50 == 0:
                print(f"Epoch {epoch} Step {steps} Loss {loss.item():.4f}")

        avg_epoch_loss = total_loss / max(1, steps)
        print(f"Epoch {epoch} done, avg loss = {avg_epoch_loss:.4f}")

        # [NEW] 每 log_interval 个 epoch 记录一次 Excel
        if epoch % args.log_interval == 0:
            log_to_excel(
                file_path=LOG_FILE_PATH,
                exp_name=args.exp_name,
                epoch=epoch,
                loss=avg_epoch_loss,
                note="Cross-Attention Baseline"  # 你可以在这里写备注
            )

        # 保存 Checkpoint
        ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "text2whucad_lm_latest.pt"))


if __name__ == "__main__":
    main()