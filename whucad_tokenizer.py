# whucad_tokenizer.py

import numpy as np
from whucad_constants import CMD_ARGS_MASK, PAD_VAL, ARGS_N, ALL_COMMANDS

NUM_CMDS = len(ALL_COMMANDS)
VAL_VOCAB_SIZE = ARGS_N  # 256

OP_OFFSET = 0
VAL_OFFSET = NUM_CMDS

PAD_TOKEN_ID = NUM_CMDS + VAL_VOCAB_SIZE
EOS_TOKEN_ID = PAD_TOKEN_ID + 1
BOS_TOKEN_ID = EOS_TOKEN_ID + 1

VOCAB_SIZE = BOS_TOKEN_ID + 1  # [0, VOCAB_SIZE-1]


def encode_sequence(vec_seq: np.ndarray, add_eos: bool = True):
    """
    vec_seq: [T, 1+N_ARGS], 每行 [cmd_idx, arg0..argN-1]
    返回: List[int] token ids
      - 操作: cmd_idx 直接用 [0..NUM_CMDS-1]
      - 参数: VAL_OFFSET + value (0..255)
      - 不编码 PAD_VAL(-1)
      - 最后可选加 EOS_TOKEN_ID
    """
    assert vec_seq.ndim == 2
    tokens = []

    for t in range(vec_seq.shape[0]):
        row = vec_seq[t]
        cmd_idx = int(row[0])
        if cmd_idx < 0 or cmd_idx >= NUM_CMDS:
            continue

        # 操作 token
        tokens.append(OP_OFFSET + cmd_idx)

        # 参数 token：按 CMD_ARGS_MASK 顺序，只取 mask=1 的槽
        mask = CMD_ARGS_MASK[cmd_idx]
        args = row[1:]

        used_indices = np.where(mask == 1)[0]
        for idx in used_indices:
            val = int(args[idx])
            if val == PAD_VAL:
                continue
            if not (0 <= val < VAL_VOCAB_SIZE):
                val = max(0, min(VAL_VOCAB_SIZE - 1, val))
            tokens.append(VAL_OFFSET + val)

    if add_eos:
        tokens.append(EOS_TOKEN_ID)

    return tokens


def decode_sequence(tokens):
    """
    token ids → [T, 1+N_ARGS] 整数矩阵
    用 CMD_ARGS_MASK 的顺序恢复参数位置，没填的设回 PAD_VAL。
    BOS/PAD 会自动忽略；遇到 EOS 停止。
    """
    if isinstance(tokens, list):
        tokens = np.array(tokens, dtype=np.int64)

    seqs = []
    cur_cmd = None
    cur_args = None
    cur_arg_cursor = 0

    N_ARGS = CMD_ARGS_MASK.shape[1]

    def flush_current():
        nonlocal cur_cmd, cur_args, cur_arg_cursor
        if cur_cmd is not None:
            cur_args[np.isnan(cur_args)] = PAD_VAL
            row = np.concatenate([[cur_cmd], cur_args.astype(np.int32)])
            seqs.append(row)
        cur_cmd = None
        cur_args = None
        cur_arg_cursor = 0

    for tok in tokens:
        tok = int(tok)
        if tok == EOS_TOKEN_ID:
            break
        if tok == PAD_TOKEN_ID or tok == BOS_TOKEN_ID:
            continue

        if tok < NUM_CMDS:
            # 新操作开启
            flush_current()
            cur_cmd = tok
            cur_args = np.full((N_ARGS,), np.nan, dtype=np.float32)
            cur_arg_cursor = 0
        else:
            # 参数 token
            if cur_cmd is None:
                continue
            val = tok - VAL_OFFSET
            mask = CMD_ARGS_MASK[cur_cmd]
            used_indices = np.where(mask == 1)[0]
            if cur_arg_cursor >= len(used_indices):
                continue
            arg_idx = used_indices[cur_arg_cursor]
            cur_args[arg_idx] = val
            cur_arg_cursor += 1

    flush_current()

    if len(seqs) == 0:
        return np.zeros((0, 1 + CMD_ARGS_MASK.shape[1]), dtype=np.int32)
    return np.stack(seqs, axis=0).astype(np.int32)
