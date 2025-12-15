# model_text2whucad_lm.py (适配您当前的扁平化 Tokenizer)

import torch
import torch.nn as nn
import torch.nn.functional as F

from whucad_tokenizer import VOCAB_SIZE

IGNORE_INDEX = -100


class Text2WhuCADLM(nn.Module):
    """
    升级版 Text2WhuCADLM：
    1. 使用 Cross-Attention 机制，让 CAD 生成过程能动态“查询”文本中的细节。
    2. 兼容您当前的扁平化 Tokenizer (1D Input)。
    """

    def __init__(self, text_encoder, d_model=256, n_layers=4, n_heads=8,
                 dim_ff=1024, max_len=512, dropout=0.1):
        super().__init__()

        # -------------------------------------------------------
        # 1. 文本编码部分 (Encoder)
        # -------------------------------------------------------
        self.text_encoder = text_encoder
        # 冻结 BERT 参数
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        text_hidden_size = self.text_encoder.config.hidden_size
        # 投影层：把 BERT 的 768 维映射到 d_model (256)
        self.text_proj = nn.Linear(text_hidden_size, d_model)

        # -------------------------------------------------------
        # 2. CAD 嵌入部分 (Embedding)
        # -------------------------------------------------------
        # 依然使用您当前的扁平化 VOCAB_SIZE
        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # -------------------------------------------------------
        # 3. 核心解码器 (Decoder with Cross-Attention)
        # -------------------------------------------------------
        # 使用 TransformerDecoderLayer，它自带 Self-Attention 和 Cross-Attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # -------------------------------------------------------
        # 4. 输出头
        # -------------------------------------------------------
        self.out_proj = nn.Linear(d_model, VOCAB_SIZE)
        self.max_len = max_len

    def forward(self,
                text_input_ids,
                text_attention_mask,
                cad_input_ids,
                cad_labels=None):
        """
        输入维度保持不变，兼容您现有的 train_text2whucad_lm.py
        text_input_ids:      [B, T_text]
        text_attention_mask: [B, T_text]
        cad_input_ids:       [B, L_cad]
        """
        device = cad_input_ids.device
        B, L = cad_input_ids.shape

        # ==========================================
        # Step 1: 处理文本 (作为 Memory)
        # ==========================================
        with torch.no_grad():
            # 获取 BERT 最后一层所有 token 的输出: [B, T_text, 768]
            # 注意：这里不再只取 [CLS]，而是取整个序列！
            text_out = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask
            )
            memory = text_out.last_hidden_state

        # 投影到 d_model: [B, T_text, 256]
        memory = self.text_proj(memory)

        # 生成 memory_mask (告诉 Decoder 文本里哪些是 PAD，不要看)
        # BERT mask: 1=Valid, 0=Pad
        # TransformerDecoder mask: True=Ignored(Pad), False=Attend
        memory_key_padding_mask = (text_attention_mask == 0)

        # ==========================================
        # Step 2: 处理 CAD 输入 (作为 Target)
        # ==========================================
        # 位置编码
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        tgt = self.token_emb(cad_input_ids) + self.pos_emb(pos_ids)
        tgt = self.dropout(tgt)

        # ==========================================
        # Step 3: 解码 (Cross-Attention 发生在这里)
        # ==========================================
        # 生成自回归掩码 (Causal Mask): 只能看左边
        tgt_mask = torch.triu(
            torch.ones(L, L, device=device, dtype=torch.bool),
            diagonal=1
        )

        # 输入 Decoder
        # tgt    = CAD 序列 (Query)
        # memory = 文本序列 (Key/Value) -> 模型会自动做 Cross-Attention
        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        # ==========================================
        # Step 4: 预测与 Loss
        # ==========================================
        logits = self.out_proj(out)  # [B, L, VOCAB]

        output = {"logits": logits}

        if cad_labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, VOCAB_SIZE),
                cad_labels.view(-1),
                ignore_index=IGNORE_INDEX
            )
            output["loss"] = loss

        return output