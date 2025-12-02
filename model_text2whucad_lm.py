# model_text2whucad_lm.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from whucad_tokenizer import VOCAB_SIZE

IGNORE_INDEX = -100


class Text2WhuCADLM(nn.Module):
    """
    文本 -> CAD token 的自回归 LM：
      - 冻结 BERT，只用作文本编码
      - 自适应层：BERT hidden -> d_model
      - CAD 部分：token embedding + position embedding + TransformerEncoder(带 causal mask)
    训练目标：
      - 前缀 token -> 预测下一个 token
    """

    def __init__(self, text_encoder, d_model=256, n_layers=2, n_heads=4,
                 dim_ff=1024, max_len=512):
        super().__init__()
        self.text_encoder = text_encoder  # transformers.AutoModel

        # 冻结 BERT 参数
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.text_encoder.eval()

        # 记录文本 hidden 维度
        hidden_size = self.text_encoder.config.hidden_size

        # 文本自适应层：BERT hidden -> d_model
        self.text_adapter = nn.Linear(hidden_size, d_model)

        # CAD token embedding & position embedding
        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        # 自回归 Transformer（encoder + causal mask 实现自回归）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 输出层：d_model -> VOCAB_SIZE
        self.out_proj = nn.Linear(d_model, VOCAB_SIZE)

        self.max_len = max_len

    def encode_text(self, text_input_ids, text_attention_mask):
        """
        冻结 BERT：只做前向，不反传梯度。
        输出一个条件向量 text_cond: [B, d_model]
        """
        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
            )
            # [CLS] 向量
            cls = outputs.last_hidden_state[:, 0, :]  # [B, hidden_size]

        # 过自适应层 -> [B, d_model]
        text_cond = self.text_adapter(cls)
        return text_cond

    def forward(self,
                text_input_ids,
                text_attention_mask,
                cad_input_ids,
                cad_labels=None):
        """
        text_input_ids:    [B, T_text]
        text_attention_mask:[B, T_text]
        cad_input_ids:     [B, L]   (prefix LM 的输入序列)
        cad_labels:        [B, L]   (下一 token 标签，PAD 用 IGNORE_INDEX)
        """
        B, L = cad_input_ids.shape
        device = cad_input_ids.device

        # 1) 文本编码 + 自适应层
        text_cond = self.encode_text(text_input_ids, text_attention_mask)  # [B, d_model]

        # 2) CAD token + 位置编码
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # [B, L]
        x = self.token_emb(cad_input_ids) + self.pos_emb(pos_ids)           # [B, L, d_model]

        # 3) 文本条件注入：简单相加（FiLM 的最简形式）
        x = x + text_cond.unsqueeze(1)  # [B, L, d_model]

        # 4) 自回归 mask：位置 i 只能看 0..i
        causal_mask = torch.triu(
            torch.ones(L, L, device=device, dtype=torch.bool),
            diagonal=1
        )  # [L, L] 上三角 True = 禁止注意

        # TransformerEncoder 支持 attn_mask，True 表示不可见
        x = self.transformer(x, mask=causal_mask)  # [B, L, d_model]

        logits = self.out_proj(x)  # [B, L, VOCAB_SIZE]

        out = {"logits": logits}

        if cad_labels is not None:
            # 交叉熵，忽略 IGNORE_INDEX
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                cad_labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )
            out["loss"] = loss

        return out
