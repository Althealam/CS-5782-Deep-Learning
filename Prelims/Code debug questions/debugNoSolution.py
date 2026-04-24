# 关键bug：
# 1. Cross Attention中Query/Key/Value的来源错误：对于text-vision的cross attention，query来自text，key和value来自vision
# 2. Attention的计算错误：Attention(Q, K, V) = softmax(QK^T/np.sqrt(d_k))*V，K是用来计算权重的，而不是用来做信息聚合的
# 3. VisionEncoder需要添加位置编码，由于Transformer没有位置信息，所以必须要加入positional encoding，否则模型无法区分patch顺序/token顺序（注意ViT的positional embedding是可学习的）

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding

    Expected input:
        x: (B, S, D) where
            B = batch size
            S = sequence length (number of text tokens or vision patches)
            D = hidden dimension or embedding size

    These are sinusoidal positional encodings:
        - even indices use sine
        - odd indices use cosine
        - different embedding dimensions use different frequencies
    """
    def __init__(self, dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: (B, S, D)
        return x + self.pe[:, :x.size(1)]

class VisionEncoder(nn.Module):
    def __init__(self, img_dim, hidden_dim, seq_len):
        super().__init__()
        self.proj = nn.Linear(img_dim, hidden_dim)
        self.pos = PositionalEncoding(hidden_dim, max_len=seq_len)

    def forward(self, x):
        # x: (B, P, img_dim), with P == seq_len
        # Bug4: forget to add positional embedding before using the projection layer
        x = self.proj(x)
        # fix: x = self.pos(x)
        return x                 

class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim, seq_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos = PositionalEncoding(hidden_dim, max_len=seq_len)

    def forward(self, input_ids):
        # input_ids: (B, T), with T == seq_len
        x = self.embed(input_ids)
        x = self.pos(x)
        return x

class CrossAttention(nn.Module):
    """
    Cross-attention from text to vision.

    Inputs:
        text_states:   (B, T, D)
        vision_states: (B, P, D)

        where
            B = batch size
            T = text sequence length (number of text tokens)
            P = number of image patches
            D = hidden dimension or embedding size
            T == P
    
    Computation:
        attn_scores = Q @ K^T         # (B, T, P)
        attn_weights = softmax(attn_scores, dim=-1)
        output = attn_weights @ V     # (B, T, D)

    Important:
        Each text token attends over all image patches and gathers a weighted
        combination of visual information.

        T == P
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, text_states, vision_states):
        """
        text_states:   (B, T, D)
        vision_states: (B, P, D)
        """
        d_q = text_states.size(-1)
        ### Bug1 and 2: this is cross attention from text to vision, so the query should be text, and key and value should be vision
        # Q = self.q_proj(text_states)
        # K = self.k_proj(vision_states)
        # V = self.v_proj(vision_states)
        Q = self.q_proj(vision_states)
        K = self.k_proj(vision_states)
        V = self.v_proj(text_states)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_q) #math.sqrt(d_q) not actually a bug
        attn_weights = F.softmax(attn_scores, dim=-1)
        ### Bug3: attention = softmax(QK^T/np.sqrt(d_k))*V
        # attended = torch.matmul(attn_weights, V)
        attended = torch.matmul(attn_weights, K)   # (B, T, D)    
        return self.out_proj(attended)


class VLModel(nn.Module):
    def __init__(self, vocab_size=1000, img_dim=128, hidden_dim=64, seq_len=16):
        super().__init__()
        self.seq_len = seq_len
        self.vision_encoder = VisionEncoder(
            img_dim=img_dim,
            hidden_dim=hidden_dim,
            seq_len=seq_len,
        )
        self.text_embedding = TextEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            seq_len=seq_len,
        )
        self.cross_attn = CrossAttention(hidden_dim=hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, image_patches, input_ids):
        vision_states = self.vision_encoder(image_patches)
        text_states = self.text_embedding(input_ids)
        fused = self.cross_attn(text_states, vision_states)
        logits = self.lm_head(fused)
        return logits
