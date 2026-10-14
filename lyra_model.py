"""
迷你语言模型 Lyra（天琴）—— 模型骨架
========================================
一个基于 Transformer 的"最小可训练"语言模型结构。

当前阶段：只搭建骨架、初始化神经元（随机权重），
不喂入任何训练数据、不进行训练。

规模设计（适配当前 CPU + 4GB 内存环境）：
  vocab_size : 词汇表大小（占位值，喂数据前再按实际数据确定）
  n_embd     : 嵌入 / 隐藏维度（每个 token 被表示成的向量长度）
  n_head     : 注意力头数
  n_layer    : Transformer 层数
  block_size : 上下文长度（模型一次最多能"看"多少个 token）

底层原理：语言模型 = "预测下一个字"。给定一段 token，逐位置
输出"下一个 token 的概率分布"。所有组件都为此服务。
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================
# 0. 超参数（模型的"身形"）
# ============================================================
class Config:
    vocab_size = 4096   # 词表容量上限（词级分词后词更多，训练时按实际数据动态调整）
    n_embd     = 192    # 每个 token 的向量维度（256→192，回调到约200万参数）
    n_head     = 8      # 注意力头数
    n_layer    = 4      # Transformer 层数（6→4，参数约到 217 万）
    block_size = 1024   # 上下文长度：一次最多能看 1K 个 token
    dropout    = 0.05   # 随机丢弃：配合增多的数据防过拟合


# ============================================================
# 1. 层归一化 LayerNorm
#    对每个 token 的特征做标准化，让训练更稳定。
#    gamma/beta 就是"神经元"（可学习参数）。
# ============================================================
class LayerNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_embd))   # 可学习缩放
        self.beta  = nn.Parameter(torch.zeros(n_embd))  # 可学习偏移

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + 1e-5)
        return self.gamma * x + self.beta


# ============================================================
# 2. 因果自注意力（Transformer 的核心）
#    每个 token 去"关注"它之前的所有 token，捕捉词与词的关系。
#    因果（Causal）= 只能看过去，不能偷看未来（否则就是作弊）。
# ============================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "隐藏维度必须能被头数整除"

        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head

        # 把输入向量映射成 Query / Key / Value 三份（线性层=神经元）
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

        # 下三角掩码：保证注意力只能看到当前及之前的 token
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.shape  # B=batch, T=序列长度, C=隐藏维度

        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        # 拆成多个注意力头: (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 计算注意力分数：q 和 k 做点积
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        # 遮盖未来位置（未来位置的分数设为极小值）
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)          # 归一化成权重
        att = self.dropout(att)

        # 按权重聚合 value，得到每个位置的输出
        y = att @ v                              # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)                      # 多头结果合并


# ============================================================
# 3. 前馈网络 MLP
#    每个 token 独立经过的"全连接"处理，增强表达能力。
# ============================================================
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)   # 先放大
        self.act = nn.GELU()                                # 激活函数
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd)   # 再缩回
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))


# ============================================================
# 4. 一个 Transformer 块（堆 n_layer 个）
#    每层 = 注意力(建关系) + 前馈(增强) ，各带残差连接和归一化。
# ============================================================
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # 残差连接：保留原始信息
        x = x + self.mlp(self.ln2(x))
        return x


# ============================================================
# 5. 主模型 Lyra
#    token → 向量 → 逐层处理 → 预测下一个 token 的概率
# ============================================================
class Lyra(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # 每个 token 一个可学习向量（词嵌入）—— 模型的"词表记忆"
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # 每个位置一个可学习向量（位置编码）—— 让模型知道顺序
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd)          # 最后一层归一化
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # 输出层 与 词嵌入 共享权重（常见做法，省参数）

        self.token_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化神经元（权重）：随机赋值，让模型"一无所知"地开始"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        """idx: (B, T) 的 token 索引。当前阶段只定义接口，等待数据。"""
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"序列长度 {T} 超过上限 {self.cfg.block_size}"

        tok_emb = self.token_embedding(idx)                 # (B, T, C)
        pos = torch.arange(0, T, device=idx.device)
        pos_emb = self.position_embedding(pos)              # (T, C)
        x = tok_emb + pos_emb                               # 语义 + 位置
        x = self.blocks(x)                                  # 逐层处理
        x = self.ln_f(x)
        logits = self.lm_head(x)                            # (B, T, vocab_size)
        return logits


# ============================================================
# 6. 参数统计工具
# ============================================================
def count_parameters(model):
    """返回总参数量（可训练），以及按模块分解"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    per_module = {}
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if n > 0:
            per_module[name] = n
    return total, per_module


# ============================================================
# 7. 主入口：只初始化，不喂数据，不训练
# ============================================================
if __name__ == "__main__":
    cfg = Config()
    model = Lyra(cfg)

    total, per_module = count_parameters(model)

    print("=" * 55)
    print("迷你大语言模型 Lyra —— 已搭建骨架，未喂数据")
    print("=" * 55)
    print(f"模型结构: {cfg.n_layer}层 Transformer, {cfg.n_head}头, 维度{cfg.n_embd}")
    print(f"词汇表(占位): {cfg.vocab_size}   上下文长度: {cfg.block_size}")
    print(f"运行设备: CPU   PyTorch版本: {torch.__version__}")
    print("-" * 55)
    print(f"总参数量（神经元总数）: {total:,}")
    print("-" * 55)
    print("按模块分解:")
    for name, n in per_module.items():
        print(f"  {name:20s} {n:>10,}")
    print("  (注: lm_head 与 token_embedding 共享权重，表中各自列出，")
    print("    故表中相加会比总数多一份 lm_head 的 32,768 参数)")
    print("=" * 55)
    print("当前状态：模型已初始化（权重随机），等待喂数据后开始训练。")
