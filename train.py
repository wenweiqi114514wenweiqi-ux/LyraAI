"""
训练脚本：把对话数据喂给 Lyra（监督微调 SFT）+ 权重管理
=============================================================
- 数据: train_data 里的对话样本
- 目标: 自回归预测下一个 token，损失只用"助手回复"部分计算
- 权重管理：
    * lyra_best.pt —— 验证 loss 最低的最佳权重（只保留"好"的）
    * lyra_last.pt —— 最近一次训练的权重
    * 坏的权重（验证 loss 更差）不会覆盖 best，等于被剔除
    * --resume 可从已有权重继续训练，不用每次从零开始
    * 数据指纹校验：换过训练数据就不允许直接续训，防止污染

用法：
    从零训练：    python3 train.py --steps 4000
    续训：        python3 train.py --resume lyra_best.pt --steps 2000
"""

import argparse
import copy
import datetime
import hashlib
import json
import random
import torch
import torch.nn.functional as F

from lyra_model import Lyra, Config
from train_data import (CONVERSATIONS, KNOWLEDGE, build_vocab, build_samples,
                        build_text_samples, decode)

BEST_PATH = "lyra_best.pt"   # 最佳权重（验证 loss 最低）
LAST_PATH = "lyra_last.pt"   # 最近一次权重


# ============================================================
# 数据指纹：用来判断"训练数据是否变过"
# ============================================================
def data_fingerprint(conversations):
    text = json.dumps(conversations, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_windows(samples, block_size, stride):
    """把每条长对话切成长度 block_size 的滑动窗口，每个窗口是一个训练样本。"""
    windows = []
    for ids, mask in samples:
        n = len(ids)
        if n < 2:
            continue
        start = 0
        while start + 2 <= n:
            end = min(start + block_size, n)
            if end - start >= 2:
                windows.append((ids[start:end], mask[start:end]))
            start += stride
    return windows


def get_batch(windows, batch_size):
    """随机抽一个 batch 的窗口，返回 (x, labels)。标签右移+ignore_index。"""
    chosen = random.sample(windows, min(batch_size, len(windows)))
    maxlen = max(len(c[0]) for c in chosen)
    xs, ys = [], []
    for ids, mask in chosen:
        T = len(ids)
        xs.append(ids + [0] * (maxlen - T))
        lab = [-100] * maxlen
        for i in range(T - 1):
            if mask[i] == 1:
                lab[i] = ids[i + 1]
        ys.append(lab)
    return torch.tensor(xs), torch.tensor(ys)


def evaluate_loss(model, windows, batch_size, vocab_size):
    """在验证集上算平均 loss（不更新梯度），作为"权重好坏"的评判标准。"""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for _ in range(max(1, len(windows) // batch_size)):
            x, y = get_batch(windows, batch_size)
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size), y.view(-1), ignore_index=-100)
            total += loss.item() * x.size(0)
            count += x.size(0)
    model.train()
    return total / max(1, count)


def make_ckpt(model, cfg, char2idx, idx2char, step, loss, val_loss,
              fingerprint, note):
    return {
        "model_state": copy.deepcopy(model.state_dict()),
        "char2idx": char2idx,
        "idx2char": idx2char,
        "vocab_size": cfg.vocab_size,
        "cfg": {"vocab_size": cfg.vocab_size, "n_embd": cfg.n_embd,
                "n_head": cfg.n_head, "n_layer": cfg.n_layer,
                "block_size": cfg.block_size, "dropout": cfg.dropout},
        "step": step,
        "loss": loss,
        "val_loss": val_loss,
        "data_fingerprint": fingerprint,
        "note": note,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def load_ckpt(path):
    return torch.load(path, weights_only=False)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None,
                        help="从已有权重续训，如 lyra_best.pt")
    parser.add_argument("--steps", type=int, default=1000, help="本次训练步数（建议1000为一段）")
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)

    # 1. 词表与样本 + 数据指纹（对话 SFT + 正式材料预训练）
    char2idx, idx2char, vocab_size = build_vocab(CONVERSATIONS, KNOWLEDGE)
    samples = build_samples(CONVERSATIONS, char2idx) + build_text_samples(KNOWLEDGE, char2idx)
    fingerprint = data_fingerprint({"conversations": CONVERSATIONS, "texts": KNOWLEDGE})

    # 2. 模型（支持续训）
    cfg = Config()
    cfg.vocab_size = vocab_size
    if args.resume:
        ckpt = load_ckpt(args.resume)
        # 数据指纹校验：换过数据就不允许续训，防止旧权重被污染
        old_fp = ckpt.get("data_fingerprint")
        if old_fp is not None and old_fp != fingerprint:
            raise SystemExit(
                f"[拒绝续训] 数据已变化(旧指纹 {old_fp} ≠ 新指纹 {fingerprint})。"
                f"换过训练数据必须从零训练，否则旧权重会被新数据污染。")
        for k, v in ckpt["cfg"].items():
            setattr(cfg, k, v)
        model = Lyra(cfg)
        model.load_state_dict(ckpt["model_state"])
        resume_step = ckpt.get("step", 0)
        best_val = ckpt.get("val_loss", float("inf"))
        print(f"已从 {args.resume} 续训（已有 {resume_step} 步，历史最佳 val_loss={best_val:.4f}）")
    else:
        model = Lyra(cfg)
        resume_step = 0
        best_val = float("inf")
        print("从零初始化模型权重。")

    # 3. 窗口：90% 训练 / 10% 验证（用验证 loss 选最佳权重，防止过拟合当"好"）
    windows = build_windows(samples, cfg.block_size, stride=8)
    random.shuffle(windows)
    n_val = max(1, len(windows) // 10)
    val_windows = windows[:n_val]
    train_windows = windows[n_val:]
    batch_size = 2      # 500万模型在4GB内存下需进一步降低 batch 防 OOM
    print(f"训练样本 {len(train_windows)}  验证样本 {len(val_windows)}  batch={batch_size}")

    # 4. 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # 5. 训练循环
    max_steps = args.steps + resume_step
    print_every = 200
    eval_every = 200
    best_state = None

    def persist_best():
        """把当前最佳权重立刻写盘，防止训练中途崩溃丢失好权重"""
        torch.save({
            "model_state": best_state,
            "char2idx": char2idx,
            "idx2char": idx2char,
            "vocab_size": vocab_size,
            "cfg": {"vocab_size": vocab_size, "n_embd": cfg.n_embd,
                    "n_head": cfg.n_head, "n_layer": cfg.n_layer,
                    "block_size": cfg.block_size, "dropout": cfg.dropout},
            "step": best_step,
            "loss": None,
            "val_loss": best_val,
            "data_fingerprint": fingerprint,
            "note": "best",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }, BEST_PATH)

    print("=" * 60)
    print("开始训练（只对助手回复部分计算损失）...")
    print("=" * 60)

    for step in range(resume_step + 1, max_steps + 1):
        x, y = get_batch(train_windows, batch_size)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1),
                               ignore_index=-100)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % print_every == 0 or step == resume_step + 1:
            print(f"step {step:>5}   train_loss: {loss.item():.4f}")

        # 定期在验证集上评判，只记住"更好"的权重
        if step % eval_every == 0:
            val_loss = evaluate_loss(model, val_windows, batch_size, vocab_size)
            print(f"      -> 验证 loss: {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                best_step = step
                persist_best()   # 立刻写盘，崩溃也不丢
                print(f"      -> [已更新最佳权重] step {step}, val_loss {val_loss:.4f} [已写盘]")

    # 6. 保存：best（若本次有更优则更新）+ last
    final_val = evaluate_loss(model, val_windows, batch_size, vocab_size)
    if best_state is None:   # 训练太短没评估过，用当前权重兜底
        best_state = copy.deepcopy(model.state_dict())
        best_step = max_steps
        best_val = final_val
    # 用 best_state 直接保存最佳权重（best_state 已在训练中更新为最优）
    torch.save({
        "model_state": best_state,
        "char2idx": char2idx,
        "idx2char": idx2char,
        "vocab_size": vocab_size,
        "cfg": {"vocab_size": vocab_size, "n_embd": cfg.n_embd,
                "n_head": cfg.n_head, "n_layer": cfg.n_layer,
                "block_size": cfg.block_size, "dropout": cfg.dropout},
        "step": best_step,
        "loss": None,
        "val_loss": best_val,
        "data_fingerprint": fingerprint,
        "note": "best",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }, BEST_PATH)
    print(f"[保存最佳权重] {BEST_PATH}  step={best_step}  val_loss={best_val:.4f}")

    # 保存最新（本次训练结束时的权重）
    torch.save({
        "model_state": copy.deepcopy(model.state_dict()),
        "char2idx": char2idx,
        "idx2char": idx2char,
        "vocab_size": vocab_size,
        "cfg": {"vocab_size": vocab_size, "n_embd": cfg.n_embd,
                "n_head": cfg.n_head, "n_layer": cfg.n_layer,
                "block_size": cfg.block_size, "dropout": cfg.dropout},
        "step": max_steps,
        "loss": loss.item(),
        "val_loss": final_val,
        "data_fingerprint": fingerprint,
        "note": "last",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }, LAST_PATH)
    print(f"[保存最近权重] {LAST_PATH}  step={max_steps}  val_loss={final_val:.4f}")
    print("权重管理说明：坏权重不会覆盖 best，下次可用 --resume lyra_best.pt 续训。")


if __name__ == "__main__":
    train()
