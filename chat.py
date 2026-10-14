"""
推理脚本：让训练好的 Lyra 自己组织语言回答
=============================================
- 输入一段用户话术，前面拼 <user>，后面拼 <asst>
- 模型自回归逐个生成 token，直到输出 <end> 停止
- 用【温度采样】生成，所以同一句话每次回答可能略有不同
  （证明它是在"组织语言"，而不是背诵固定答案）
- 【工具辅助】数学计算与简单翻译交给确定逻辑（Python 精确计算 + 常用词表），
  避免小模型在需要精确数字的地方产生幻觉。这是现代 AI 的"工具调用"做法：
  模型负责理解与表达，工具负责精确结果。
"""

import re
import sys
import torch
import torch.nn.functional as F

from lyra_model import Lyra, Config
from train_data import SPECIAL_TOKENS, encode, decode

# ============================================================
# 工具辅助：数学计算 + 简单翻译（确定逻辑，不靠模型瞎猜）
# ============================================================

# 常用中英互译词表（开源时可扩展）
TRANS_CN2EN = {
    "苹果": "apple", "香蕉": "banana", "水": "water", "太阳": "sun",
    "月亮": "moon", "星星": "star", "谢谢": "thank you", "不客气": "you're welcome",
    "对不起": "sorry", "再见": "goodbye", "你好": "hello", "猫": "cat",
    "狗": "dog", "红": "red", "蓝": "blue", "绿": "green",
    "米饭": "rice", "面包": "bread", "鸡蛋": "egg", "今天": "today",
    "明天": "tomorrow", "昨天": "yesterday", "雨": "rain", "雪": "snow",
    "爸爸": "father", "妈妈": "mother", "我": "I", "你": "you",
    "爱": "love", "学习": "learning", "早上好": "good morning",
    "晚安": "good night", "天琴": "Lyra", "家": "home", "朋友": "friend",
    "一": "one", "二": "two", "三": "three", "四": "four", "五": "five",
    "六": "six", "七": "seven", "八": "eight", "九": "nine", "十": "ten",
}
TRANS_EN2CN = {
    "thank you": "谢谢", "you're welcome": "不客气", "sorry": "对不起",
    "goodbye": "再见", "hello": "你好", "hi": "你好", "apple": "苹果",
    "cat": "猫", "dog": "狗", "water": "水", "sun": "太阳", "moon": "月亮",
    "red": "红", "blue": "蓝", "green": "绿", "rice": "米饭", "bread": "面包",
    "egg": "鸡蛋", "today": "今天", "tomorrow": "明天", "yesterday": "昨天",
    "rain": "雨", "snow": "雪", "father": "爸爸", "mother": "妈妈",
    "good morning": "早上好", "good night": "晚安", "one": "一", "two": "二",
    "three": "三", "four": "四", "five": "五", "i love you": "我爱你",
    "i love learning": "我爱学习", "i am fine": "我很好",
    "my name is lyra": "我的名字叫Lyra", "star": "星星", "home": "家", "friend": "朋友",
}

_CN_NUM = {"零": 0, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def to_int(s):
    """把阿拉伯数字或简单中文数字（0~99）转成 int，识别不了返回 None"""
    s = s.strip()
    if s.isdigit():
        return int(s)
    if s in _CN_NUM:
        return _CN_NUM[s]
    if "十" in s:                     # 十、十二、二十、二十三 等
        parts = s.split("十")
        tens = _CN_NUM.get(parts[0], 0) if parts[0] else 1
        ones = _CN_NUM.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return tens * 10 + ones
    return None


def tool_handle(prompt):
    """工具辅助入口：命中就返回确定答案字符串，否则返回 None 交给模型"""
    p = prompt.strip()
    # ---- 翻译：中文 → 英文（"苹果用英语怎么说" / "把苹果翻译成英文"）----
    m = re.search(r'([\u4e00-\u9fff]+)用(?:英语|英文)怎么说', p) or \
        re.search(r'把([\u4e00-\u9fff]+)翻译成(?:英文|英语)', p)
    if m:
        word = m.group(1).strip("，。！?！？的了！ ")
        if word in TRANS_CN2EN:
            return f"{word}用英语说是 {TRANS_CN2EN[word]}。"
        return f"这个词我暂时还没学过，建议去词典里查一下更准确。"
    # ---- 翻译：英文 → 中文（"thank you 是什么意思"）----
    m = re.search(r'([a-zA-Z][a-zA-Z ]*?)\s*是什么意思', p)
    if m:
        phrase = m.group(1).strip().lower()
        if phrase in TRANS_EN2CN:
            return f"{phrase}的意思是{TRANS_EN2CN[phrase]}。"
        return f"这句话我拿不准，暂时不敢硬翻。"
    # ---- 比大小（"5和8哪个大"）----
    m = re.search(r'([0-9零一二两三四五六七八九十]+)\s*和\s*([0-9零一二两三四五六七八九十]+)\s*哪个大', p)
    if m:
        a, b = to_int(m.group(1)), to_int(m.group(2))
        if a is not None and b is not None:
            if a > b:
                return f"{m.group(1)}比{m.group(2)}大。"
            if b > a:
                return f"{m.group(2)}比{m.group(1)}大。"
            return f"{m.group(1)}和{m.group(2)}一样大。"
    # ---- 四则运算（"3加4等于多少" / "2乘以3" / "7减2" / "6除以2"）----
    m = re.search(r'([0-9零一二两三四五六七八九十]+)\s*(加|加上|减|减去|乘以|乘|除以|除)\s*([0-9零一二两三四五六七八九十]+)', p)
    if m:
        a, op, b = to_int(m.group(1)), m.group(2), to_int(m.group(3))
        if a is not None and b is not None:
            if op in ("加", "加上"):
                return f"{m.group(1)}加{m.group(3)}等于{a + b}。"
            if op in ("减", "减去"):
                return f"{m.group(1)}减{m.group(3)}等于{a - b}。"
            if op in ("乘以", "乘"):
                return f"{m.group(1)}乘以{m.group(3)}等于{a * b}。"
            if op in ("除以", "除"):
                if b == 0:
                    return "除数不能是零哦。"
                res = a / b
                if res == int(res):   # 整除时显示整数，更干净
                    res = int(res)
                return f"{m.group(1)}除以{m.group(3)}等于{res}。"
    return None


# ============================================================
# 推理：模型自己组织语言（工具未命中时才走这里）
# ============================================================

def generate(model, char2idx, idx2char, prompt, max_new=120, temperature=0.3,
             top_k=20, block_size=512, repetition_penalty=1.2):
    """根据用户话术回答。先问工具（数学/翻译），未命中才让模型自己生成。"""
    tool_ans = tool_handle(prompt)
    if tool_ans is not None:
        return tool_ans

    model.eval()
    # 拼成模型认识的格式: <user> 话术 <asst>
    ids = [SPECIAL_TOKENS["<user>"]] + encode(prompt, char2idx) + \
          [SPECIAL_TOKENS["<asst>"]]
    idx = torch.tensor([ids])

    generated = []
    with torch.no_grad():
        for _ in range(max_new):
            idx_cond = idx[:, -block_size:]            # 只看最近 block_size
            logits = model(idx_cond)[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            # 重复惩罚：对已经生成过的 token 压低其分数，抑制循环
            if repetition_penalty is not None and generated:
                for gid in set(generated):
                    g = logits[0, gid]
                    logits[0, gid] = g / repetition_penalty if g > 0 \
                        else g * repetition_penalty
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # 随机采样

            if next_id.item() == SPECIAL_TOKENS["<end>"]:
                break
            generated.append(next_id.item())
            idx = torch.cat([idx, next_id], dim=1)

    return decode(generated, idx2char)


def load_model(path="lyra_best.pt"):
    data = torch.load(path, weights_only=False)
    cfg = Config()
    for k, v in data["cfg"].items():
        setattr(cfg, k, v)
    model = Lyra(cfg)
    model.load_state_dict(data["model_state"])
    return model, data["char2idx"], data["idx2char"]


def test_cases(model, char2idx, idx2char, cases, temperature=0.3):
    """对一组输入各生成 n 次，展示模型自己组织语言的表现"""
    for prompt in cases:
        print(f"\n【用户】{prompt}")
        for trial in range(2):   # 同一句话生成 2 次，看是否每次不同
            reply = generate(model, char2idx, idx2char, prompt,
                             temperature=temperature)
            print(f"  [回复{trial+1}] {reply}")


if __name__ == "__main__":
    model, char2idx, idx2char = load_model()
    print("模型已加载。（数学/翻译走工具辅助，其余走模型生成）")

    if len(sys.argv) > 1:
        # 命令行传入一句话，如: python chat.py "3加4等于多少"
        user_input = sys.argv[1]
        print(f"\n【用户】{user_input}")
        reply = generate(model, char2idx, idx2char, user_input)
        print(f"【Lyra】{reply}")
    else:
        # 默认跑一组测试
        print("\n" + "=" * 55)
        print("测试：工具辅助（数学 / 翻译 / 比大小）")
        print("=" * 55)
        test_cases(model, char2idx, idx2char, [
            "3加4等于多少", "7减2等于几", "2乘以3等于多少",
            "6除以2等于多少", "5和8哪个大", "苹果用英语怎么说",
            "谢谢用英语怎么说", "thank you 是什么意思", "我喜欢你用英语怎么说",
        ], temperature=0.3)

        print("\n" + "=" * 55)
        print("测试：模型自己组织语言（工具未命中）")
        print("=" * 55)
        test_cases(model, char2idx, idx2char, [
            "你好", "你是谁", "什么是人", "什么是人工智能",
            "我很难过", "你回答问题时要注意什么",
        ], temperature=0.3)
