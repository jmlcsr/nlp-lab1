"""收尾分析：BERT 的截断统计 + 错误分析。

产出（报告素材）：
    results/truncation_stats.json   # 截断统计（token 长度分布、超 64 的比例）
    results/error_analysis.md       # 错误分析（混淆矩阵 + 典型错例）

用法（项目根目录执行）：
    python code/analyze.py
    python code/analyze.py --limit_text 400    # 控制错例展示的文本长度
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from common import load_splits

MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LEN = 64


# ----------------------------------------------------------------------
# 1) 截断统计
# ----------------------------------------------------------------------
def truncation_stats(tokenizer, texts):
    """统计 tokenizer 编码后的真实长度，量化 max_length=64 的截断程度。"""
    lengths = []
    for text in texts:
        # truncation=False：拿到完整长度，才能知道被截掉多少
        lengths.append(len(tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]))
    lengths = np.asarray(lengths)

    stats = {
        "n_docs": int(len(lengths)),
        "max_length": MAX_LEN,
        "token_len_mean": float(lengths.mean()),
        "token_len_median": float(np.median(lengths)),
        "token_len_p90": float(np.percentile(lengths, 90)),
        "token_len_max": int(lengths.max()),
        "frac_exceeding_max_len": float((lengths > MAX_LEN).mean()),
        # 进入模型的比例 = 被保留 token 数 / 原始 token 数（逐文档求均值）
        "mean_kept_ratio": float(np.minimum(lengths, MAX_LEN).sum() / lengths.sum()),
    }
    return stats, lengths


# ----------------------------------------------------------------------
# 2) 错误分析
# ----------------------------------------------------------------------
def error_analysis(df, label_order, limit_text=300):
    """基于 results/bert_test_predictions.csv 做混淆矩阵与错例抽取。"""
    n = len(df)
    acc = float((df["gold"] == df["pred"]).mean())
    wrong = df[df["gold"] != df["pred"]].copy()

    # 混淆矩阵（行=真实，列=预测）
    cm = pd.crosstab(df["gold"], df["pred"]).reindex(
        index=label_order, columns=label_order, fill_value=0
    )

    lines = []
    lines.append("# BERT 错误分析\n")
    lines.append(f"- 测试集样本数：**{n}**")
    lines.append(f"- 正确数：**{n - len(wrong)}**，错误数：**{len(wrong)}**")
    lines.append(f"- Accuracy：**{acc:.4f}**（与 results.json 一致）\n")

    lines.append("## 混淆矩阵（行 = 真实，列 = 预测）\n")
    lines.append("| 真实 \\ 预测 | " + " | ".join(label_order) + " | 合计 |")
    lines.append("| --- | " + " | ".join(["---"] * (len(label_order) + 1)) + " |")
    for lab in label_order:
        row = cm.loc[lab]
        lines.append(
            f"| **{lab}** | " + " | ".join(str(int(v)) for v in row.values) + f" | {int(row.sum())} |"
        )
    lines.append("")

    lines.append("## 各类错分数量\n")
    lines.append("| 真实类别 | 样本数 | 错分数 | 错分率 |")
    lines.append("| --- | --- | --- | --- |")
    for lab in label_order:
        tot = int((df["gold"] == lab).sum())
        bad = int(((df["gold"] == lab) & (df["pred"] != lab)).sum())
        lines.append(f"| {lab} | {tot} | {bad} | {bad / max(tot, 1):.2%} |")
    lines.append("")

    lines.append("## 典型错例\n")
    if wrong.empty:
        lines.append("（无错例）")
    else:
        # 优先展示混淆次数最多的那一对
        pair_counts = wrong.groupby(["gold", "pred"]).size().sort_values(ascending=False)
        lines.append("按混淆类别对统计：\n")
        lines.append("| 真实 → 预测 | 次数 |")
        lines.append("| --- | --- |")
        for (g, p), c in pair_counts.items():
            lines.append(f"| {g} → {p} | {c} |")
        lines.append("")

        # 每个主要的混淆对挑 2 条
        shown = 0
        for (g, p), _ in pair_counts.items():
            if shown >= 5:
                break
            sub = wrong[(wrong["gold"] == g) & (wrong["pred"] == p)].head(2)
            for _, r in sub.iterrows():
                if shown >= 5:
                    break
                shown += 1
                snippet = " ".join(str(r["text"]).split())[:limit_text]
                lines.append(f"### 错例 {shown}：真实 `{g}` → 预测 `{p}`\n")
                lines.append(f"> {snippet}…\n")

    return "\n".join(lines), cm, acc, wrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--pred", default=os.path.join("results", "bert_test_predictions.csv"))
    ap.add_argument("--limit_text", type=int, default=300, help="错例展示的文本截断长度")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)

    # ---- 1) 截断统计（用全部 NYT 样本，最能代表整体）----
    train_df, val_df, test_df = load_splits(args.data)
    all_texts = pd.concat([train_df, val_df, test_df], ignore_index=True)["text"].tolist()
    print(f"截断统计：正在 tokenize {len(all_texts)} 篇文档 …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    stats, lengths = truncation_stats(tokenizer, all_texts)

    out_json = os.path.join("results", "truncation_stats.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)

    print("\n=== 截断统计（max_length = 64）===")
    print(f"  文档数                : {stats['n_docs']}")
    print(f"  平均 token 数         : {stats['token_len_mean']:.1f}")
    print(f"  中位数 / P90 / 最长   : {stats['token_len_median']:.0f} / "
          f"{stats['token_len_p90']:.0f} / {stats['token_len_max']}")
    print(f"  超过 64 的文档占比    : {stats['frac_exceeding_max_len']:.2%}")
    print(f"  平均保留 token 比例   : {stats['mean_kept_ratio']:.2%}")
    print(f"  -> 已写入 {out_json}")

    # ---- 2) 错误分析 ----
    if not os.path.exists(args.pred):
        print(f"\n[skip] 找不到 {args.pred}，跳过错误分析（请先跑 task3_bert.py）")
        return

    df = pd.read_csv(args.pred)
    label_order = sorted(set(df["gold"]) | set(df["pred"]))
    md, cm, acc, wrong = error_analysis(df, label_order, args.limit_text)

    out_md = os.path.join("results", "error_analysis.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(md)

    print("\n=== 错误分析 ===")
    print(cm.to_string())
    print(f"\n  错误样本数: {len(wrong)} / {len(df)}")
    print(f"  -> 已写入 {out_md}")


if __name__ == "__main__":
    main()
