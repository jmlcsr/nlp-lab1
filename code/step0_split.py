"""Step 0: NYT 数据加载、随机打乱、80/10/10 划分，并把结果落盘。

为什么必须先做这一步：
  作业要求「除特别说明外，所有实验均应使用相同的数据划分，以保证不同方法之间
  具有可比性」。把划分结果保存成 csv，后面每个 Task 直接读取，就彻底避免了
  「每个脚本各自设置随机种子」导致结果不可比的问题。

用法：
    python step0_split.py --stratify
"""
import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nyt", default=os.path.join("HW-1", "nyt.csv"), help="NYT 原始数据路径")
    ap.add_argument("--out_dir", default="data", help="划分结果输出目录")
    ap.add_argument("--seed", type=int, default=42, help="全局随机种子")
    ap.add_argument(
        "--stratify",
        action="store_true",
        help="按标签分层划分（推荐，可让 Macro-F1 更稳定）；启用后全部实验保持一致",
    )
    args = ap.parse_args()

    if not os.path.exists(args.nyt):
        raise SystemExit(f"找不到数据文件：{args.nyt}")

    df = pd.read_csv(args.nyt)
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    print("=" * 60)
    print("NYT 数据集概览")
    print("=" * 60)
    print(f"总样本数（去空去重后）: {len(df)}")
    print(f"类别数: {df['label'].nunique()}")
    print("类别分布:")
    print(df["label"].value_counts().to_string())
    print(f"文本平均词数（粗略）: {df['text'].str.split().str.len().mean():.1f}")
    print(f"最长文本词数（粗略）: {df['text'].str.split().str.len().max()}")

    # 统一随机打乱（作业要求：首先对 NYT 数据集进行随机打乱）
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # 80% / 10% / 10%
    strat = df["label"] if args.stratify else None
    train_df, tmp_df = train_test_split(
        df, test_size=0.2, random_state=args.seed, stratify=strat, shuffle=True
    )
    strat_tmp = tmp_df["label"] if args.stratify else None
    val_df, test_df = train_test_split(
        tmp_df, test_size=0.5, random_state=args.seed, stratify=strat_tmp, shuffle=True
    )

    os.makedirs(args.out_dir, exist_ok=True)
    print("=" * 60)
    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        path = os.path.join(args.out_dir, f"{name}.csv")
        part.to_csv(path, index=False)
        print(f"{name:5s}: {len(part):6d} 条  ({len(part) / len(df) * 100:.1f}%)  -> {path}")

    total = len(train_df) + len(val_df) + len(test_df)
    assert total == len(df), f"划分后总数 {total} != 原始 {len(df)}"
    print("=" * 60)
    print("数据划分完成。后续所有 Task 都读取 data/ 下的这三份文件。")


if __name__ == "__main__":
    main()
