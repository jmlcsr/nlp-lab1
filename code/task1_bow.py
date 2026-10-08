"""Task 1: Binary Bag of Words / Word Frequency + Logistic Regression（30 分）

两种表示只有 CountVectorizer(binary=True/False) 一处差别，其余设置完全相同，
这样才是严格的「控制变量」对比。

用法：
    python task1_bow.py --mode binary
    python task1_bow.py --mode freq
"""
import argparse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from common import evaluate, load_splits, save_result, tokenize


def run(mode, train_df, val_df, test_df, min_df=2, C=1.0, class_weight=None):
    binary = mode == "binary"

    # 关键：词表只用训练集构建，val/test 只做 transform，避免信息泄漏
    vectorizer = CountVectorizer(
        binary=binary,           # True -> Binary BoW；False -> Word Frequency
        tokenizer=tokenize,      # 使用自定义分词器时必须把 token_pattern 置 None
        token_pattern=None,
        lowercase=True,
        min_df=min_df,           # 丢弃出现次数 < min_df 的极低频词，降维去噪
    )
    X_tr = vectorizer.fit_transform(train_df["text"])
    X_va = vectorizer.transform(val_df["text"])
    X_te = vectorizer.transform(test_df["text"])

    vocab_size = len(vectorizer.vocabulary_)
    print(f"[{mode}] 词表大小 |V| = {vocab_size}, 训练集矩阵维度 = {X_tr.shape}")
    print(f"[{mode}] 稀疏矩阵非零元素占比 = {X_tr.nnz / (X_tr.shape[0] * X_tr.shape[1]):.6f}")

    # max_iter 要放大：高维稀疏 + 多分类时默认 100 次往往不收敛
    # 注：sklearn >= 1.8 已忽略 n_jobs（多分类 LR 本身不用它），故不再传
    clf = LogisticRegression(max_iter=1000, C=C, class_weight=class_weight)
    clf.fit(X_tr, train_df["label"])

    # 类别不均衡的参照系：全部预测训练集里最多的类别
    # NYT 中 sports 约占 75%，所以"多数类基线"的 Accuracy 约 0.75 而 Macro-F1 仅约 0.29
    top_label = train_df["label"].value_counts().idxmax()
    baseline_pred = [top_label] * len(test_df)
    evaluate(
        test_df["label"], baseline_pred, name=f"多数类基线(全预测 {top_label})", verbose=False
    )

    model_name = "Binary-BoW + LR" if binary else "WordFrequency + LR"
    if class_weight:
        model_name += f" [class_weight={class_weight}]"
    evaluate(val_df["label"], clf.predict(X_va), name=f"{model_name} (val)")
    record = evaluate(test_df["label"], clf.predict(X_te), name=model_name)
    record["vocab_size"] = int(vocab_size)
    save_result(record)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="Step 0 产出的划分目录")
    ap.add_argument("--mode", choices=["binary", "freq"], default="binary")
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--class_weight", default=None, help='可传 "balanced" 处理类别不均衡')
    args = ap.parse_args()

    train_df, val_df, test_df = load_splits(args.data)
    run(args.mode, train_df, val_df, test_df, args.min_df, args.C, args.class_weight)


if __name__ == "__main__":
    main()
