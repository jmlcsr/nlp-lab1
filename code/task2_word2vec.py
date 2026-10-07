"""Task 2: Word2Vec / GloVe 词向量平均 + Logistic Regression（50 分）

三组实验共用同一套「文档向量化 + LR」流程，只有词向量来源不同：
    glove -> 预训练 GloVe-100
    ag    -> 用 AG News 文本训练 Word2Vec-100
    nyt   -> 用 NYT 训练集文本训练 Word2Vec-100

文档向量 = 文档内所有「命中词表」的词向量取平均，即 e(d) = 1/n * sum(e(w_i))

用法：
    python task2_word2vec.py --source glove --glove D:\\datasets\\glove\\glove.6B.100d.txt
    python task2_word2vec.py --source ag
    python task2_word2vec.py --source nyt
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import evaluate, load_splits, save_result, tokenize

DIM = 100


def load_glove(path, dim=DIM):
    """按行读取 glove.6B.100d.txt，返回 {word: np.float32[dim]}。"""
    emb = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:  # 跳过格式异常的行走
                continue
            emb[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    print(f"GloVe 词表大小: {len(emb)}")
    return emb


def train_w2v(texts, dim=DIM, seed=42, epochs=10, min_count=5, sg=1):
    """用给定文本训练 Word2Vec，返回 {word: np.float32[dim]}。"""
    from gensim.models import Word2Vec

    sentences = [tokenize(t) for t in texts]
    print(f"训练语料句子数: {len(sentences)}, 总词数: {sum(len(s) for s in sentences)}")

    model = Word2Vec(
        sentences,
        vector_size=dim,
        window=5,
        min_count=min_count,
        sg=sg,          # skip-gram，小语料上通常优于 CBOW
        negative=5,
        epochs=epochs,
        workers=8,
        seed=seed,
    )
    emb = {w: model.wv[w].astype(np.float32) for w in model.wv.index_to_key}
    print(f"Word2Vec 词表大小: {len(emb)}")
    return emb


def build_doc_vectors(df, emb, dim=DIM, tag=""):
    """把每篇文档表示为其有效词向量的平均；无有效词的文档回退为零向量。"""
    X = np.zeros((len(df), dim), dtype=np.float32)
    hit_words = total_words = empty_docs = 0

    for i, text in enumerate(df["text"].tolist()):
        tokens = tokenize(text)
        vecs = [emb[t] for t in tokens if t in emb]
        total_words += len(tokens)
        hit_words += len(vecs)
        if vecs:
            X[i] = np.mean(vecs, axis=0)
        else:
            empty_docs += 1

    coverage = hit_words / max(total_words, 1)
    print(f"[{tag}] 词覆盖率 {coverage:.4f} | 无有效词文档 {empty_docs}/{len(df)}")
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--source", choices=["glove", "ag", "nyt"], required=True)
    ap.add_argument("--glove", default="embeddings/glove.6B.100d.txt")
    ap.add_argument("--ag", default="HW-1/ag.csv")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--min_count", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()

    train_df, val_df, test_df = load_splits(args.data)

    if args.source == "glove":
        emb = load_glove(args.glove)
        name = "GloVe-100 mean + LR"
    elif args.source == "ag":
        ag = pd.read_csv(args.ag)
        print(f"AG News 文本数: {len(ag)}")
        emb = train_w2v(ag["text"].tolist(), epochs=args.epochs, min_count=args.min_count)
        name = "W2V(AGNews)-100 mean + LR"
    else:
        # 只用训练集文本训练词向量，避免把 val/test 的文本统计泄漏进来
        emb = train_w2v(train_df["text"].tolist(), epochs=args.epochs, min_count=args.min_count)
        name = "W2V(NYT)-100 mean + LR"

    X_tr = build_doc_vectors(train_df, emb, tag=f"{args.source}:train")
    X_va = build_doc_vectors(val_df, emb, tag=f"{args.source}:val")
    X_te = build_doc_vectors(test_df, emb, tag=f"{args.source}:test")

    clf = LogisticRegression(max_iter=2000, C=args.C, n_jobs=-1)
    clf.fit(X_tr, train_df["label"])

    evaluate(val_df["label"], clf.predict(X_va), name=f"{name} (val)")
    record = evaluate(test_df["label"], clf.predict(X_te), name=name)
    record["source"] = args.source
    record["vec_dim"] = DIM
    save_result(record)


if __name__ == "__main__":
    main()
