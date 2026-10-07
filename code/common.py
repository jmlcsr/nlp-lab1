"""公共工具：分词、数据读取、指标计算、结果汇总。

所有实验脚本都必须通过本模块读取同一份 data/ 划分，
以保证「不同文本表示方法之间具有可比性」（作业要求）。

运行方式：这些脚本不直接执行，被 task*.py 导入。
"""
import json
import os
import re

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

# nltk 分词数据是否可用（惰性检测，避免每个脚本都重复探测）
_NLTK_OK = None


def tokenize(text):
    """统一分词器：小写 + nltk word_tokenize。

    nltk 的 punkt / punkt_tab 数据缺失时自动退化为正则分词，
    避免整个流程中断。正式跑实验前请先执行：

        import nltk; nltk.download("punkt"); nltk.download("punkt_tab")

    所有方法共用同一个分词器，保证公平比较。
    """
    global _NLTK_OK
    text = str(text).lower()

    if _NLTK_OK is None:
        try:
            from nltk import word_tokenize

            word_tokenize("sanity check")
            _NLTK_OK = True
        except Exception as exc:  # LookupError / ImportError 等
            print(
                f"[warn] nltk 分词数据不可用（{exc.__class__.__name__}），"
                "已退化为正则分词。建议执行 nltk.download('punkt_tab') 后重跑。"
            )
            _NLTK_OK = False

    if _NLTK_OK:
        from nltk import word_tokenize

        return word_tokenize(text)
    return re.findall(r"[a-z0-9']+", text)


def load_splits(data_dir="data"):
    """读取 Step 0 落盘的统一划分，返回 (train_df, val_df, test_df)。"""
    paths = {n: os.path.join(data_dir, f"{n}.csv") for n in ("train", "val", "test")}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "缺少划分文件：" + ", ".join(missing) + "\n请先运行：python step0_split.py --stratify"
        )
    return (pd.read_csv(paths["train"]), pd.read_csv(paths["val"]), pd.read_csv(paths["test"]))


def evaluate(y_true, y_pred, name="model", verbose=True):
    """计算 Accuracy 与 Macro-F1（作业指定的两个指标）。"""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"[{name}] Accuracy={acc:.4f}  Macro-F1={f1:.4f}")
    if verbose:
        print(classification_report(y_true, y_pred, digits=4))
    return {"name": name, "accuracy": float(acc), "macro_f1": float(f1)}


def save_result(record, path=None):
    """把一次实验的指标追加进 results/results.json（同名记录会被覆盖）。"""
    path = path or os.path.join("results", "results.json")
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    records = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            try:
                records = json.load(fh)
            except json.JSONDecodeError:
                records = []

    records = [r for r in records if r.get("name") != record.get("name")]
    records.append(record)
    records.sort(key=lambda r: r.get("name", ""))

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
    print(f"结果已记录到 {path}")
