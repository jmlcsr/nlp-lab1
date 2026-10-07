"""Task 3: BERT-base-uncased 微调（20 分）

作业硬性要求：
    预训练模型 : google-bert/bert-base-uncased
    max_length : 64
    epochs     : 3

用法：
    python task3_bert.py --limit 64 --epochs 1     # 先冒烟测试
    python task3_bert.py --epochs 3                # 正式训练
"""
import argparse
import inspect
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# transformers 5.x 仍从顶层导出 Trainer/TrainingArguments，这里做兼容兜底
try:
    from transformers import Trainer, TrainingArguments
except ImportError:  # pragma: no cover
    from transformers.trainer import Trainer
    from transformers.training_args import TrainingArguments

from common import load_splits, save_result

MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LEN = 64


class TextClfDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(
            list(texts), truncation=True, padding="max_length", max_length=MAX_LEN
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=-1)
    return {
        "accuracy": accuracy_score(pred.label_ids, preds),
        "macro_f1": f1_score(pred.label_ids, preds, average="macro"),
    }


def build_training_args(out_dir, epochs, batch_size, lr):
    """按当前 transformers 版本过滤参数名，避免版本改名导致崩溃。"""
    use_cuda = torch.cuda.is_available()
    bf16_ok = use_cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()

    kwargs = dict(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        bf16=bf16_ok,                 # 40 系显卡支持 bf16，数值更稳
        fp16=use_cuda and not bf16_ok,
        seed=42,
        report_to="none",
        dataloader_num_workers=0,
    )

    params = inspect.signature(TrainingArguments.__init__).parameters
    # 4.46 起 evaluation_strategy 改名为 eval_strategy
    kwargs["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "epoch"

    dropped = [k for k in kwargs if k not in params]
    if dropped:
        print(f"[warn] 当前 transformers 版本不支持这些参数，已忽略: {dropped}")
        kwargs = {k: v for k, v in kwargs.items() if k in params}

    return TrainingArguments(**kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="models/bert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3, help="作业要求 3")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只用前 N 条做冒烟测试")
    args = ap.parse_args()

    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"设备: {torch.cuda.get_device_name(0)}")

    train_df, val_df, test_df = load_splits(args.data)
    if args.limit > 0:
        train_df = train_df.head(args.limit)
        val_df = val_df.head(max(args.limit // 4, 8))
        test_df = test_df.head(max(args.limit // 4, 8))
        print(f"[冒烟测试] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # 标签映射：只在训练集上建立，val/test 复用同一映射
    label2id = {lab: i for i, lab in enumerate(sorted(train_df["label"].unique()))}
    id2label = {i: lab for lab, i in label2id.items()}
    y_tr = train_df["label"].map(label2id).tolist()
    y_va = val_df["label"].map(label2id).tolist()
    y_te = test_df["label"].map(label2id).tolist()
    assert not pd.isna(y_te).any(), "测试集出现训练集未见的类别"
    print(f"类别映射: {label2id}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    ds_train = TextClfDataset(train_df["text"], y_tr, tokenizer)
    ds_val = TextClfDataset(val_df["text"], y_va, tokenizer)
    ds_test = TextClfDataset(test_df["text"], y_te, tokenizer)

    trainer = Trainer(
        model=model,
        args=build_training_args(args.out, args.epochs, args.batch_size, args.lr),
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("=== validation ===")
    print(trainer.evaluate(ds_val))
    print("=== test ===")
    test_metrics = trainer.evaluate(ds_test)

    record = {
        "name": f"BERT-base-uncased (max_len={MAX_LEN}, epochs={args.epochs})",
        "accuracy": float(test_metrics["eval_accuracy"]),
        "macro_f1": float(test_metrics["eval_macro_f1"]),
    }
    save_result(record)

    preds = np.argmax(trainer.predict(ds_test).predictions, axis=-1)
    os.makedirs("results", exist_ok=True)
    out_csv = os.path.join("results", "bert_test_predictions.csv")
    pd.DataFrame(
        {
            "text": test_df["text"].tolist(),
            "gold": [id2label[i] for i in y_te],
            "pred": [id2label[i] for i in preds],
        }
    ).to_csv(out_csv, index=False)
    print(f"测试集预测已保存到 {out_csv}（可用于报告的错误分析）")


if __name__ == "__main__":
    main()
