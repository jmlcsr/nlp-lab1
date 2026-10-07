# 作业一：文本分类实现 —— 代码运行说明

本仓库为《作业一：文本分类实现》的完整代码与运行说明。

依次使用 **Bag of Words**、**Word Embedding / Word2Vec**、**预训练语言模型（BERT）** 三种文本表示方法，
在 NYT 新闻数据集上训练文本分类器，并在统一的 NYT Test Set 上以 **Accuracy** 与 **Macro-F1** 评价。

---

## 1. 目录结构

```
lab1/
├── HW-1/
│   ├── nyt.csv                 # 主数据集：text, label（评测用）
│   └── ag.csv                  # 辅助数据集：text（仅用于训练 Word2Vec）
├── code/
│   ├── common.py               # 公共工具：统一分词器、指标计算、结果汇总
│   ├── step0_split.py          # Step 0：统一数据划分（所有实验复用）
│   ├── task1_bow.py            # Task 1：Binary BoW / Word Frequency + LR
│   ├── task2_word2vec.py       # Task 2：GloVe / Word2Vec(AG) / Word2Vec(NYT) + LR
│   ├── task3_bert.py           # Task 3：BERT-base-uncased 微调
│   └── requirements.txt        # 依赖清单
├── data/                       # 【运行时生成】train/val/test.csv，被 .gitignore 排除
├── embeddings/                 # 【需自行下载】glove.6B.100d.txt
├── models/                     # 【运行时生成】BERT 权重
├── results/
│   ├── results.json            # 各方法的 Accuracy / Macro-F1 汇总
│   └── bert_test_predictions.csv  # BERT 测试集预测（供错误分析）
├── 作业一要求.md
├── 作业总览与详细步骤.md          # 实验步骤与踩坑记录
└── README.md
```

---

## 2. 环境

- Python **3.11**
- 依赖见 `code/requirements.txt`（实际验证版本：torch 2.6.0+cu124、transformers 5.19.0、
  gensim 4.4.0、scikit-learn 1.9.1、pandas 3.0.6、numpy 2.4.6、nltk 3.10.3）

```bash
# 1) 创建环境（conda 为例）
conda create -n lab1 python=3.11 -y
conda activate lab1

# 2) 安装 PyTorch —— 必须指定 CUDA 索引，否则会装到 CPU 版
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 3) 安装其余依赖
pip install -r code/requirements.txt
```

> **不激活环境也可以**，直接用解释器的绝对路径调用即可：
> `D:\Miniconda\envs\lab1\python.exe code/step0_split.py --stratify`

**下载 nltk 分词数据（首次必须执行，否则会退化为正则分词）：**

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

**下载 GloVe 预训练词向量（Task 2.1 必需）：**

- 官方地址：<http://nlp.stanford.edu/data/glove.6B.zip>（约 822 MB）
- 解压后**只需要** `glove.6B.100d.txt`（约 347 MB），放到 `embeddings/` 目录
- 由于体积原因该文件不随仓库提供，请自行下载

**推荐用仓库内的脚本完成**（自动只解压 100d、按 5% 显示速度与 ETA、下载完删除 zip）：

```bash
python code/download_glove.py
# 官方地址不可达时可换源：
python code/download_glove.py --url <镜像地址>
```

> 说明：Windows 自带的 `curl.exe` **不读取系统代理**，直连境外站点容易被限速到
> ~12 KB/s；而 Python 的 `urllib` **会**读取系统代理。这是该脚本用 Python 实现的原因。

---

## 3. 运行顺序

**⚠️ 必须按顺序执行。尤其 `step0_split.py` 不能跳过** ——
作业要求「所有实验使用相同的数据划分」，该脚本把划分结果落盘到 `data/`，
后续所有 Task 都读取这三份文件，从而保证不同方法之间的可比性。

```bash
# Step 0：统一数据划分（80% / 10% / 10%，随机打乱，seed=42）
python code/step0_split.py --stratify

# Task 1：两种词袋表示（30 分）
python code/task1_bow.py --mode binary
python code/task1_bow.py --mode freq

# Task 2：三组词向量实验（50 分）
python code/task2_word2vec.py --source glove --glove embeddings/glove.6B.100d.txt
python code/task2_word2vec.py --source ag
python code/task2_word2vec.py --source nyt

# Task 3：BERT 微调（20 分）
python code/task3_bert.py --limit 64 --epochs 1   # 可选：先冒烟测试
python code/task3_bert.py --epochs 3
```

---

## 4. 实验设置（所有方法统一）

| 项目 | 设定 |
| --- | --- |
| 数据划分 | NYT 先随机打乱，再按 **Training 80% / Validation 10% / Test 10%** 划分 |
| 随机种子 | `seed = 42`（划分与模型初始化统一使用） |
| 划分复用 | 由 Step 0 落盘，六个实验共用同一份 `data/{train,val,test}.csv` |
| 评价 | 统一在 **NYT Test Set** 上报告 **Accuracy** 与 **Macro-F1** |
| 分词 | 统一使用 `common.tokenize`（小写 + nltk `word_tokenize`） |
| 分类器 | Task 1 / Task 2 使用 Logistic Regression |
| 词向量维度 | 100（Task 2 三组实验统一） |
| 文档向量 | Task 2 中为文档内所有有效词向量的平均 $e(d)=\frac{1}{n}\sum_{i=1}^{n}e(w_i)$ |

**Task 3 额外参数：**

| 项目 | 值 |
| --- | --- |
| 预训练模型 | `google-bert/bert-base-uncased` |
| max_length | 64 |
| epochs | 3 |
| learning_rate | 2e-5 |
| batch size | 16 |

---

## 5. 输出说明

- `data/{train,val,test}.csv`：统一划分结果
- `results/results.json`：六个实验的 Accuracy / Macro-F1 汇总，格式：

  ```json
  [{"name": "Binary-BoW + LR", "accuracy": 0.9, "macro_f1": 0.89, "vocab_size": 12345}]
  ```

- `results/bert_test_predictions.csv`：BERT 在测试集上的 `text / gold / pred`，用于报告的错误分析
- 每个脚本运行时会同时打印 **词表大小 / 词覆盖率 / 无有效词文档数** 等诊断信息，便于在报告中分析

---

## 6. 常见问题

| 现象 | 原因与解决 |
| --- | --- |
| `LookupError: Resource punkt not found` | 未下载 nltk 数据，见第 2 节；脚本会自动退化为正则分词并打印警告 |
| pip 报 `WinError 10061 由于目标计算机积极拒绝` | 系统配置了本地代理（如 `127.0.0.1:7897`）但代理未启动；启动代理或改用国内镜像 |
| 走清华源报 `SSLEOFError` | 代理与镜像的 TLS 冲突；改用官方 PyPI，或设置 `NO_PROXY` 后重试 |
| `torch.cuda.is_available()` 为 False | 装成了 CPU 版；必须用 `--index-url https://download.pytorch.org/whl/cu124` 重装 |
| 显存不足（OOM） | 调小 `--batch_size`（如 8），或关闭占用显存的其它程序 |
| HuggingFace 模型下载慢 | 设置镜像：`$env:HF_ENDPOINT="https://hf-mirror.com"` |
| 下载脚本提示 `not on PATH` | 无害。本仓库一律用解释器绝对路径调用，不使用 `torchrun` 等命令行工具 |

---

## 7. 说明

- `embeddings/`、`models/`、`data/` 因体积或可由代码重新生成，已被 `.gitignore` 排除。
  拉取仓库后按第 2、3 节操作即可完整复现。
- 由于不同运行环境可能存在随机差异，各方法的绝对指标会有小幅波动，
  但**数据划分、分词、评价方式在所有方法之间保持完全一致**，方法间的横向比较是有效的。
