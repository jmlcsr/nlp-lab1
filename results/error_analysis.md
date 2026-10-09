# BERT 错误分析

- 测试集样本数：**1145**
- 正确数：**1125**，错误数：**20**
- Accuracy：**0.9825**（与 results.json 一致）

## 混淆矩阵（行 = 真实，列 = 预测）

| 真实 \ 预测 | business | politics | sports | 合计 |
| --- | --- | --- | --- | --- |
| **business** | 129 | 11 | 2 | 142 |
| **politics** | 4 | 139 | 0 | 143 |
| **sports** | 0 | 3 | 857 | 860 |

## 各类错分数量

| 真实类别 | 样本数 | 错分数 | 错分率 |
| --- | --- | --- | --- |
| business | 142 | 13 | 9.15% |
| politics | 143 | 4 | 2.80% |
| sports | 860 | 3 | 0.35% |

## 典型错例

按混淆类别对统计：

| 真实 → 预测 | 次数 |
| --- | --- |
| business → politics | 11 |
| politics → business | 4 |
| sports → politics | 3 |
| business → sports | 2 |

### 错例 1：真实 `business` → 预测 `politics`

> this spring, the missouri chamber of commerce urged the state legislature to accept the federal government's plan to expand medicaid for the poor and disabled.the business lobbying group had not suddenly gone rogue. here is how daniel p. mehan, its president, summarized his feelings about president …

### 错例 2：真实 `business` → 预测 `politics`

> carlsbad, calif. — on a calm day, a steady rain just about masks the sound of pacific ocean water being drawn into the intake valve from agua hedionda lagoon. listen hard, and a faint sucking sound emerges from the concrete openings, like a distant straw pulling liquid from a cup.at the moment, the …

### 错例 3：真实 `politics` → 预测 `business`

> washington — george w. bush hopes historians will judge his presidency more kindly than his contemporaries have, but a transition coming later this year — the departure of chairman ben s. bernanke of the federal reserve after an eight-year tenure — already casts a different light on the 43rd preside…

### 错例 4：真实 `politics` → 预测 `business`

> those are just some of the costs of the 16-day partial government shutdown that ended last month, the obama administration said in a detailed report released thursday . the effects may seem small in the context of the $16 trillion economy, but they can add up to less effective government service, an…

### 错例 5：真实 `sports` → 预测 `politics`

> baldwin, mich. — the search started the night of may, a sunday, when cullen finnerty went missing in the woods here, amid towering white pines and shrubby scrub oak trees and owls and white-tailed deer. by monday morning, helicopters circled overhead as cadaver dogs combed through the brush below.by…
