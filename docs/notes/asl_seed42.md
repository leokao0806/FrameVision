# Subtask 1：DeBERTa-v3 + ASL seed 42 失敗診斷

依計畫 Task 11（P3），只看現有紀錄，不重新訓練。資料來源：

- `baselines/results/runs/st1_deberta_{asl,bce}_{13,42,2026}.json` 的逐 epoch 紀錄（train loss、dev Micro F1）
- 訓練程式 `baselines/src/models/text_baseline.py`、`baselines/src/loss.py`
- 訓練段落的標籤

以下沒有使用任何 test 資料。

## 觀察到的現象

訓練設定（三個 seed、兩種 loss 都相同）：

- AdamW，學習率固定 2e-5，沒有 warmup，也沒有學習率衰減
- batch size 4，最多 10 個 epoch，fp32
- gradient clipping 1.0
- loss 若出現非有限值，程式會直接停止

ASL 的參數是 gamma_neg 4、gamma_pos 1、clip 0.05。

| 模型 | 第 1 epoch loss | 第 2 epoch loss | 第 3 epoch loss | 第 4–10 epoch loss | dev Micro F1 |
|---|---|---|---|---|---|
| ASL seed 13 | 0.0535 | 0.0402 | 0.0277 | 持續下降到 0.0014 | 0.307–0.352 |
| ASL seed 2026 | 0.0549 | 0.0396 | 0.0280 | 持續下降到 0.0019 | 0.290–0.346 |
| **ASL seed 42** | 0.0537 | 0.0393 | **0.0488** | **0.0548–0.0561，不再下降** | 第 2 epoch 0.274，其餘全部 **0.245** |

- 三個 BCE 的 loss 都穩定下降，dev 在 0.259–0.362 之間，沒有類似現象。
- seed 42 的前兩個 epoch 和另外兩個 seed 幾乎一樣，第 3 個 epoch 的 loss 卻跳回接近第 1 個 epoch 的水準，之後 8 個 epoch 都停在那裡。
- 0.245 正是 dev 上「全部預測 None」的分數。每個 epoch 都會在 0.05–0.95 之間重新搜尋 threshold，所以從第 3 個 epoch 起，不論 threshold 設多少，都不如什麼都不預測。
- loss 一直是有限值，所以不是 NaN 或數值爆掉。

## 檢查：loss 平台是否等於「不看輸入」的最佳解

如果模型塌縮成輸出不隨輸入改變，它能達到的最低 loss，就是每個 frame 只輸出一個固定機率時的最佳 ASL loss。我用 708 個英文訓練段落的標籤直接算出這個值，不需要模型：

- **每個 frame 輸出固定機率時，最佳 ASL loss 為 0.0512。** 各 frame 的最佳固定機率約在 0.42–0.52 之間。
- seed 42 第 4–10 個 epoch 的 loss 是 0.0548–0.0561，只比這個下限高 7–10%。epoch 內還有 dropout 和參數更新，略高於下限是合理的。
- 第 1 個 epoch 的 loss（三個 seed 都是 0.054 左右）也落在同一個水準，代表模型剛開始時同樣接近「不看輸入」的狀態。

固定機率約 0.45，也解釋了 dev 為什麼一直是 0.245：

- threshold 低於約 0.45 時，每個段落都會預測幾乎所有 frame，precision 極低。
- threshold 高於約 0.45 時，什麼都不預測，等於全部猜 None。
- 搜尋結果因此總是選後者。

ASL 在這裡的最佳固定機率接近 0.5 而不是接近 0，原因是 gamma_neg = 4：負例的 loss 乘上 (p − 0.05)^4，p 小於 0.5 時幾乎為 0，所以對「到處都猜中間值」的懲罰很小。

計算方式（可重現；在 `baselines/` 下執行）：

```python
import json, sys
import numpy as np
sys.path.insert(0, ".")
from src.data_loader import load_official_data, stack_labels
paragraphs, _ = load_official_data()
ids = set(json.load(open("results/course_split.json"))["train"])
train = [p for p in paragraphs if p["lang"] == "EN" and
         ((p["source"] == "second_chunk" and p["doc_id"] in ids) or p["source"] == "semeval23")]
Y = stack_labels(train).astype(float)                      # 708 x 14

def asl(p, y, gp=1, gn=4, clip=0.05, eps=1e-8):            # same as src/loss.py, element-wise
    xs_neg = np.minimum(1 - p + clip, 1.0)
    return -(y * np.log(np.maximum(p, eps)) * (1 - p) ** gp
             + (1 - y) * np.log(np.maximum(xs_neg, eps)) * (1 - xs_neg) ** gn)

grid = 1 / (1 + np.exp(-np.linspace(-12, 6, 20001)))
print(np.mean([min(asl(p, Y[:, j]).mean() for p in grid) for j in range(14)]))   # 0.0512
```

## 結論

- **發生了什麼：已找到。** seed 42 在第 3 個 epoch 塌縮成幾乎不看輸入的輸出，停在 ASL 的「固定輸出」解附近，之後再也沒有離開。支持這個判斷的證據有兩點：
  - loss 平台與固定輸出的最佳 loss 相符。
  - 所有 threshold 下，dev 都等於全部猜 None 的分數。
- **為什麼是 seed 42、為什麼在第 3 個 epoch：未解決。** 現有紀錄只有每個 epoch 的平均 loss 和 dev 分數，沒有 gradient norm、逐步 loss 或各 epoch 的 checkpoint，無法確認是哪一次更新造成的。

以下是可能有關、但**未經驗證**的因素：

- 沒有 warmup 的固定學習率
- batch size 只有 4
- ASL 的負例權重在 p < 0.5 時幾乎為 0，使固定輸出解周圍很平坦，一旦落入就難以離開

若要驗證，需要重跑並記錄逐步的 loss 與 gradient norm，這不在期末範圍內（Subtask 1 凍結）。

## 報告可以怎麼寫

Subtask 1 小節可以用一兩句說明：

- ASL seed 42 在第 3 個 epoch 塌縮成與輸入無關的輸出：其 loss 與最佳固定輸出解相同，dev 分數等於全部猜 None。
- 觸發原因未查明。
- 報告沿用 dev 選出的第 2 個 epoch，與期中相同。

數字若要放進報告，應從上面的程式與 runs JSON 產生，不要手打。
