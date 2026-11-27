# 2026-11-10：鎖定前讀過 test 的探索性 C2 分析（例外紀錄）

## 發生了什麼

為了回答老師對期中報告的評論 (2)「為什麼加文字反而變差？加的是標題還是全文？」，我們用現有的 Qwen2.5-VL 預測做了一次探索性分析。這次分析讀了 **test 的 Qwen 預測與 test 的 gold 標籤**，而計畫規定 `results/LOCKED` 建立前，C2 不得讀 test。當時計畫還沒開始執行、規則也還沒 commit，但這仍違反了已核准計畫的精神，所以在這裡記錄下來，並在期末報告的 Experimental Setup 揭露。

- 腳本：`scripts/explore_c2_text_effect.py`，只讀 `results/vlm_{dev,test}_metrics.json` 與資料標註，不改任何結果
- 這些 test 分數在期中報告時就已經看過、也已發表；這次新看到的是逐張圖的新增／拿掉標籤

## 看到的內容

Qwen 2a／2b 讀到的文字是「標題 + 文章前 1500 字元」（`MAX_TEXT_CHARS = 1500`）：

- train + dev 有 95% 的文章被截斷，中位數只看到全文的 50%
- test 每篇都被截斷，中位數只看到 43%

只用 train + dev 的 gold 標籤：每篇文章平均有 5.0 個文字 frame，每張圖平均有 1.11 個 frame；直接照抄文字 frame 的 precision 是 0.151，recall 是 0.683。

image-only → 加文字（2a）的變化：

| split | 新增標籤 | 新增中正確 | 新增中屬於文字 frame | 拿掉標籤 | 拿掉中原本正確 |
|---|---|---|---|---|---|
| dev（20 張） | 7 | 1 | 4 | 3 | 1 |
| test（40 張） | 14 | 0 | 9 | 14 | 4 |

在 2b（再加 gold 文字 frame）中，test 有 94% 的預測落在文字 frame 裡。

## 對計畫的影響

- E12（只給標題、給全文）是在看過上面的結果之後才寫定的。為了避免 test 影響選擇，E12 不做任何選擇：三種文字長度都跑 dev 與 test，全部報告。
- 「best model」仍然只用 dev 選；E12 的兩個設定和其他模型一樣是候選，規則不變。
- 期末報告的 C2 數字一律由 `analysis/vlm_text_overlap.py`（Task 10）在鎖定後產生，不引用本紀錄的數字。
