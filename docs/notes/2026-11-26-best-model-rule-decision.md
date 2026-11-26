# 2026-11-26：最佳模型照預先註冊的規則選，報告揭露

## 發生了什麼

`docs/preregistration.md` 在 E12 跑之前就已 commit（`df106d4`），規則寫明 E12 的兩個設定（`qwen_2a_title`、`qwen_2a_fulltext`）和其他模型一樣，都是「最佳模型」的候選，只用 dev 的 Micro F1 選。

E12 跑完後（`c89d479`），`qwen_2a_fulltext` 的 dev Micro F1 以些微差距高過 `qwen_image_only`，因此依規則成為最佳模型。但它的 test 分數明顯較低，所以兩項主要比較也跟著改變：

- 「最佳模型 vs Qwen image-only」的差變成負值。
- 「最佳模型 vs 永遠猜最常見標籤」的差變小。

11/26 當時的數字只作為內部背景，報告不可引用；報告的數字一律由 `results/subtask2_final.json` 產生：

- dev：0.4800 對 0.4783，只有 20 張圖
- test：0.340 對 0.417

## 決定

En-Shuo 於 2026-11-26 決定：**照預先註冊的規則，不修改，報告中揭露。**

理由：規則寫定的時間早於 E12 的任何輸出；看過 dev 與 test 之後才改規則，就是事後調整。

## 報告要寫的內容

- **Experimental Setup**
  - 寫明選擇規則。
  - 寫明規則 commit 的時間早於 E12。
  - 寫明 E12 的設定也是候選。
- **Results**：照規則報告 4 項主要比較，不另外挑一個「看起來比較好」的模型取代。
- **Limitations**
  - 用這個例子說明：dev 只有 20 張，選擇本身有雜訊，一個 dev 上幾乎平手的模型在 test 上可能明顯較差。
  - 同時報告 `qwen_image_only` 在 test 上的表現，作為探索性比較。

## 之後可能的變化

Leo 的 E1–E8 跑完後，若有模型的 dev 分數更高，最佳模型會再改變。

這是同一條規則的正常結果，不需要另做決定。若最終最佳模型已不是 `qwen_2a_fulltext`，Limitations 裡的這個例子可以改成一句話帶過。
