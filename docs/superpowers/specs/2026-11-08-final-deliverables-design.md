# FrameVision 期末交付設計：In-class Presentation 與 Final Report

- 日期：2026-11-08
- 作者：En-Shuo Zhang（與 Claude 討論）
- 範圍：project.pdf 第 6 節（In-class Presentation, 10%）與第 7 節（Final Report, 15%）
- 狀態：待使用者審閱

## 1. 目標與限制

**目標**：在 2026-12-03 完成三件事，並讓教授與同學能清楚看到「我們自己的 key ideas」。

| 交付物 | 截止 | 規格 |
|---|---|---|
| 上台報告 | 2026-12-03 課堂（抽籤結果） | 20 分鐘 + 約 5 分鐘 QA |
| 投影片 PDF | 2026-12-03 23:59 | 上台後仍可修改 |
| Final Report PDF | 2026-12-03 23:59 | ≤10 頁，自含（不假設讀者看過其他文件） |
| Code zip | 2026-12-03 23:59 | ≤10 MB，不含模型、資料、快取 |

**已確認的事實**
- 團隊兩人（Leo Kao、En-Shuo Zhang），教授已同意兩人一組。
- 任務：SemEval-2027 Task 9，英文，主任務 Subtask 2（圖片 frame），Subtask 1 為輔。
- 11/19、11/24、12/1、12/3 四堂課都必須出席（缺席扣分）。

**成功標準**
1. 所有數字只有一個來源（結果 JSON），README、報告、投影片一致。
2. 「最佳模型」依事先寫死的規則用 dev 選出，test 只報告。
3. 每個主張都附 CI 或明確標示為探索性。
4. 報告有獨立的「Our Key Ideas」一節。
5. 投影片每個評分項（Team、Task、Resources、Best Model、Results、Analysis、Lessons、Visualization）都有對應頁面。

## 2. 現況與缺口（2026-11-08）

- **兩套程式碼分開**：本機 `nlp_project/`（Intermediate Report 用，有官方 scorer、bootstrap CI、3 seeds、Qwen VLM）；GitHub `leokao0806/FrameVision` main（Leo 的 Faster R-CNN 區域 + SigLIP + DeBERTa-v3-base + cross-attention + ASL，Exp 1–5）。En-Shuo 的程式碼在 `origin/enshuo/workspace`。兩邊 `course_split.json` 完全相同。
- **數字不一致**：GitHub README 寫 Exp 5 = Micro 0.4421 / Macro 0.3189 / Hamming 0.082；`reports/ablation_results.md` 寫 0.417 / 0.298 / 0.117。
- **嚴謹度不足**：GitHub 的實驗只跑 1 個 seed、沒有 CI、用 sklearn 重寫的計分而非原版官方 scorer；README 宣稱「outperforms all baselines」，但 40 張 test 圖的 CI 寬度約 ±0.13，差距不顯著。
- **核心假設未獲支持**：Exp 3（加區域，0.381）< Exp 2（只用整張圖，0.410）；Exp 5（0.417）≈ Exp 2。
- **與 proposal 的偏離未交代**：SAM / Grounding-DINO → Faster R-CNN MobileNetV3（COCO）+ 固定裁切備援；DeBERTa-v3-large → base；影像骨幹（proposal：微調 SigLIP／ViT-Large）與 Subtask 2 的文字骨幹（proposal：微調 DeBERTa-v3-large）實際都凍結、只訓練上層；Grad-CAM 未做，改以 attention 熱圖呈現（C5）。
- **老師對期中報告的評論（2026-11-10 收到）**：(1) 系統設計的動機與直覺是什麼；(2) 為什麼加文字反而變差，加的是標題還是全文；(3) 條列雖好讀但浪費版面，應節制使用。回應方式見 8.1。
- **Intermediate Report 承諾未完成**：第一批資料是否有幫助、ASL seed 42 失敗原因、VLM 的 None 判斷、稀有 frame 與質性錯誤分析、可解釋性（現有熱圖有一張來自訓練集）。

## 3. 設計 A：統一程式碼與評估

### 3.1 Repo 結構

最終只交一個 repo：GitHub `FrameVision` main。

```
FrameVision/
  src/ scripts/ tests/        ← Leo 的 region + cross-attention（原樣保留在根目錄）
  baselines/                  ← En-Shuo 的 baselines、VLM、Subtask 1（從 enshuo/workspace 原樣搬入，在此資料夾內執行）
  evaluation/                 ← 新增：共用評估層
  predictions/subtask2/<model>/seed<k>/{dev,test}.tsv
  results/                    ← evaluation/ 產生的 JSON 與圖表
  report/                     ← Final Report 的 md 與產生腳本
  slides/                     ← 投影片原始檔與 PDF
  docs/superpowers/specs/     ← 本 spec
```

兩邊都有 `src/`，放在不同資料夾就不會衝突，也不需要重寫任何一邊。

### 3.2 介面：只透過預測檔交接

- 每個模型、每個 seed，輸出 dev 與 test 的官方 Subtask 2 格式 TSV。Leo 端沿用 `export_predictions_tsv`，En-Shuo 端沿用 `official_scorer.py` 所用的格式；`evaluation/` 第一步檢查所有 TSV 格式一致，不一致就停止。
- 不需要隨機性的模型（trivial、LR、Qwen temperature 0）只有 `seed0/`。

### 3.3 評估層 `evaluation/`

1. 用 `baselines/src/official_scorer.py` 呼叫**未修改的**官方 scorer 計分。若與 sklearn 計算結果不同，以官方 scorer 為準，並在 log 記錄差異。
2. 指標：Micro F1（主要）、Macro F1（官方算法，15 類）、Hamming loss。
3. 多 seed 模型：點估計 = 3 個 seed 的平均；另報 std。
4. 95% CI：以文章為單位 bootstrap 2000 次。每次重抽後，各 seed 分別計分再取平均，因此 CI 反映文章抽樣變異，seed 變異由 std 另報。
5. Paired CI：同一組重抽樣本上，計算兩模型（seed 平均後）分數差。
6. 輸出 `results/subtask2_final.json`，README、報告、投影片的所有數字與圖表都由此檔自動產生，**不手打數字**。0.4421 與 0.417 的不一致由這一步解決。

### 3.4 事先寫死的規則（在跑任何新實驗前 commit 進 repo）

- **選擇**：epoch、threshold、E9 的 α、E10 的 prompt 版本，都只用 dev 決定。
- **最佳模型**：所有模型中，3 個 seed 平均 dev Micro F1 最高者（單次執行的模型就用該次分數）；平手時選較簡單的模型。test 不參與任何選擇。
- **最佳訓練模型**（E9、E11 會用到）：E1–E8 中 3 個 seed 平均 dev Micro F1 最高者。
- **主要比較（預先指定，共 4 項）**：
  1. 最佳模型 vs 永遠猜最常見標籤
  2. 最佳模型 vs Qwen image-only
  3. E8 vs E5（換成開放詞彙區域是否有幫助，核心假設）
  4. E6 vs E2（文字是否有幫助）
- 其他所有比較標示為探索性、未經多重比較校正。
- dev 只有 20 張圖，選擇本身有雜訊，於報告「Limitations」說明。
- **例外紀錄**：2026-11-10 為回答老師的評論 (2)，探索性 C2 分析讀過現有 Qwen 的 test 預測與 test 標籤（紀錄見 `docs/notes/2026-11-10-c2-prelock-test-look.md`）。E12 是在那之後寫定的，因此 E12 不做任何選擇：三種文字長度都跑 dev 與 test 並全部報告。報告的 Experimental Setup 須揭露這件事。

### 3.5 Subtask 1

凍結。沿用 Intermediate Report 的結果，不跑新實驗；Final Report 以一小節作為文字面的對照。唯一例外是 ASL seed 42 的診斷（見 4.3，P3）。

## 4. 設計 B：實驗清單（Subtask 2）

### 4.1 現有 baseline（不重跑，只匯出預測檔）

隨機、永遠猜最常見標籤（`Political`）、照抄 gold 文字 frame（2b）、Image LR（SigLIP）、Text LR（DeBERTa-v3）、late fusion（2a / 2b）、Qwen2.5-VL 三種 prompt（image only、+text、+gold frames）。

### 4.2 訓練模型（各 3 個 seed）

| 編號 | 設定 | 回答的問題 | 優先 |
|---|---|---|---|
| E1 | 只用文字（DeBERTa-v3-base + ASL） | 文字單獨能做到多少 | P0 |
| E2 | 只用整張圖（SigLIP + ASL） | 圖片單獨能做到多少 | P0 |
| E3 | 整張圖 + COCO 區域（K=3）+ ASL | 區域對純視覺模型是否有幫助 | P0 |
| E4 | Fusion + COCO 區域，BCE | 與 E5 只差 loss | P0 |
| E5 | Fusion + COCO 區域，ASL | 目前的完整模型 | P0 |
| E6 | Fusion，不加區域（只有整張圖 token），ASL | E5 vs E6：區域的貢獻；E6 vs E2：文字的貢獻 | P0 |
| E7 | 整張圖 + Grounding-DINO 區域（K=3）+ ASL | 與 E3 比：換偵測器的效果 | P1 |
| E8 | Fusion + Grounding-DINO 區域，ASL | 與 E5 比：核心假設 | P1 |
| E11 | 最佳訓練模型設定，換訓練資料：(a) 預設（Leo 的 pipeline 只用英文，第一批 400 + 第二批 37 篇）、(b) 移除 `initial-training-data` 來源（只剩第二批 37 篇） | 第一批資料是否有幫助 | P3 |

E1–E8 共 24 次訓練，都在快取特徵上訓練小模型。E11 的實際張數以資料載入器的統計為準。注意：Leo 的模型只用英文圖訓練，而 Image LR 用了含多語的 647 張，報告必須寫明各模型的訓練集。

### 4.3 VLM 與組合

| 編號 | 設定 | 優先 |
|---|---|---|
| E9 | 組合分數 = α × Qwen image-only 的 0/1 預測 + (1−α) × 最佳訓練模型的機率；α ∈ {0.25, 0.5, 0.75}，α 與 threshold 只用 dev 選；對最佳訓練模型的 3 個 seed 各做一次 | P2 |
| E10 | Qwen image-only prompt v2：明確列出 None 選項與其定義；最多試 2 個版本，只在 dev 上比較，選定後才跑 test | P2 |
| E12 | Qwen 圖 + 文字（2a）的文字長度：(a) 只給標題、(b) 給全文不截斷；與現有 2a（標題 + 前 1500 字元）合成三種長度。回答老師「加的是標題還是全文」。三種設定在 2026-11-10 寫定，dev 與 test 都跑、全部報告，不從中挑選任何設定 | P2 |
| — | Subtask 1 ASL seed 42 失敗診斷：只看現有 log 與 loss 曲線，不重跑；查不出原因則在報告寫「未解決」 | P3 |

### 4.4 開放詞彙區域（E7、E8）

- 模型：`IDEA-Research/grounding-dino-tiny`（Hugging Face transformers 內建）。
- 文字提示詞彙**只從 14 個 frame 的定義（Card et al., 2015）推導**，每個 frame 1–3 個可見實體（例如：protest sign、police officer、national flag、hospital、banknote）。詞彙表在跑 E7/E8 之前寫定並 commit，只依 frame 定義，不參考任何 dev/test 圖片。
- box threshold 0.35、text threshold 0.25（模型預設值），不調參。
- 取分數最高的 3 個框；不足 3 個時，用與 COCO 版本相同的固定裁切補足。除了偵測器，其他設定與 E3/E5 完全相同。
- 兩種偵測器都要統計**涵蓋率**：多少張圖的區域全部或部分來自固定裁切。
- 執行位置：區域抽取只需跑一次（約 700 張圖）；若 Leo 的 Mac（MPS）不支援或太慢，改在 En-Shuo 的 8 GB CUDA GPU 上抽取，再把結果交給 Leo。

### 4.5 優先順序

時間不夠時從後面砍：P0 → P1 → P2 → P3。P0 與統一評估是 Final Report 的最低要求。

## 5. 設計 C：分析

| # | 分析 | 資料 | 負責 |
|---|---|---|---|
| C1 | 圖文 frame 差異：每篇文章的圖片 frame vs 文字 frame（段落聯集），分成一致、互補、不同三類；畫「有某文字 frame 時，各圖片 frame 出現的機率」熱圖 | 只用 gold 標籤；11/20 鎖定數字前只用 train + dev，避免 test 標籤影響模型設計，鎖定後再把 test 加入圖表 | En-Shuo |
| C2 | 加文字讓 VLM 變差的原因：比較 image-only、三種文字長度（E12 與現有 2a）與 2b 的預測標籤數、預測與 gold 文字 frame 的重疊，以及相對 image-only「新增／拿掉」了哪些標籤、其中幾個正確、幾個來自文字 frame；另列各模型實際讀到的文字長度 | 現有 Qwen 預測與 E12 | En-Shuo |
| C3 | 每個 frame 的 F1（附 support）、稀有 frame 與 None | 最佳模型、Qwen image-only、E2 | En-Shuo |
| C4 | 質性錯誤分析：從 dev/test 挑約 8 個案例，歸納錯誤類型 | 預測檔 | En-Shuo |
| C5 | 區域分析：兩種偵測器的涵蓋率、偵測範例、dev/test 圖片上的 attention 熱圖（註明 attention 不等於解釋） | E3、E5、E7、E8 | Leo |
| C6 | 沿用資料洩漏分析（隨機段落切分 0.292 vs 按文章切分 0.203） | 已完成 | En-Shuo |

C1 不依賴任何新實驗，可立即開始。

## 6. 分工與時程

### 6.1 分工

- **Leo Kao**：在自己的 pipeline 加 seed 參數與 TSV 輸出；E1–E8、E11；Grounding-DINO 區域抽取與詞彙表；C5；更新架構圖。
- **En-Shuo Zhang**：把程式碼搬回 main 的 `baselines/`；建立 `evaluation/` 並匯出現有 baseline 預測檔；E9、E10、E12；ASL 診斷；C1–C4、C6。
- **共同**：投影片與報告依上面的負責範圍分頁、分節撰寫，最後互相審閱。

### 6.2 時程

| 日期 | 里程碑 |
|---|---|
| 11/9–11/11 | A 完成：repo 合併、評估層可用、3.4 的規則與 4.4 的詞彙表 commit；C1 同步開始 |
| 11/11–11/15 | P0：E1–E6；E10 在 dev 上試；E12 dev 與 test |
| 11/14–11/19 | P1：E7、E8；P0 完成後做 E9 |
| **11/20** | **鎖定所有數字**，之後不再改任何結果 |
| 11/18–11/24 | C 分析 |
| 11/23–12/1 | 投影片與報告同時撰寫（11/26 感恩節） |
| 12/1–12/2 | 計時排練 20 分鐘；檢查報告 ≤10 頁、zip ≤10 MB |
| 12/3 | 上台；23:59 前交報告 PDF、code zip、投影片 PDF |

## 7. 設計 D：投影片（20 分鐘，約 17 頁 + 備用頁）

| # | 內容 | 評分項 | 主講 |
|---|---|---|---|
| 1 | 隊名、Team number、兩位成員 | Team | — |
| 2 | 開場：真實新聞照片 + 標題，請台下猜 frame，揭曉圖片與文字 frame 不同 | Excitement | En-Shuo |
| 3 | 什麼是 framing：14 個 frame 的圖示格子 | Task | En-Shuo |
| 4 | 任務輸入輸出、2a / 2b、None、Micro F1 | Task | En-Shuo |
| 5 | 資料與按文章切分的理由（洩漏 0.292 → 0.203） | Soundness | En-Shuo |
| 6 | C1：圖片 frame ≠ 文字 frame（熱圖） | Excitement | En-Shuo |
| 7 | Resources 精簡表（附 URL） | Resources | En-Shuo |
| 8 | Baseline 總覽（圖示）；若最佳模型是 Qwen 或 E9，此頁講其完整架構 | Best Model | En-Shuo |
| 9 | Region-aware cross-attention 架構圖，每個元件旁標一句設計直覺（見 8.1） | Best Model | Leo |
| 10 | COCO vs Grounding-DINO 偵測範例與涵蓋率 | Analysis | Leo |
| 11 | 主要結果：帶 CI 的點圖 | Results | En-Shuo |
| 12 | Ablation：區域有沒有用？文字有沒有用？ | Results | Leo |
| 13 | C2：為什麼加文字讓 VLM 變差 | Analysis | En-Shuo |
| 14 | 每個 frame 的 F1、稀有 frame 與 None | Analysis | En-Shuo |
| 15 | 2–3 個錯誤案例（附圖） | Analysis | Leo |
| 16 | Lessons learned 3–4 點 | Lessons | 各一半 |
| 17 | 總結與兩人分工 | Team | — |
| 備用 | 完整結果表、超參數、Subtask 1 結果 | QA | — |

- 不貼大表格；所有圖表由 `results/subtask2_final.json` 自動產生（擴充 `baselines/src/visualize.py`）。
- 每人約講 10 分鐘。

## 8. 設計 E：Final Report（≤10 頁）

| 節 | 內容 | 頁數 |
|---|---|---|
| 1 Introduction | framing 背景、為何看圖片、設計動機（見 8.1），列出 3 個 key ideas | 0.75 |
| 2 Our Key Ideas | K1–K3（措辭待結果確定） | 0.5 |
| 3 Task and Data | 範例圖、資料統計、切分 | 1 |
| 4 Related Work | Card 2015、Piskorski 2023、Arora 2025、ASL、Grounding-DINO，用自己的話寫 | 0.5 |
| 5 Resources | 表格附 URL | 0.5 |
| 6 Methods | baseline、VLM prompt、region-aware 模型（架構圖）、開放詞彙區域、E9 組合 | 2 |
| 7 Experimental Setup | 3.4 的規則、seed、官方 scorer、CI 算法 | 0.5 |
| 8 Results | 主圖 + 精簡表、ablation | 1.5 |
| 9 Analysis | C1–C5 | 1.5 |
| 10 Limitations, Lessons, Conclusion | | 0.5 |
| 11 Contributions | 兩人分工 | 0.25 |
| References | | 約 0.5 |

**Key ideas 候選**
- **K1**：圖片 frame 不等於文字 frame，有量化數據；文字甚至可能誤導模型（C1、C2）。
- **K2**：以 frame 定義推導的開放詞彙區域加上 cross-attention，檢驗局部視覺線索是否有用；結果無論正負都報告（E5–E8、C5）。
- **K3**：小資料下的評估方法：按文章切分、量化資料洩漏、事先寫死選模型規則、附 CI；說明許多看似的進步落在雜訊內。

**產生方式**：沿用 `generate_doc.py`（md → docx → PDF），頁數檢查由 4 改為 10，zip 仍檢查 ≤10 MB。

### 8.1 回應老師對期中報告的評論

| 評論 | 回應位置與做法 |
|---|---|
| (1) 設計動機與直覺 | Introduction 用一段話，把每個設計選擇對應到一個觀察：M1 圖片 frame 不等於文字 frame（C1 的數字），所以模型必須看圖，文字只能當脈絡；M2 訓練圖片只有數百張、test 40 張，所以先做 trivial baseline、凍結編碼器與零樣本 VLM，並附 CI；M3 framing 線索常在圖的局部（標語、警徽、病床、國旗），整張圖的向量會稀釋它們，所以取區域，並以文字為 query 做 cross-attention，讓文字決定看哪裡、而不是直接注入 frame；M4 標籤稀疏，所以用 ASL。M3、M4 寫成假設，由 E5–E8 檢驗。投影片第 9 頁同步 |
| (2) 加文字變差；標題或全文 | Methods 用一張小表寫明各模型讀到的文字：Qwen 2a／2b 為標題 + 前 1500 字元；Text LR 與 late fusion 為全文（逐段編碼後取平均）；Leo 的模型為前 256 個 DeBERTa token。Analysis 的 C2 回答原因，E12 回答標題 vs 1500 字元 vs 全文 |
| (3) 節制條列 | 正文以段落為主，每段第一句寫結論，接著是數字與原因；條列只用於少數平行的短清單（例如 proposal 偏離），全文最多 3 處；數字放表格或圖，不在正文重複列出 |

## 9. 風險與對策

| 風險 | 對策 |
|---|---|
| Leo 不同意合併方式或分工 | 11/10 前開會確認本 spec；有分歧先改 spec 再動工 |
| Grounding-DINO 在 MPS 上跑不動 | 區域抽取改在 En-Shuo 的 CUDA GPU 上做 |
| 所有新實驗都不顯著 | K3（評估方法）與 K1（圖文差異）撐起 key ideas；誠實報告「什麼沒用、為什麼」 |
| dev 只有 20 張，選擇不穩 | 預先指定主要比較；於 Limitations 說明 |
| 時間不夠 | 依 P3 → P2 → P1 順序砍；P0 為底線 |
| zip 超過 10 MB | 排除 data、cache、checkpoints；預測 TSV 很小 |

## 10. 不做的事

- LoRA 微調 Qwen2.5-VL（8 GB GPU 與時程風險過高）
- SAM 分割（以 Grounding-DINO 取代，報告中說明理由）
- Subtask 1 新實驗
- 官方 SemEval-2027 提交（學期後可選，與本 spec 無關）

## 11. 開工前需確認

1. Team number（向教授或 Canvas 確認），投影片第 1 頁需要。
2. Leo 同意本 spec 的分工與 repo 結構。
3. 投影片工具（PowerPoint、Google Slides 或 Beamer），12/3 需輸出 PDF。
