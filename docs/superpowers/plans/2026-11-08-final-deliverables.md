# FrameVision 期末交付 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把兩套程式碼統一成一個 repo，用同一個評估層產生所有數字，補齊實驗與分析，並在 2026-12-03 交出投影片、Final Report 與 code zip。

**Architecture:** GitHub `FrameVision` main 為唯一 repo。Leo 的 pipeline 留在根目錄，En-Shuo 的程式碼搬進 `baselines/`；兩邊只透過 `predictions/subtask2/` 底下的官方格式 TSV（加上 E9 需要的機率 JSON）交接。`evaluation/` 不 import 任何一邊的 `src`，只讀預測檔、呼叫未修改的官方 scorer，輸出 `results/subtask2_final.json`；分析、圖表、報告表格都只讀這個 JSON。

**Tech Stack:** Python 3.14（En-Shuo 的 venv：PyTorch 2.11 + CUDA、transformers 5.22.1、scikit-learn）、pytest、matplotlib、Ollama + Qwen2.5-VL 7B、Grounding-DINO tiny、python-docx + Microsoft Word（PDF 轉檔）。

**Spec:** `nlp_project/docs/superpowers/specs/2026-11-08-final-deliverables-design.md`

## Global Constraints

- 工作目錄：`C:\UC-main\FrameVision`（新 clone，不用 scratchpad 裡的那份）。En-Shuo 的 Python：`PY=/c/UC-main/nlp_project/.venv/Scripts/python.exe`。
- Branch：En-Shuo 在 `final/unify`，Leo 在 `final/region-experiments`，各自開 PR 進 main。`git push` 與開 PR 前先問使用者。
- Commit message 用 conventional 前綴（`feat:`、`fix:`、`test:`、`docs:`、`chore:`）。
- Seeds：`[13, 42, 2026]`（與 Intermediate Report 相同）。
- Bootstrap：以文章為單位，2000 次，`seed=0`；文章 id = 圖片檔名去掉副檔名。
- Threshold 搜尋範圍：`np.arange(0.05, 0.96, 0.05)`，以 dev Micro F1 最大化。
- E9 的 α ∈ `{0.25, 0.5, 0.75}`。
- Grounding-DINO：`IDEA-Research/grounding-dino-tiny`，box threshold 0.35、text threshold 0.25，K=3，不足時用與 COCO 版本相同的固定裁切補足。
- 官方 scorer：`data/starter-kit/starter-kit/subtask2a_2b/scorer-subtask-2.py`，不修改，呼叫方式 `-g gold -p pred -f labels -o`，最後一行為 `micro\tmacro`。
- 預測檔格式：每行 `image_name\tframe1\tframe2...`；沒有 frame 時只有 `image_name`（代表 None）。15 欄矩陣 = 14 個 frame + 最後一欄 None。
- 預測檔路徑：`predictions/subtask2/<model>/seed<k>/{dev,test}.tsv`，機率檔 `{dev,test}.probs.json`。單次執行的模型用 `seed0`。以 `_` 開頭的資料夾（`_gold`、`_reference`）不是模型。
- 模型名稱（資料夾名）固定為：`most_frequent`、`copy_text_frames_2b`、`image_lr`、`text_lr`、`late_fusion_2a`、`late_fusion_2b`、`qwen_image_only`、`qwen_2a`、`qwen_2b`、`qwen_image_only_v2`、`qwen_2a_title`、`qwen_2a_fulltext`、`combo_qwen_best`、`e1_text_only`、`e2_vision_global`、`e3_vision_region_coco`、`e4_fusion_coco_bce`、`e5_fusion_coco_asl`、`e6_fusion_global_asl`、`e7_vision_region_gdino`、`e8_fusion_gdino_asl`、`e11_<最佳訓練模型名>_no_batch1`。
- 不手打任何結果數字：README、報告、投影片的數字都從 `results/subtask2_final.json` 產生。
- 11/20 鎖定數字：建立 `results/LOCKED` 之前，C1、C2、C4 不得讀 test 標籤或 test 預測。已發生一次例外（2026-11-10，見 `docs/notes/2026-11-10-c2-prelock-test-look.md`），報告須揭露。
- 老師對期中報告的三點評論（設計動機、加文字變差與文字長度、節制條列）的回應方式照 spec 8.1。
- Final Report ≤10 頁；code zip ≤10 MB，不含 data、cache、checkpoints、`.venv`、模型權重。
- Subtask 1 不跑新實驗。
- 報告與投影片用英文（課程要求）；計畫與內部筆記用繁體中文；程式碼與註解用英文（沿用 repo）。

## Review Focus

1. **某個預測 TSV 少了圖片、多了圖片或有重複列** → 評估必須停下，錯誤訊息要指出是哪個檔案、哪些圖片。（Task 3）
2. **某張圖沒有任何預測 frame（空列），或預測裡寫了字面上的 `None`** → 一律當作 None 欄 = 1，不當成格式錯誤。（Task 3）
3. **某個模型還沒跑完**（例如 E8 只有 dev、或少一個 seed、或完全沒跑）→ 評估照樣完成；選模型時跳過沒有 dev 的模型；主要比較標成 `missing`，不崩潰。（Task 5）
4. **Leo 的機率檔類別順序和 `labels.txt` 不同** → E9 依類別名稱重新排序，結果與順序無關。（Task 8）
5. **Grounding-DINO 版特徵的整張圖 token 或文字 token 跟 COCO 版不同** → E7/E8 與 E3/E5 的比較會混入其他變因；必須檢查並在不一致時停下。（Task L3）

## 前置條件（11/10 前）

- [ ] En-Shuo 與 Leo 開會確認 spec 與本計畫的分工、branch 與資料夾結構。Leo 不同意的地方先改 spec 與本計畫，再動工。
- [ ] 確認 Team number（投影片第 1 頁）。
- [ ] 決定投影片工具（PowerPoint、Google Slides 或 Beamer），Task 14 會用到。

## 任務總覽

| Task | 負責 | 優先 | 目標日期 | 依賴 |
|---|---|---|---|---|
| 1 Repo 合併 | En-Shuo | P0 | 11/9 | 前置條件 |
| 2 事先寫死的規則 | En-Shuo | P0 | 11/9 | 1 |
| 3 預測檔讀寫 | En-Shuo | P0 | 11/10 | 1 |
| 4 計分與 bootstrap | En-Shuo | P0 | 11/10 | 3 |
| 5 評估主程式 | En-Shuo | P0 | 11/11 | 4 |
| 6 匯出現有 baseline | En-Shuo | P0 | 11/11 | 5 |
| L1 seed 與預測檔輸出 | Leo | P0 | 11/11 | 1 |
| L2 E6 模型 | Leo | P0 | 11/11 | L1 |
| L3 Grounding-DINO 區域 | Leo | P1 | 11/14 | L1 |
| L4 跑 E1–E8、E11 | Leo | P0/P1/P3 | 11/19 | L1–L3、2 |
| 7 E10 Qwen prompt v2 | En-Shuo | P2 | 11/15 | 6 |
| 7b E12 Qwen 文字長度 | En-Shuo | P2 | 11/15 | 7 |
| 8 E9 組合 | En-Shuo | P2 | 11/19 | 5、L4 的 P0 |
| 9 C1 圖文 frame 差異 | En-Shuo | — | 11/12 起 | 6 |
| 10 C2 VLM 與文字 | En-Shuo | — | 11/18 | 6、7b |
| 11 ASL seed 42 診斷 | En-Shuo | P3 | 11/19 | 1 |
| 12 鎖定數字與圖表 | En-Shuo | — | 11/20–11/24 | 全部實驗 |
| L5 C5 區域分析與架構圖 | Leo | — | 11/24 | L4 |
| 13 Final Report | 兩人 | — | 12/2 | 12、L5 |
| 14 投影片與排練 | 兩人 | — | 12/2 | 12、L5 |
| 15 打包與交件 | En-Shuo | — | 12/3 | 13、14 |

---

### Task 1: Repo 合併（En-Shuo）

**Files:**
- Create: `C:\UC-main\FrameVision\`（clone）、`baselines/`（從本機 `nlp_project` 複製）
- Modify: `.gitignore`、`README.md`（只加「Repo layout」一節，結果表格留給 Task 15）

**Interfaces:**
- Produces: `baselines/` 可在其資料夾內執行；`baselines/data` 指向本機資料；根目錄可跑 pytest。

- [ ] **Step 1: Clone 並開 branch**

```bash
cd /c/UC-main && git clone https://github.com/leokao0806/FrameVision.git && cd FrameVision && git checkout -b final/unify
```

- [ ] **Step 2: 複製 En-Shuo 的程式碼**：來源以本機 `C:\UC-main\nlp_project` 為準（它是交 Intermediate Report 用的版本，比 `origin/enshuo/workspace` 新）。複製 `src/`、`scripts/`、`tests/`、`README.md`、`requirements.txt`、`requirements.lock.txt`、`results/`（含 `runs/`、`cache/`、`course_split.json`）到 `baselines/`。不要複製 `.venv/`、`_backup_*`、`data/`、PDF/DOCX、`__pycache__/`。另外把 `nlp_project/docs/superpowers/specs/2026-11-08-final-deliverables-design.md` 與 `nlp_project/docs/superpowers/plans/2026-11-08-final-deliverables.md` 複製到 repo 的 `docs/superpowers/specs/` 與 `docs/superpowers/plans/`，把 `nlp_project/docs/notes/` 整個複製到 repo 的 `docs/notes/`（含 11/10 的例外紀錄與給 Leo 的資料檢查說明）。

- [ ] **Step 3: 連結資料**：在 Windows 建立 junction，讓 `baselines/data` 指向本機資料。

```bash
cmd //c mklink /J "C:\UC-main\FrameVision\baselines\data" "C:\UC-main\nlp_project\data"
```

- [ ] **Step 4: 更新 `.gitignore`**：加入 `baselines/data`、`baselines/results/cache/`、`baselines/results/official_scoring/`、`cache/`、`checkpoints/`、`report/*.docx`、`report/*.pdf`、`report/*.zip`、`__pycache__/`。`predictions/` 與 `results/*.json` 要進 git。

- [ ] **Step 5: 在 venv 安裝 pytest**

```bash
$PY -m pip install pytest
```

- [ ] **Step 6: 驗證兩邊的測試都還能跑**

Run: `cd baselines && $PY -m unittest discover -s tests`，Expected: 全部 OK（與 Intermediate Report 交件時相同）。
Run: `cd .. && $PY -m pytest tests -q`，Expected: Leo 的 32 個測試通過；若因為缺少 torchvision 等套件而無法匯入，記錄是哪個套件、補裝後重跑，不修改 Leo 的程式。

- [ ] **Step 7: Commit**

```bash
git add baselines .gitignore README.md docs/superpowers docs/notes && git commit -m "chore: merge En-Shuo baselines into baselines/"
```

---

### Task 2: 事先寫死的規則（En-Shuo）

**Files:**
- Create: `docs/preregistration.md`

- [ ] **Step 1: 寫 `docs/preregistration.md`**（英文），內容逐條照抄 spec 3.4：dev-only 選擇（epoch、threshold、E9 的 α、E10 的 prompt 版本）；「best model」與「best trained model」的定義與平手規則；4 項主要比較；其他比較標為 exploratory, uncorrected；dev 只有 20 張的限制。加一行：「Test labels and test predictions are not used by C1, C2 or C4 before `results/LOCKED` exists.」再加一段例外紀錄（照抄 spec 3.4 最後一條，英文）：on 2026-11-10 an exploratory C2 look read the existing Qwen test predictions and test labels to answer the instructor's comment, see `docs/notes/2026-11-10-c2-prelock-test-look.md`; E12 was specified after that look, so E12 selects nothing: all three text lengths run on dev and test and are all reported.平手時的「較簡單」順序寫成：trivial < logistic regression < zero-shot VLM < trained single-modality < trained fusion < combination。

- [ ] **Step 2: Commit**（必須在 Task L4 與 Task 7 跑任何新實驗之前）

```bash
git add docs/preregistration.md && git commit -m "docs: pre-register model selection rule and primary comparisons"
```

---

### Task 3: 預測檔讀寫（En-Shuo）

**Files:**
- Create: `evaluation/__init__.py`、`evaluation/prediction_io.py`
- Test: `tests/test_eval_io.py`

**Interfaces:**
- Produces:
  - `NONE_LABEL = "None"`
  - `read_labels(path: str) -> list[str]`（14 個 frame，依檔案順序）
  - `read_frames_tsv(path: str, labels: list[str]) -> dict[str, list[str]]`：image_name → frame 清單；空列或字面 `None` → 空清單；不在 labels 裡的名稱或重複的 image_name → `ValueError`（訊息含檔案路徑）
  - `write_frames_tsv(path: str, rows: dict[str, list[str]]) -> None`（自動建立資料夾，換行用 `\n`）
  - `to_matrix(rows: dict[str, list[str]], image_names: list[str], labels: list[str]) -> np.ndarray`：形狀 `(N, len(labels)+1)`，最後一欄 = 1 當且僅當該圖沒有 frame
  - `check_same_images(rows: dict, gold_rows: dict, source: str) -> None`：集合不同時 `ValueError`，訊息列出缺少與多出的 image_name 與 `source`
  - `discover_runs(pred_root: str) -> dict[str, dict[int, dict[str, str]]]`：model → seed → split（`dev`/`test`）→ TSV 路徑；跳過 `_` 開頭資料夾；`seed0` 的 seed 為 0

- [ ] **Step 1: 寫失敗的測試**（用 `tmp_path` 自建小檔案，labels 用 `["Economic", "Political", "Morality"]`）

```python
def test_empty_row_and_literal_none_mean_none(tmp_path): ...
    # 檔案內容 "a.jpg\n" 與 "b.jpg\tNone\n"
    assert rows == {"a.jpg": [], "b.jpg": []}
    assert to_matrix(rows, ["a.jpg", "b.jpg"], labels).tolist() == [[0, 0, 0, 1], [0, 0, 0, 1]]

def test_unknown_label_raises_with_path(tmp_path): ...
    # "a.jpg\tEconomy\n" → pytest.raises(ValueError, match="pred.tsv")

def test_duplicate_image_raises(tmp_path): ...

def test_check_same_images_reports_missing_and_extra():
    with pytest.raises(ValueError, match="missing.*c.jpg") : check_same_images({"a.jpg": [], "b.jpg": []}, {"a.jpg": [], "c.jpg": []}, "m/seed13/test.tsv")

def test_discover_runs_skips_underscore_dirs(tmp_path): ...
    # 建立 _gold/test.tsv、image_lr/seed0/test.tsv、e5_fusion_coco_asl/seed13/{dev,test}.tsv
    assert set(runs) == {"image_lr", "e5_fusion_coco_asl"}
    assert runs["e5_fusion_coco_asl"][13]["dev"].endswith("dev.tsv")

def test_write_then_read_roundtrip(tmp_path): ...
```

- [ ] **Step 2: 執行確認失敗**：`$PY -m pytest tests/test_eval_io.py -v`，Expected: FAIL（`ModuleNotFoundError: evaluation.prediction_io`）
- [ ] **Step 3: 實作 `evaluation/prediction_io.py`** 的上述函式。
- [ ] **Step 4: 執行確認通過**：同上指令，Expected: 6 passed
- [ ] **Step 5: Commit**：`git add evaluation tests/test_eval_io.py && git commit -m "feat(eval): prediction TSV io and run discovery"`

---

### Task 4: 計分與 bootstrap（En-Shuo）

**Files:**
- Create: `evaluation/scoring.py`、`evaluation/bootstrap.py`
- Test: `tests/test_eval_scoring.py`

**Interfaces:**
- Consumes: Task 3 的 `read_frames_tsv`、`to_matrix`、`read_labels`、`check_same_images`
- Produces:
  - `run_official_scorer(scorer_path: str, gold_tsv: str, pred_tsv: str, labels_path: str) -> tuple[float, float]`（micro, macro），用 `subprocess.run([sys.executable, scorer, "-g", gold, "-p", pred, "-f", labels, "-o"], check=True)`，取 stdout 最後一行
  - `micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float`、`hamming(y_true, y_pred) -> float`（15 欄矩陣）
  - `per_label_f1(y_true, y_pred, names: list[str]) -> dict[str, float]`（每欄 F1；該欄 gold 與預測皆為 0 時記為 `None` 而不是 1，畫圖時省略）
  - `score_split(pred_paths: list[str], gold_path: str, labels_path: str, scorer_path: str) -> dict`，回傳鍵：`micro`、`macro`、`hamming`（各為每個 seed 的 list）、`per_label_f1`（seed 平均）、`support`（每欄 gold 數）、`matrices`（每個 seed 的預測矩陣）、`y_true`、`image_names`、`mismatches`（官方 micro 與 `micro_f1` 差超過 `1e-6` 的 seed 數；回報值一律用官方的）
  - `group_of(image_name: str) -> str`（去副檔名）；bootstrap 內部的文章順序一律依名稱排序（與 `baselines/src/metrics.py` 的 `_micro_counts_by_group` 相同），這樣單一 seed 時的 CI 會與 Intermediate Report 完全一致
  - `bootstrap_ci(y_true, preds_by_seed: list[np.ndarray], groups: list[str], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]`
  - `paired_ci(y_true, preds_a: list[np.ndarray], preds_b: list[np.ndarray], groups, n_boot=2000, seed=0) -> tuple[float, float]`

- [ ] **Step 1: 寫失敗的測試**

```python
def test_bootstrap_single_seed_matches_intermediate_function():
    # 用固定亂數產生 40 列 15 欄的 y_true / y_pred，groups = 每列一個文章
    # bootstrap_ci(y, [p], g) 必須等於 baselines/src/metrics.py 的 bootstrap_micro_f1(y[:, :14], p[:, :14], g)
    # （直接把該函式的演算法複製進測試做對照，不 import baselines 的 src）

def test_bootstrap_averages_seeds_within_each_resample():
    # 兩個 seed 預測完全相同時，CI 與只給一個 seed 時相同
    assert bootstrap_ci(y, [p, p], g) == bootstrap_ci(y, [p], g)

def test_paired_ci_of_identical_models_is_zero():
    assert paired_ci(y, [p], [p], g) == (0.0, 0.0)

def test_official_score_wins_and_mismatch_is_counted(tmp_path):
    # 用一支假的 scorer 腳本，固定印出 "0.5\t0.4"
    out = score_split([pred], gold, labels, fake_scorer)
    assert out["micro"] == [0.5] and out["mismatches"] == 1

def test_per_label_f1_absent_label_is_none(): ...
```

- [ ] **Step 2: 執行確認失敗**：`$PY -m pytest tests/test_eval_scoring.py -v`，Expected: FAIL
- [ ] **Step 3: 實作 `evaluation/scoring.py` 與 `evaluation/bootstrap.py`**。bootstrap 的做法：對每個 seed 先算每篇文章的 (tp, fp, fn)，用同一組重抽 index 對每個 seed 算 micro，再對 seed 取平均；paired_ci 對 a、b 用同一組 index 相減。
- [ ] **Step 4: 執行確認通過**，Expected: 5 passed
- [ ] **Step 5: Commit**：`git commit -m "feat(eval): official scorer wrapper and seed-averaged article bootstrap"`

---

### Task 5: 評估主程式（En-Shuo）

**Files:**
- Create: `evaluation/run_eval.py`
- Test: `tests/test_eval_run.py`

**Interfaces:**
- Consumes: Task 3、Task 4 全部
- Produces:
  - `TRAINED_MODELS: list[str]` = `e1_text_only` 到 `e8_fusion_gdino_asl` 共 8 個
  - `COMPLEXITY_ORDER: list[str]`（依 Task 2 的順序列出所有固定模型名，用來處理平手）
  - `PRIMARY_COMPARISONS = [("best", "most_frequent"), ("best", "qwen_image_only"), ("e8_fusion_gdino_asl", "e5_fusion_coco_asl"), ("e6_fusion_global_asl", "e2_vision_global")]`
  - `select_best(models: dict, candidates: list[str] | None = None) -> str | None`：只考慮有 dev 分數的模型；以 dev Micro 平均最高者勝，平手取 `COMPLEXITY_ORDER` 較前者；不在 `COMPLEXITY_ORDER` 裡的名稱（例如 `e11_*`）排在所有已列名稱之後
  - `evaluate_all(pred_root: str, scorer_path: str) -> dict`
  - CLI：`$PY -m evaluation.run_eval --pred_root predictions/subtask2 --scorer <scorer> --out results/subtask2_final.json`
- 輸出 JSON 結構（後續所有 Task 都依此讀取）：

```json
{
  "labels": ["Economic", "...", "None"],
  "n_dev": 20, "n_test": 40,
  "models": {
    "<model>": {
      "seeds": [13, 42, 2026],
      "dev":  {"micro_mean": 0.0, "micro_std": 0.0, "macro_mean": 0.0, "hamming_mean": 0.0},
      "test": {"micro_mean": 0.0, "micro_std": 0.0, "micro_ci95": [0.0, 0.0], "macro_mean": 0.0,
               "macro_std": 0.0, "hamming_mean": 0.0, "per_label_f1": {}, "support": {}},
      "scorer_mismatches": 0
    }
  },
  "reference_only": {"random_prior": {}},
  "selection": {"best_model": "", "best_trained_model": "", "rule": "docs/preregistration.md"},
  "primary_comparisons": [{"a": "", "b": "", "diff_mean": 0.0, "ci95": [0.0, 0.0], "status": "ok"}]
}
```

  缺 dev 的模型沒有 `dev` 鍵；缺 test 的模型沒有 `test` 鍵；`status` 為 `ok`、`missing` 或 `same_model`。`reference_only.random_prior` 原封不動地讀自 `{pred_root}/_reference/random_prior.json`（檔案不存在則省略）。

- [ ] **Step 1: 寫失敗的測試**（用 Task 4 的假 scorer，在 `tmp_path` 建 `_gold/{dev,test}.tsv`、`_gold/labels.txt` 與數個模型）

```python
def test_select_best_ignores_models_without_dev(): ...
def test_tie_breaks_to_simpler_model():
    models = {"image_lr": {"dev": {"micro_mean": 0.4}}, "e5_fusion_coco_asl": {"dev": {"micro_mean": 0.4}}}
    assert select_best(models) == "image_lr"
def test_missing_model_marks_comparison_missing(tmp_path):
    # 沒有 e8_fusion_gdino_asl
    assert comparison("e8_fusion_gdino_asl", "e5_fusion_coco_asl")["status"] == "missing"
def test_best_equal_to_reference_is_same_model(tmp_path): ...
def test_partial_seeds_are_reported(tmp_path):
    # e5 只有 seed13 與 seed42
    assert out["models"]["e5_fusion_coco_asl"]["seeds"] == [13, 42]
def test_mismatched_images_stop_evaluation(tmp_path):
    with pytest.raises(ValueError, match="seed13/test.tsv"): ...
```

- [ ] **Step 2: 執行確認失敗**：`$PY -m pytest tests/test_eval_run.py -v`
- [ ] **Step 3: 實作 `evaluation/run_eval.py`**
- [ ] **Step 4: 執行確認通過**，Expected: 6 passed；並跑 `$PY -m pytest tests/test_eval_*.py -q` 全部通過
- [ ] **Step 5: Commit**：`git commit -m "feat(eval): unified evaluation with pre-registered selection"`

---

### Task 6: 匯出現有 baseline（En-Shuo）

**Files:**
- Modify: `baselines/scripts/run_experiments.py`（`run_subtask2` 也保存 dev 預測；`run_vlm` 已分 dev/test 存檔，不用改）
- Create: `baselines/scripts/export_predictions.py`
- Create（產出）：`predictions/subtask2/_gold/{labels.txt,dev.tsv,test.tsv,train.tsv,train_text_frames.tsv,dev_text_frames.tsv,test_text_frames.tsv}`、`predictions/subtask2/_reference/random_prior.json`、9 個 baseline 模型的 `seed0/{dev,test}.tsv`、`results/subtask2_final.json`

**Interfaces:**
- Consumes: `baselines/src/data_loader.py` 的 `load_official_data`、`stack_labels`；`baselines/src/official_scorer.py` 的 `vector_to_frames`；Task 3 的 `write_frames_tsv` 不能 import（baselines 在自己的資料夾執行），所以 export 腳本自己寫 TSV，格式同 Global Constraints。
- Produces: `_gold/train.tsv` 只含有文字 frame 的訓練文章（`has_text_frames`）；`*_text_frames.tsv` 的每列是 `image_name` 加上該文章所有段落 frame 的聯集。

- [ ] **Step 1: 修改 `run_subtask2`**：`add()` 同時存 dev 預測到 `results["dev_predictions"][name]`（threshold 模型用同一個 threshold；Most frequent 與 Copy text frames 直接對 dev 產生）。Random 不存預測，只保留現有的期望值指標。
- [ ] **Step 2: 重跑**（LR 在快取特徵上，很快；VLM 用 `results/runs` 的快取，不會重新呼叫 Ollama）

Run: `cd baselines && $PY scripts/run_experiments.py subtask2 vlm_dev vlm_test`
Expected: 印出的 test micro 與 Intermediate Report 第 4 節完全相同。

- [ ] **Step 3: 寫 `export_predictions.py`**：`main(out_root: str = "../predictions/subtask2")`，名稱對照：`Most frequent label`→`most_frequent`、`Copy text frames (2b)`→`copy_text_frames_2b`、`Image LR (SigLIP)`→`image_lr`、`Text LR (DeBERTa-v3)`→`text_lr`、`Late fusion (image + text)`→`late_fusion_2a`、`Late fusion + text frames (2b)`→`late_fusion_2b`、VLM `image_only`/`subtask_2a`/`subtask_2b`→`qwen_image_only`/`qwen_2a`/`qwen_2b`（讀 `vlm_dev_metrics.json` 與 `vlm_test_metrics.json` 的 `variants[v]["predictions"]`，用 `doc_order` 對回 image_name）。Copy text frames 沒有 dev 選擇問題，照樣輸出 dev。Random 寫成 `_reference/random_prior.json`，內容為 `subtask2_metrics.json` 中 `Random (training prior)` 的指標原樣複製。
- [ ] **Step 4: 執行匯出並評估**

Run: `$PY scripts/export_predictions.py && cd .. && $PY -m evaluation.run_eval --pred_root predictions/subtask2 --scorer baselines/data/starter-kit/starter-kit/subtask2a_2b/scorer-subtask-2.py --out results/subtask2_final.json`
Expected（test `micro_mean`，取 3 位小數，全部要與 Intermediate Report 相同）：`most_frequent` 0.295、`copy_text_frames_2b` 0.238、`image_lr` 0.353、`text_lr` 0.152、`late_fusion_2a` 0.367、`late_fusion_2b` 0.291、`qwen_image_only` 0.417、`qwen_2a` 0.333、`qwen_2b` 0.340；`micro_ci95` 也要相同：`most_frequent` [0.169, 0.422]、`image_lr` [0.269, 0.444]、`qwen_image_only` [0.292, 0.553]；`scorer_mismatches` 全為 0。任何一個不同就停下來查，不要調整。

- [ ] **Step 5: Commit**：`git add baselines/scripts predictions results/subtask2_final.json && git commit -m "feat: export baseline predictions and reproduce intermediate scores"`

---

### Task L1: Seed 與預測檔輸出（Leo）

**Files:**
- Create: `src/reproducibility.py`
- Modify: `scripts/04_run_ablations.py`（`train_unimodal_or_ablation`、`run_ablation_suite`、`main`）；刪除 `format_ablation_markdown_table` 中手打的 Intermediate Report 數字列與 `reports/ablation_results.md`；README 的結果表格改成「見 `results/subtask2_final.json`」
- Test: `tests/test_train_pipeline.py`（新增測試）

**Interfaces:**
- Produces:
  - `set_seed(seed: int) -> None`：設定 `random`、`numpy`、`torch`（含 cuda、mps）的 seed；DataLoader 的 shuffle 用 `torch.Generator().manual_seed(seed)`
  - `EXPERIMENTS: dict[str, tuple[str, str, str]]`：run name →（model_type, loss_type, regions），regions 為 `"coco"` 或 `"gdino"`：
    `e1_text_only: (text_only, asl, coco)`、`e2_vision_global: (vision_global, asl, coco)`、`e3_vision_region_coco: (vision_region, asl, coco)`、`e4_fusion_coco_bce: (fusion_bce, bce, coco)`、`e5_fusion_coco_asl: (fusion_asl, asl, coco)`、`e6_fusion_global_asl: (fusion_global, asl, coco)`、`e7_vision_region_gdino: (vision_region, asl, gdino)`、`e8_fusion_gdino_asl: (fusion_asl, asl, gdino)`
  - `train_unimodal_or_ablation(..., seed: int = 13, pred_root: Optional[str] = None, run_name: Optional[str] = None)`：`pred_root` 不為 None 時，用 dev 選出的 checkpoint 與 threshold，寫出 `{pred_root}/{run_name}/seed{seed}/dev.tsv`、`test.tsv`（用現有 `export_predictions_tsv`）以及 `dev.probs.json`、`test.probs.json`，內容為 `{"image_names": [...], "classes": [...14], "probs": [[...14]], "threshold": t}`
  - CLI：`python scripts/04_run_ablations.py --experiments e1_text_only ... --seeds 13 42 2026 --pred_root predictions/subtask2 --feature_dir_coco cache/features --feature_dir_gdino cache/features_gdino`

- [ ] **Step 1: 寫失敗的測試**（沿用 `tests/test_train_pipeline.py` 現有的合成快取特徵 fixture）

```python
def test_same_seed_gives_identical_probs(tmp_path):
    a = run("e2_vision_global", seed=13); b = run("e2_vision_global", seed=13)
    assert a["test"]["probs"] == b["test"]["probs"]
def test_different_seed_changes_probs(tmp_path): ...
def test_writes_dev_and_test_files_for_every_split_image(tmp_path):
    # 檔案存在，且 TSV 的 image_name 集合 = 該 split 的樣本集合
def test_probs_json_has_class_names_and_threshold(tmp_path): ...
```

- [ ] **Step 2: 執行確認失敗**：`pytest tests/test_train_pipeline.py -v`
- [ ] **Step 3: 實作**上述介面
- [ ] **Step 4: 執行確認通過**，並跑 `pytest tests -q` 全部通過
- [ ] **Step 5: Commit**：`git commit -m "feat(train): seeded runs and per-seed prediction export"`

---

### Task L2: E6 不加區域的 fusion（Leo）

**Files:**
- Modify: `scripts/04_run_ablations.py`（新增 model_type `fusion_global`）
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: model_type `fusion_global` = `CrossModalFramingModel`，`_forward` 傳入 `v_tokens[:, :1, :]`（只有整張圖 token）

- [ ] **Step 1: 寫失敗的測試**

```python
def test_fusion_global_ignores_region_tokens():
    # 同一個模型、同一批輸入，只把 visual_tokens[:, 1:, :] 換成隨機值
    assert torch.allclose(forward_fusion_global(v1, t, m), forward_fusion_global(v2, t, m))
```

- [ ] **Step 2: 執行確認失敗**；**Step 3: 實作**；**Step 4: 執行確認通過**
- [ ] **Step 5: Commit**：`git commit -m "feat(models): fusion_global ablation without region tokens"`

---

### Task L3: Grounding-DINO 開放詞彙區域（Leo；抽取可在 En-Shuo 的 GPU 上跑）

**Files:**
- Create: `src/frame_vocabulary.py`、`scripts/06_region_coverage.py`
- Modify: `src/entity_extractor.py`、`scripts/01_extract_regions.py`、`scripts/02_extract_features.py`（只需確認 `--crop_dir`、`--output_dir` 可指到新資料夾）
- Test: `tests/test_entity_extractor.py`

**Interfaces:**
- Produces:
  - `FRAME_VOCABULARY: dict[str, list[str]]`、`build_prompt_text() -> str`（全部片語小寫，以 `". "` 連接並以 `"."` 結尾）
  - `fill_with_fallback(boxes: list[list[int]], w: int, h: int, top_k: int) -> list[list[int]]`（把現有的固定裁切邏輯抽出來，COCO 與 Grounding-DINO 共用）
  - `SalientEntityExtractor.extract_regions_with_stats(img, top_k=3, conf_thresh=0.3) -> tuple[list[Image.Image], list[list[int]], int]`，第三個值為偵測器實際提供的框數（0–3）；原本的 `extract_regions` 保留並改為呼叫它
  - `OpenVocabEntityExtractor(device=None, model_id="IDEA-Research/grounding-dino-tiny")`，同名方法 `extract_regions_with_stats(img, top_k=3, box_threshold=0.35, text_threshold=0.25)`，用 transformers 的 `AutoProcessor`、`AutoModelForZeroShotObjectDetection` 與 `post_process_grounded_object_detection`（參數名依 transformers 5.22 的版本）
  - `01_extract_regions.py --detector {coco,gdino}`，輸出到 `cache/crops`（coco）或 `cache/crops_gdino`，`metadata.json` 每張圖多一個 `n_detected`
  - `06_region_coverage.py` → `results/region_coverage.json`：`{detector: {"n_images", "all_detected", "partial", "all_fallback", "mean_detected"}}`

- [ ] **Step 1: 寫 `src/frame_vocabulary.py` 並先 commit**（必須在抽取任何 Grounding-DINO 區域之前；只依 Card et al. 2015 的 frame 定義，不看 dev/test 圖片）。草案如下，Leo 可修改，但須遵守同樣規則：

| Frame | 片語 |
|---|---|
| Economic | money, price sign, stock market display |
| Capacity_and_resources | construction site, empty store shelves, fuel pump |
| Morality | religious symbol, place of worship |
| Fairness_and_equality | equal rights sign, wheelchair |
| Legality_Constitutionality_and_jurisprudence | courtroom, judge, gavel |
| Policy_prescription_and_evaluation | official document, politician signing document |
| Crime_and_punishment | police officer, handcuffs, prison cell |
| Security_and_defense | soldier, military vehicle, firearm |
| Health_and_safety | doctor, hospital bed, face mask |
| Quality_of_life | family, children playing, damaged home |
| Cultural_identity | traditional clothing, cultural festival |
| Public_opinion | protest sign, crowd of protesters |
| Political | politician at podium, parliament building, ballot box |
| External_regulation_and_reputation | flags of several countries, diplomatic handshake |

```bash
git add src/frame_vocabulary.py && git commit -m "feat(regions): frame-derived open-vocabulary prompt (pre-registered)"
```

- [ ] **Step 2: 寫失敗的測試**

```python
def test_vocabulary_covers_all_14_frames(): assert set(FRAME_VOCABULARY) == set(labels_from_labels_txt)
def test_prompt_text_format(): assert build_prompt_text().endswith(".") and build_prompt_text() == build_prompt_text().lower()
def test_fill_with_fallback_matches_previous_crops():
    # 對 0、1、3 個偵測框，fill_with_fallback 的結果與重構前的 extract_regions 相同
def test_stats_report_number_of_detected_boxes():  # 用 mock 偵測器回傳 1 個高分框
    _, boxes, n = extractor.extract_regions_with_stats(img); assert n == 1 and len(boxes) == 3
def test_gdino_features_share_global_and_text_tokens(tmp_path):
    # 同一 sample 的 cache/features 與 cache/features_gdino：
    assert torch.allclose(a["visual_tokens"][0], b["visual_tokens"][0]) and torch.equal(a["text_tokens"], b["text_tokens"])
```

最後一個測試另外寫成 `06_region_coverage.py --check_features` 的檢查，在真實快取上跑；不一致就以非零代碼結束。

- [ ] **Step 3: 執行確認失敗**；**Step 4: 實作**；**Step 5: 執行確認通過**（`pytest tests -q`）
- [ ] **Step 6: 抽取區域與特徵**（Mac 跑不動就在 En-Shuo 的 CUDA 機器上跑）

```bash
python scripts/01_extract_regions.py --detector gdino --output_dir cache/crops_gdino
python scripts/02_extract_features.py --crop_dir cache/crops_gdino --output_dir cache/features_gdino
python scripts/06_region_coverage.py --check_features
```

Expected：`results/region_coverage.json` 有 `coco` 與 `gdino` 兩組；`--check_features` 結束代碼 0。（COCO 的 `n_detected` 需要以 `--detector coco` 重新抽一次 metadata；裁切結果應與既有快取相同。）

- [ ] **Step 7: Commit**：`git add src scripts tests results/region_coverage.json && git commit -m "feat(regions): Grounding-DINO extractor and coverage report"`

---

### Task L4: 跑 E1–E8 與 E11（Leo）

**Files:**
- Modify: `src/dataset.py`（`FramingDataset(..., include_batch1: bool = True)`；False 時訓練集只含第二批的 37 篇）、`src/cached_dataset.py`（透傳參數）、`scripts/04_run_ablations.py`（`train_unimodal_or_ablation(..., include_batch1: bool = True)` 與 CLI `--no_batch1`）
- Create（產出）：`predictions/subtask2/e*/seed*/`

- [ ] **Step 1: 確認 Task 2 已 commit**（`git log --oneline docs/preregistration.md` 有紀錄），才開始跑。
- [ ] **Step 2: P0**：`python scripts/04_run_ablations.py --experiments e1_text_only e2_vision_global e3_vision_region_coco e4_fusion_coco_bce e5_fusion_coco_asl e6_fusion_global_asl --seeds 13 42 2026 --pred_root predictions/subtask2`。Expected: 18 個 `seed*/` 資料夾，各有 4 個檔案；log 印出的訓練集張數（應為 437）記進 `predictions/subtask2/RUNS.md`。
- [ ] **Step 3: P1**（Task L3 完成後）：同上指令跑 `e7_vision_region_gdino e8_fusion_gdino_asl`。
- [ ] **Step 4: P3 E11**：先由 En-Shuo 跑一次 `run_eval` 取得 `selection.best_trained_model`，再以 `--no_batch1` 跑該設定，run name `e11_<best_trained_model>_no_batch1`，3 個 seed。spec 原本的變體 (b)「只用英文」就是 Leo pipeline 的預設（437 張英文圖），所以 E11 只比較預設與移除 `initial-training-data` 兩種。加一個測試：`test_include_batch1_false_keeps_only_second_batch_train`（fixture 資料下訓練集只剩第二批的 id）。
- [ ] **Step 5: 每完成一批就 commit 並 push 到 `final/region-experiments`**：`git add predictions && git commit -m "feat: E1-E6 predictions (3 seeds)"`

---

### Task 7: E10 Qwen prompt v2（En-Shuo）

**Files:**
- Modify: `baselines/src/models/vlm_pipeline.py`、`baselines/scripts/run_experiments.py`、`baselines/scripts/export_predictions.py`
- Test: `baselines/tests/test_pipeline.py`

**Interfaces:**
- Produces: `VARIANTS` 加入 `"image_only_v2"`；`parse_frames` 把 `"None"` 視為合法並回傳空清單；匯出名稱 `image_only_v2` → `qwen_image_only_v2`

- [ ] **Step 1: 寫失敗的測試**（unittest，沿用該檔風格）

```python
def test_v2_prompt_offers_none(self):
    self.assertIn('{"frames": ["None"]}', build_prompt(doc, "image_only_v2"))
    self.assertNotIn("Article text", build_prompt(doc, "image_only_v2"))
def test_parse_none_is_valid_empty(self):
    self.assertEqual(parse_frames('{"frames": ["None"]}'), ([], True))
```

- [ ] **Step 2: 執行確認失敗**：`cd baselines && $PY -m unittest tests.test_pipeline -v`
- [ ] **Step 3: 實作**。v2 與 `image_only` 相同，只把最後一段換成以下文字：

```
None: the image shows no clear visual evidence for any category above (for example, a generic portrait, a logo, or scenery unrelated to the issue).
Select the framings conveyed by the IMAGE itself. Choose only framings with clear visual evidence. If no category has clear visual evidence, answer {"frames": ["None"]}. Answer with JSON only, in the form {"frames": ["Category_name"]}.
```

- [ ] **Step 4: 執行確認通過**
- [ ] **Step 5: 只在 dev 上跑**：`$PY scripts/run_experiments.py vlm_dev`（只會查詢新的 variant）。若 parse failure > 2，只修改措辭一次（記為 v3，同樣只在 dev 跑），取 dev Micro 較高的版本。
- [ ] **Step 6: 選定後跑 test、匯出、評估**：`$PY scripts/run_experiments.py vlm_test && $PY scripts/export_predictions.py && cd .. && $PY -m evaluation.run_eval ...`（參數同 Task 6）。Expected：`qwen_image_only_v2` 出現在 JSON，dev 與 test 都有。
- [ ] **Step 7: Commit**：`git commit -m "feat(vlm): image-only prompt v2 with explicit None"`

---

### Task 7b: E12 Qwen 文字長度（En-Shuo，P2）

回答老師「加的是標題還是全文」。現有 `subtask_2a` 讀「標題 + 前 1500 字元」，E12 補上「只給標題」與「全文不截斷」，三者只差文字長度。依 spec 3.4 的例外紀錄，E12 不做任何選擇：兩個新設定都跑 dev 與 test，全部報告，不修改措辭。**在 Task 7 完成後才做**：`vlm_test` 會查詢 `VARIANTS` 裡的每個 variant，若 E10 的 v2 還沒在 dev 上選定，就會提前跑到 test。

**Files:**
- Modify: `baselines/src/models/vlm_pipeline.py`、`baselines/scripts/run_experiments.py`（`run_vlm` 已依 `VARIANTS` 迴圈，只需確認新 variant 會被查詢）、`baselines/scripts/export_predictions.py`
- Create: `baselines/tests/fixtures/prompts_before_e12.json`
- Test: `baselines/tests/test_pipeline.py`

**Interfaces:**
- Produces：
  - `VARIANTS` 加入 `"subtask_2a_title"`、`"subtask_2a_fulltext"`；匯出名稱分別為 `qwen_2a_title`、`qwen_2a_fulltext`
  - `build_prompt(document, variant)`：`subtask_2a_title` 放 `"Article title (it may be in any language):\n" + <文章第一個非空行>`；`subtask_2a_fulltext` 放 `"Article text (it may be in any language):\n" + document["text"]`（不截斷）；其餘段落（frame 定義、最後的指示）與 `subtask_2a` 完全相同
  - `query_vlm` 回傳的 dict 加上 `prompt_eval_count`（取自 Ollama 回應的同名欄位，沒有時為 `None`），隨快取一起存檔

- [ ] **Step 1: 先存下現有 prompt 當基準**：用**本機原始** `C:\UC-main\nlp_project` 的 `build_prompt`，對一份固定的假文件（`{"text": "A Title\n\nFirst paragraph.\n\n" + "x" * 2000 + " END", "text_frames": ["Political"]}`）產生 `image_only`、`subtask_2a`、`subtask_2b` 三個 prompt，寫入 `prompts_before_e12.json`。現有 Qwen 預測是用這三個 prompt 產生的，之後不能改動它們。
- [ ] **Step 2: 寫失敗的測試**（unittest，沿用該檔風格）

```python
def test_existing_prompts_unchanged(self):
    saved = json.load(open(FIXTURE, encoding="utf-8"))
    for variant in ("image_only", "subtask_2a", "subtask_2b"):
        self.assertEqual(build_prompt(DOC, variant), saved[variant])
def test_title_variant_has_title_only(self):
    p = build_prompt(DOC, "subtask_2a_title")
    self.assertIn("A Title", p)
    self.assertNotIn("First paragraph.", p)
def test_fulltext_variant_is_not_truncated(self):
    self.assertIn(" END", build_prompt(DOC, "subtask_2a_fulltext"))
    self.assertNotIn(" END", build_prompt(DOC, "subtask_2a"))
```

- [ ] **Step 3: 執行確認失敗**：`cd baselines && $PY -m unittest tests.test_pipeline -v`；Expected：後兩個測試 FAIL，第一個 PASS
- [ ] **Step 4: 實作**，執行確認全部通過
- [ ] **Step 5: 跑 dev 與 test**：`$PY scripts/run_experiments.py vlm_dev vlm_test`（已快取的舊 variant 不會重新查詢）。Expected：兩個新 variant 的 dev 20 張、test 40 張都有預測；印出 parse failure 數（如實報告，不修改措辭重跑）；`subtask_2a_fulltext` 的 `prompt_eval_count` 最大值 < 8192（`num_ctx`）。若達到 8192，代表 Ollama 可能截斷了文字：只對 `subtask_2a_fulltext` 把 `num_ctx` 改為 16384 並重跑這個 variant，在 `docs/notes/` 記錄。
- [ ] **Step 6: 匯出並評估**：`$PY scripts/export_predictions.py && cd .. && $PY -m evaluation.run_eval ...`（參數同 Task 6）。Expected：`qwen_2a_title` 與 `qwen_2a_fulltext` 出現在 JSON，dev 與 test 都有；舊的 9 個 baseline 分數不變。
- [ ] **Step 7: Commit**：`git commit -m "feat(vlm): E12 text-length variants (title only, full text)"`

---

### Task 8: E9 Qwen 與最佳訓練模型的組合（En-Shuo）

**Files:**
- Create: `ensemble/__init__.py`、`ensemble/combine.py`
- Test: `tests/test_ensemble.py`

**Interfaces:**
- Consumes: Task 3 的 `read_labels`、`read_frames_tsv`、`to_matrix`、`write_frames_tsv`；Task 4 的 `micro_f1`；`results/subtask2_final.json` 的 `selection.best_trained_model`；Task L1 的 `{dev,test}.probs.json`
- Produces:
  - `load_probs(path: str, labels: list[str]) -> tuple[list[str], np.ndarray]`：依 `classes` 名稱把欄位重新排成 `labels` 的順序
  - `combine(qwen: np.ndarray, probs: np.ndarray, alpha: float) -> np.ndarray`（兩者皆為 14 欄）
  - `fit_on_dev(qwen_dev, probs_dev, y_dev) -> tuple[float, float, float]`（alpha, threshold, dev_micro），α 與 threshold 範圍見 Global Constraints
  - CLI：`$PY -m ensemble.combine --pred_root predictions/subtask2`，對最佳訓練模型的每個 seed 寫出 `combo_qwen_best/seed<k>/{dev,test}.tsv` 與 `combo_qwen_best/selection.json`（每個 seed 的 α、threshold、dev micro、來源模型名）

- [ ] **Step 1: 寫失敗的測試**

```python
def test_load_probs_reorders_by_class_name(tmp_path):
    # classes 順序為 ["Political", "Economic", ...]，結果欄位要依 labels.txt 順序
def test_combine_alpha_one_returns_qwen():
    assert np.array_equal(combine(q, p, 1.0), q)
def test_fit_on_dev_only_uses_allowed_grid():
    a, t, _ = fit_on_dev(q, p, y); assert a in (0.25, 0.5, 0.75) and round(t, 2) in [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
def test_no_label_above_threshold_means_none(tmp_path): ...
```

- [ ] **Step 2–4:** 執行確認失敗 → 實作 → 執行確認通過（`$PY -m pytest tests/test_ensemble.py -v`）
- [ ] **Step 5: 跑組合並評估**（L4 的 P0 完成後；P1 完成後若最佳訓練模型改變，重跑一次）：`$PY -m ensemble.combine --pred_root predictions/subtask2 && $PY -m evaluation.run_eval ...`。Expected：`combo_qwen_best` 有 3 個 seed。
- [ ] **Step 6: Commit**：`git commit -m "feat(ensemble): dev-tuned Qwen + trained model combination"`

---

### Task 9: C1 圖文 frame 差異（En-Shuo）

**Files:**
- Create: `analysis/__init__.py`、`analysis/lock.py`、`analysis/image_text_frames.py`
- Test: `tests/test_analysis.py`

**Interfaces:**
- Produces:
  - `analysis/lock.py`：`LOCK_PATH = "results/LOCKED"`、`is_locked() -> bool`、`require_lock() -> None`（未鎖定時 `SystemExit` 並說明原因）、CLI `$PY -m analysis.lock --create`（寫入目前 commit hash 與時間）
  - `frame_relation(image_frames: set[str], text_frames: set[str]) -> str`，回傳：
    - `aligned`：image 非空且 image ⊆ text
    - `complementary`：image ∩ text 非空且 image ⊄ text
    - `divergent`：image 非空且 image ∩ text 為空
    - `image_none`：image 為空
  - `conditional_matrix(pairs: list[tuple[set, set]], labels: list[str]) -> np.ndarray`：14×14，第 i 列第 j 欄 = P(圖片有 frame j | 文字有 frame i)；文字從未出現 frame i 時該列為 NaN
  - CLI：`$PY -m analysis.image_text_frames --pred_root predictions/subtask2 [--include_test]`，讀 `_gold/{split}.tsv` 與 `_gold/{split}_text_frames.tsv`，輸出 `results/c1_image_text.json`（各類別數量與比例、矩陣、用了哪些 split）；`--include_test` 會先呼叫 `require_lock()`

- [ ] **Step 1: 寫失敗的測試**

```python
def test_frame_relation_categories():
    assert frame_relation({"Political"}, {"Political", "Economic"}) == "aligned"
    assert frame_relation({"Political", "Morality"}, {"Political"}) == "complementary"
    assert frame_relation({"Morality"}, {"Political"}) == "divergent"
    assert frame_relation(set(), {"Political"}) == "image_none"
def test_conditional_matrix_row_without_text_frame_is_nan(): ...
def test_include_test_requires_lock(tmp_path, monkeypatch):
    with pytest.raises(SystemExit): main(["--include_test", ...])
```

- [ ] **Step 2–4:** 執行確認失敗 → 實作 → 執行確認通過（`$PY -m pytest tests/test_analysis.py -v`）
- [ ] **Step 5: 用 train + dev 跑一次**：`$PY -m analysis.image_text_frames --pred_root predictions/subtask2`，Expected：JSON 的 `splits` 為 `["train", "dev"]`。
- [ ] **Step 6: Commit**：`git commit -m "feat(analysis): C1 image-vs-text frame relations with test lock"`

---

### Task 10: C2 VLM 加文字的影響、C4 案例表（En-Shuo）

**Files:**
- Create: `analysis/vlm_text_overlap.py`、`analysis/case_table.py`
- Test: `tests/test_analysis.py`

**Interfaces:**
- Consumes: Task 3 的讀檔函式、Task 9 的 `require_lock`
- Produces:
  - `overlap_stats(pred_rows: dict[str, list[str]], text_rows: dict[str, list[str]]) -> dict`，鍵：`mean_n_labels`、`frac_pred_in_text`（預測標籤中出現在文字 frame 的比例）、`jaccard_mean`（預測與文字 frame 的平均 Jaccard，兩者皆空時記為 1）
  - `swap_stats(base_rows, variant_rows, gold_rows, text_rows) -> dict`（四者皆為 `dict[str, list[str]]`，以圖片名對齊），鍵：`added`、`added_correct`、`added_in_text`、`removed`、`removed_correct`。「新增」是 variant 有而 base 沒有的標籤，「拿掉」反之；`None` 不算標籤
  - `text_coverage(texts: list[str], max_chars: int) -> dict`，鍵：`median_chars`、`frac_truncated`（長度 > max_chars 的比例）、`median_share_seen`（min(max_chars / 長度, 1) 的中位數）
  - CLI：`$PY -m analysis.vlm_text_overlap --pred_root predictions/subtask2 [--include_test]` → `results/c2_vlm_text.json`，比較 `qwen_image_only`、`qwen_2a_title`、`qwen_2a`、`qwen_2a_fulltext`、`qwen_2b`（有 `qwen_image_only_v2` 時一併列入）；每個 variant 都列 `overlap_stats` 與相對 `qwen_image_only` 的 `swap_stats`；另列 dev／test 文章的 `text_coverage(max_chars=1500)`。文字 frame 讀 `_gold/{dev,test}_text_frames.tsv`，文章長度讀 `baselines` 的資料載入器
  - CLI：`$PY -m analysis.case_table --pred_root predictions/subtask2 --models <best> qwen_image_only e2_vision_global` → `results/case_table.md`（每張 dev/test 圖一列：gold、各模型預測、是否完全答對）；test 部分需要 `require_lock()`

- [ ] **Step 1: 寫失敗的測試**

```python
def test_overlap_stats_counts_copied_text_frames():
    s = overlap_stats({"a.jpg": ["Political", "Morality"]}, {"a.jpg": ["Political"]})
    assert s["mean_n_labels"] == 2 and s["frac_pred_in_text"] == 0.5 and s["jaccard_mean"] == 0.5
def test_swap_stats_counts_added_and_removed():
    s = swap_stats({"a.jpg": ["Political"]}, {"a.jpg": ["Health_and_safety", "Economic"]},
                   {"a.jpg": ["Political", "Economic"]}, {"a.jpg": ["Health_and_safety"]})
    assert s == {"added": 2, "added_correct": 1, "added_in_text": 1, "removed": 1, "removed_correct": 1}
def test_text_coverage():
    c = text_coverage(["a" * 1000, "a" * 3000, "a" * 6000], 1500)
    assert c["median_chars"] == 3000 and abs(c["frac_truncated"] - 2 / 3) < 1e-9 and c["median_share_seen"] == 0.5
def test_case_table_test_rows_require_lock(): ...
```

- [ ] **Step 2–4:** 執行確認失敗 → 實作 → 執行確認通過（`$PY -m pytest tests/test_analysis.py -v`）
- [ ] **Step 5: 先只跑 dev**，Expected：`results/c2_vlm_text.json` 的 `splits` 為 `["dev"]`
- [ ] **Step 6: Commit**：`git commit -m "feat(analysis): C2 VLM text overlap and C4 case table"`

---

### Task 11: Subtask 1 ASL seed 42 診斷（En-Shuo，P3）

**Files:**
- Create: `docs/notes/asl_seed42.md`

- [ ] **Step 1:** 讀 `baselines/results/runs/st1_deberta_asl_{13,42,2026}.json` 的逐 epoch 紀錄，比較 loss、dev 分數與預測標籤數；檢查 seed 42 是否從某個 epoch 起全部預測 None（dev 0.245 = Always None 的分數）。
- [ ] **Step 2:** 寫 `docs/notes/asl_seed42.md`：觀察到的現象、可能原因（例如 ASL 的負例 focusing 讓模型塌縮成全部輸出 0），以及結論：找到原因，或「未解決」加上證據。不重跑訓練。
- [ ] **Step 3: Commit**：`git commit -m "docs: ASL seed 42 failure analysis"`

---

### Task 12: 鎖定數字與產生圖表（En-Shuo，11/20–11/24）

**Files:**
- Create: `analysis/figures.py`
- Test: `tests/test_figures.py`

**Interfaces:**
- Consumes: `results/subtask2_final.json`（Task 5 結構）、`results/c1_image_text.json`、`results/c2_vlm_text.json`、`results/region_coverage.json`
- Produces（全部輸出到 `results/figures/`，PNG 與 PDF 各一份，配色常數複製自 `baselines/src/visualize.py`）：
  - `plot_main_results(final: dict, models: list[str], out_path: str) -> None`：test Micro 的點加 95% CI 橫線，依 dev 選出的最佳模型以強調色標示
  - `plot_ablation(final: dict, out_path: str) -> None`：E2、E3、E5、E6、E7、E8 的 dev 與 test Micro（mean ± std）
  - `plot_per_label(final: dict, models: list[str], out_path: str) -> None`：省略 support 為 0 的欄
  - `plot_c1_heatmap(c1: dict, out_path: str) -> None`、`plot_region_coverage(cov: dict, out_path: str) -> None`
  - `plot_c2(c2: dict, final: dict, out_path: str) -> None`：左圖為 Qwen 依文字長度排列（image only → 標題 → 前 1500 字元 → 全文 → + gold frames）的 dev 與 test Micro（附 CI）；右圖為各 variant 相對 image only 新增與拿掉的標籤數，標出其中正確、來自文字 frame 的部分

- [ ] **Step 1: 寫失敗的測試**：每個繪圖函式用一份小型 fixture JSON（放在 `tests/fixtures/final_small.json`），呼叫後檔案存在且大小 > 0；`plot_main_results` 遇到沒有 `test` 鍵的模型時跳過它而不是崩潰。
- [ ] **Step 2–4:** 執行確認失敗 → 實作 → 執行確認通過（`$PY -m pytest tests/test_figures.py -v`）
- [ ] **Step 5: 11/20 鎖定**：確認所有該跑的實驗都已在 `predictions/`，最後一次執行 `run_eval`，然後

```bash
$PY -m analysis.lock --create && git add results && git commit -m "chore: lock results for final report"
```

- [ ] **Step 6: 鎖定後跑 test 分析**：`$PY -m analysis.image_text_frames --pred_root predictions/subtask2 --include_test`、`$PY -m analysis.vlm_text_overlap --pred_root predictions/subtask2 --include_test`、`case_table`，再產生所有圖：`$PY -m analysis.figures`。
- [ ] **Step 7: Commit**：`git add analysis results tests && git commit -m "feat(analysis): final figures from locked results"`

---

### Task L5: C5 區域分析與架構圖（Leo）

**Files:**
- Modify: `scripts/05_visualize_attention.py`（只從 dev/test 選圖，加 `--split {dev,test}`）
- Create（產出）：`results/figures/c5_attention_*.png`、`results/figures/c5_regions_coco_vs_gdino.png`、`slides/figures/architecture.svg`

- [ ] **Step 1:** 寫測試 `test_visualize_rejects_train_split`：`--split train` 以非零代碼結束。實作並確認通過。
- [ ] **Step 2:** 產生 dev/test 各 2–3 張 attention 熱圖（E5 與 E8 各一份）、COCO 與 Grounding-DINO 偵測框並排對照圖；圖說註明「attention weights are not explanations」。
- [ ] **Step 3:** 畫架構圖：Faster R-CNN 或 Grounding-DINO → 3 個區域；SigLIP（整張圖 + 區域）→ visual tokens；DeBERTa-v3-base → text tokens；cross-attention（text = query）→ masked mean pooling → MLP → 14 logits → dev threshold → None 規則；loss 為 ASL。元件名稱必須與程式碼一致。
- [ ] **Step 4: Commit**：`git commit -m "feat(explain): C5 dev/test attention and architecture diagram"`

---

### Task 13: Final Report（兩人，12/2 前）

**Files:**
- Create: `report/final_report.md`、`report/build_tables.py`、`report/check_numbers.py`、`report/generate_report.py`
- Test: `tests/test_report_tools.py`

**Interfaces:**
- Produces:
  - `build_tables.py`：把 `final_report.md` 裡的 `<!-- table:main -->`、`<!-- table:ablation -->`、`<!-- table:resources -->` 換成由 `results/subtask2_final.json` 產生的 Markdown 表格，輸出 `report/final_report.built.md`
  - `check_numbers.py`：`find_unbacked_numbers(md_text: str, sources: list[dict], allow: set[str]) -> list[str]`，找出正文中所有 0.xxx 形式的數字，凡是不等於任何來源 JSON 中的值（四捨五入到 3 位）、也不在 allow 清單的都列出；有任何一個就以非零代碼結束。來源固定為：`results/subtask2_final.json`、`results/c1_image_text.json`、`results/c2_vlm_text.json`、`results/region_coverage.json`、`baselines/results/subtask1_metrics.json`（Subtask 1 小節）、`baselines/results/leakage_analysis.json`（C6）
  - `generate_report.py`：複製自 `baselines/scripts/generate_doc.py`，常數改為 `MD_PATH=report/final_report.built.md`、輸出 `report/FrameVision_Final_Report.{docx,pdf}`、`MAX_PAGES = 10`、`ZIP_PATH=report/FrameVision_Final_Code.zip`；zip 內容為 `src scripts tests evaluation ensemble analysis report baselines/src baselines/scripts baselines/tests docs predictions`、`results/*.json`、`README.md`、`requirements*.txt`；保留 ≤10 MB 檢查

- [ ] **Step 1: 寫失敗的測試**

```python
def test_check_numbers_flags_typed_number():
    assert find_unbacked_numbers("Our model gets 0.512.", [{"x": 0.417}], set()) == ["0.512"]
def test_check_numbers_accepts_value_from_any_source():
    assert find_unbacked_numbers("Qwen 0.417, leakage 0.203.", [{"m": {"t": 0.4171}}, {"by_article": [0.2031]}], set()) == []
def test_build_tables_replaces_markers(tmp_path): ...
```

- [ ] **Step 2–4:** 執行確認失敗 → 實作 → 執行確認通過
- [ ] **Step 5: 撰寫 `final_report.md`**（英文），章節與頁數依 spec 第 8 節；分工：
  - En-Shuo：1 Introduction、3 Task and Data、4 Related Work、5 Resources、7 Experimental Setup、8 Results、9 Analysis 的 C1–C4、C6、11 Contributions
  - Leo：2 Our Key Ideas 的 K2、6 Methods 中 region-aware 模型與開放詞彙區域、9 Analysis 的 C5
  - 兩人：2 Our Key Ideas 的 K1 與 K3（依鎖定後的結果定稿）、10 Limitations, Lessons, Conclusion
  - 必寫事項：
    - Subtask 1 小節（沿用 `baselines/results/subtask1_metrics.json`，不跑新實驗）
    - C6 資料洩漏（`baselines/results/leakage_analysis.json`）
    - 與 proposal 的偏離與理由，每項一句，理由寫實際原因、不臆測：
      1. SAM → Faster R-CNN（COCO）／Grounding-DINO
      2. DeBERTa-v3-large → base
      3. 影像骨幹不微調：proposal 寫微調 SigLIP／ViT-Large，實際凍結 SigLIP、只訓練上層（Image LR 與 E2–E8 皆是）
      4. Subtask 2 的文字模型不微調：proposal 寫微調 DeBERTa-v3-large，實際用凍結特徵（Text LR 與 E1）；微調只用在 Subtask 1
      5. Grad-CAM → attention 熱圖（C5），並註明 attention 不等於解釋
      - 期中已列的其他變更（Micro F1 為主要指標、從有標籤的訓練資料切 dev／test、Qwen2.5-VL 取代 LLaVA-1.6／Qwen2-VL、加入 Subtask 1）在 7 Experimental Setup 用一兩句帶過
    - 各模型的訓練集不同（Leo 的模型用 437 張英文圖，Image LR 用 647 張含多語圖，以 `RUNS.md` 與 `subtask2_metrics.json` 的實際數字為準）
    - 兩人一組已獲教授同意
    - dev 只有 20 張的限制
    - 最佳模型照預先註冊的規則選（2026-11-26 決定，見 `docs/notes/2026-11-26-best-model-rule-decision.md`）：
      - Experimental Setup 寫明規則 commit 早於 E12、E12 設定也是候選。
      - Results 照規則報告 4 項主要比較。
      - 若最佳模型是 dev 上與 image-only 近乎平手、test 卻較差的模型，在 Limitations 用它說明 20 張 dev 的選擇雜訊。
      - 數字一律來自 `results/subtask2_final.json`。
    - README 舊數字 0.4421 為何作廢
    - 回應老師期中評論（spec 8.1）：
      1. Introduction 的設計動機段落：M1–M4 各一句「觀察 → 設計選擇」；M1 的數字引用 `results/c1_image_text.json`；M3、M4 寫成假設，並指向 E5–E8 的結果
      2. Methods 的「各模型讀到的文字」小表：Qwen 2a／2b 為標題 + 前 1500 字元、E12 為只給標題或全文、Text LR 與 late fusion 為全文逐段編碼後取平均、Leo 的模型為前 256 個 DeBERTa token（請 Leo 確認）；Analysis 的 C2 與 E12 回答「為什麼變差」與「標題或全文」，數字只引用 `results/c2_vlm_text.json` 與 `results/subtask2_final.json`
      3. Experimental Setup 揭露 2026-11-10 的例外（鎖定前讀過 test 的 Qwen 預測），以及 E12 因此不做任何選擇
  - 寫作規則（回應評論 3）：正文以段落為主，每段第一句寫結論，接著是數字與原因；條列只用於少數平行短清單（例如 proposal 偏離），全文最多 3 處；數字放在表格或圖中，正文不重複整排數字。
- [ ] **Step 6: 產生與檢查**

```bash
$PY report/build_tables.py && $PY report/check_numbers.py && $PY report/generate_report.py
```

Expected：check_numbers 結束代碼 0；PDF ≤10 頁；zip ≤10 MB（腳本超過就會停止）。

- [ ] **Step 7: 互相審閱**：除了內容，逐項確認老師的三點都有回應（動機段落、文字長度小表與 C2／E12、條列不超過 3 處）。審閱後 commit：`git commit -m "docs(report): final report"`

---

### Task 14: 投影片與排練（兩人，12/2 前）

**Files:**
- Create: `slides/outline.md`、`slides/FrameVision_Final_Slides.pdf`（加上所選工具的原始檔）

- [ ] **Step 1:** 依 spec 第 7 節寫 `slides/outline.md`：每頁的標題、要講的一句重點、使用的圖（路徑指到 `results/figures/` 或 `slides/figures/`）、主講人。最佳模型頁依 `selection.best_model` 決定內容。
- [ ] **Step 2:** 用前置條件決定的工具製作投影片；數字只引用 `results/subtask2_final.json`（可用 `check_numbers.py` 檢查講稿或投影片匯出的文字）。
- [ ] **Step 3: 對照評分項檢查**：Team（含 Team number 與分工）、Task、Resources（附 URL）、Best Model、Results、Analysis、Lessons Learned、沒有整張大表格。
- [ ] **Step 4: 計時排練**（12/1–12/2）：Expected：19–20 分鐘；超時就把細節移到備用頁。準備 QA 可能問題：為什麼不用 SAM、為什麼不微調影像與文字骨幹、為什麼加文字變差（給的是標題還是全文）、40 張圖的 CI、dev 只有 20 張怎麼選模型。
- [ ] **Step 5:** 匯出 PDF 並 commit：`git commit -m "docs(slides): final slides"`

---

### Task 15: 打包與交件（En-Shuo，12/3）

**Files:**
- Modify: `README.md`（結果表格改由 `build_tables.py` 產生的 `<!-- table:main -->` 區塊；說明 repo 結構與重現步驟）

- [ ] **Step 1:** 更新 README，並在 README 加上 `check_numbers.py` 的檢查（README 也不可有手打數字）。
- [ ] **Step 2: 最終驗證**

```bash
$PY -m pytest tests -q && (cd baselines && $PY -m unittest discover -s tests) && $PY report/check_numbers.py && $PY report/generate_report.py
```

Expected：全部測試通過；報告 ≤10 頁；zip ≤10 MB。

- [ ] **Step 3:** 經使用者同意後 push `final/unify` 並開 PR 進 main；Leo 的 `final/region-experiments` 先合併。
- [ ] **Step 4:** 使用者在 Canvas 上傳三個檔案（12/3 23:59 前）：`report/FrameVision_Final_Report.pdf`、`report/FrameVision_Final_Code.zip`、`slides/FrameVision_Final_Slides.pdf`。
