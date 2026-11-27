# 資料一致性檢查（給 Leo）

目的：確認 Leo 本機的 `data_official/` 和 En-Shuo 的報名後官方資料完全相同。已知兩邊的 Subtask 2 標籤一致（Leo 先前 commit 的 99 張 gold 全部對得上），這次要比對的是圖片和文字檔本身的位元組。

## 要傳給 Leo 的兩個檔案

- `scripts/data_manifest.py`：只用 Python 標準函式庫，Python 3.8 以上都能直接跑，不必另外安裝套件
- `results/data_manifest_enshuo.json`：En-Shuo 的 5270 個檔案的 SHA-256 雜湊，約 1.4 MB。內容只有相對路徑、雜湊、大小和 14 個 frame 名稱，沒有任何絕對路徑或帳號名稱。

## Leo 要跑的指令

把兩個檔案放在 FrameVision repo 根目錄（也就是 `data_official/` 所在的那層），然後執行：

```bash
python data_manifest.py compare data_manifest_enshuo.json data_official --ref-name En-Shuo --other-name Leo --save-other data_manifest_leo.json
```

跑完後請把終端機輸出和 `data_manifest_leo.json` 傳回來。

## 怎麼看結果

- `Verdict: PASS`：Subtask 2 pipeline 讀到的檔案全部相同，包括 `initial-training-data/subtask2/` 和 `training-second-batch-oct/EN/` 底下的所有檔案，以及 `labels.txt` 的 frame 名稱。結束碼是 0。
- `Verdict: FAIL`：會在 `FAIL:` 段落列出缺少、被移動或內容不同的檔案。結束碼是 1。
- 下列情況只列為警告，不算失敗：
  - Leo 沒放 Subtask 1、BG、PT 等 pipeline 用不到的資料夾
  - 文字檔只差在換行符號（CRLF 與 LF）或 BOM
  - `labels.txt` 名稱相同但順序不同
- 會自動忽略 `.DS_Store`、`__MACOSX/` 和 `._*` 這類系統檔。
- 如果給錯資料夾層級，程式會提示正確的根目錄，結束碼是 2。

## En-Shuo 這邊如何重新產生 manifest

```bash
python scripts/data_manifest.py write --root data/data-second-batch/data --labels data/trialdataSemEval2027Task9/labels.txt --out results/data_manifest_enshuo.json
```

En-Shuo 的資料根目錄沒有 `labels.txt`，所以這裡用 `--labels` 指向試用資料的那份。它的內容和官方 `SEMEVAL23_FRAMING_INITIAL_TRAIN-B/labels.txt` 相同。
