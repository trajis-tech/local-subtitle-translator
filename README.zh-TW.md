# Trajis SmartSRT（可攜式、離線模型）

Trajis SmartSRT 使用 **llama-cpp-python** 執行本地字幕翻譯流程。
設計為**可攜式**：所有依賴項都在此資料夾內。

✅ **離線優先**：先執行 `install.bat`（Windows）或 `./install.sh`（Linux/macOS）**一次**（需網路）下載並安裝所有依賴；之後使用 `start.bat` 或 `./start.sh` **離線啟動**即可。

❗ **模型不會自動下載。** 您需要手動下載 GGUF 檔案並放入 `./models/`。

---

## 翻譯流程概覽

翻譯流程分為 **6 個執行階段**（A→B→C/D→E→F），依序執行。**單一 brief**：目前僅一份 `./work/brief_work.jsonl`，每階段更新；在 C/D/E/F 前會先複製為 `brief_v1.jsonl` / `brief_v2.jsonl` / `brief_v3.jsonl` / `brief_v4.jsonl`（snapshot）。

- **Run A**：音訊情緒/語氣分析（所有字幕行）
  - 從影片中提取每條字幕對應的音訊片段
  - 分析情緒、語氣、強度、說話方式
  - 結果保存至 `./work/audio_tags.jsonl`
  - **注意**：音訊模型已**預先打包**且與代碼**緊密耦合**。**請勿修改或替換。**

- **Run B**：主模型產生翻譯描述（所有字幕行）
  - 使用主推理模型（GGUF、llama-cpp-python）產生翻譯指導。
  - 輸入：英文字幕 + 前後各一句上下文 + 音訊標籤（皆英文）。
  - 輸出：JSON — **全部為英文**（語言中立）：`meaning_tl`、`draft_tl`、`tl_instruction`、`ctx_brief`、**`referents`**、**`tone_note`**、**`scene_brief`** 等。**每階段只輸出一個 need**：v1 僅輸出 **`need_vision`**（布林）。可選 `transliteration_requests`、`omit_sfx`；`reasons`（含供 Run F 使用的 PACK）。
  - 結果保存至 `./work/brief_work.jsonl`（單一當前 brief）。

- **Run C**：單張影像 fallback（選用、條件觸發）
  - 當 **`need_vision === true`**（來自當前 brief）時觸發。
  - 更新前：複製 `brief_work.jsonl` → `brief_v1.jsonl`。在字幕時間區間正中間提取一張影像；視覺模型分析場景/人物/動作。
  - 重新產生 brief（帶單張視覺提示）→ 更新 `./work/brief_work.jsonl`。**每階段只輸出一個 need**：v2 僅輸出 **`need_multi_frame_vision`**（布林）每筆。

- **Run D**：多張影像 fallback（選用、條件觸發）
  - 當 **`need_multi_frame_vision === true`**（來自當前 brief）時觸發。
  - 更新前：複製 `brief_work.jsonl` → `brief_v2.jsonl`。在字幕時間區間內等距提取 N 張影像（可設定，預設：3）；視覺模型分析並合併描述。
  - 重新產生 brief（帶多張影像提示）→ 更新 `./work/brief_work.jsonl`。**每階段只輸出一個 need**：v3 僅輸出 **`need_more_context`**（布林）；供 Run E（上下文擴充）使用。

- **Run E**：上下文擴充（條件）
  - 更新前：複製 `brief_work.jsonl` → `brief_v3.jsonl`。對 **`need_more_context === true`** 的項目以 **前 3／後 3 句** 再跑 stage2 更新 brief，寫回 `./work/brief_work.jsonl`。僅更新 brief，不產出目標語。

- **Run F**：最終翻譯（所有字幕行）
  - Run F 開始前會先將當前 brief 複製為 `brief_v4.jsonl` snapshot，之後讀取**當前 brief**（`brief_work.jsonl`，已由 E 更新）與 PACK 產出整集目標語字幕並寫入 SRT。**設定**：`config.PipelineConfig.run_e_scheme`（預設 `"full"`）— **UI**：Run F scheme 下拉。**執行**：`pipeline_runs.run_final_translate()`；輸出：`items.translated_text` 與 `work_dir/final_translations.jsonl`。**SRT 寫回**：`app.py` 依 `(round(start_ms), round(end_ms))` 對齊，保留原文 `<i>` 標籤。
  - **一次只載入一個模型**：同一時間僅載入 MAIN 或 LOCAL 其一；詞彙表僅在輸出端套用。
  - **穩健性**：所有 chat 呼叫經統一入口 **chat_dispatch**。heavy 請求可改以 one-shot 子行程執行；失敗時 fallback 進程內。

### Run F 方案（依主模型／在地化模型強弱選擇）

在介面中從 **Run F scheme** 下拉選單選擇方案。選項：`full` | `main_led` | `local_led` | `draft_first`（無效時 fallback 為 `full`）。

| Scheme | 選擇時機（主／地） | Phase1（初稿來源） | Phase2（潤飾） |
|--------|---------------------|------------------------|-----------------|
| **Full** | 主強、地強 | MAIN 群組翻譯 → draft_map | LOCAL 潤飾 |
| **MAIN-led** | 主強、地弱 | MAIN 群組翻譯 → draft_map | 無 |
| **LOCAL-led** | 主弱、地強 | PACK draft_tl → draft_map；可選 LOCAL 填 idiom slot | LOCAL 潤飾 |
| **Draft-first** | 主弱、地弱 | PACK draft_tl → draft_map；可選 LOCAL 填 idiom slot | 無 |

- **Full**：Phase 1 — 載入 MAIN（reason），建立句子群組，依 `group_translate_max_segments`（預設 4）切 chunk；每 chunk 呼叫 `stage_main_group_translate`；**可部分採用 MAIN 輸出**（依 **id** 對齊；缺的句用 PACK draft_tl 或 en_text）。Phase 2 — 載入 LOCAL，依 chunk 執行 `local_polish`（`local_polish_chunk_size` 預設 60）；僅接受請求內 key 且通過長度／污染檢查的結果寫入 draft_map。最終：詞彙表 + strip_punctuation → `translated_text`。
- **MAIN-led**：Phase 1 — 同 Full（MAIN 群組翻譯 → draft_map）。Phase 2 — **略過**。最終：詞彙表 + strip_punctuation。
- **LOCAL-led**：不載入 MAIN。`draft_map = _build_draft_map_from_pack(...)`。若有項目含 `idiom_requests`，載入 LOCAL，逐行呼叫 `stage3_suggest_local_phrases`，以 `_fill_draft_with_suggestions` 填入 slot 並更新 draft_map。再載入 LOCAL 對全部行執行 `local_polish`。最終：詞彙表 + strip_punctuation。
- **Draft-first**：不載入 MAIN。僅以 PACK 建立 draft_map；若有 idiom_requests 則載入 LOCAL 取得建議並填入；**不**潤飾。弱在地化模型使用 **STRICT** prompt 與 raw_decode 後備解析，避免輸出格式錯誤。

**對齊**：Run F 對齊一律依 **sub_id** 與**時間戳 (start_ms, end_ms)**，不依列表索引，以降低錯位風險。

**語言政策（階段間不混用）**：
- **Run A–E**：所有 prompt 與所有模型輸出皆**僅限英文**。Run A（音訊）、Run B/C/D（brief）、Run E（上下文擴充）不接收也不產出目標語；brief 為語言中立的英文，供 Run F 以單一英文介面翻譯。
- **Run F**：所有**指令（prompt）為英文**；主模型與在地化模型的輸入為**英文**（片段、tl_instruction、上下文）。僅**模型輸出**（segment_texts[].text、潤飾後句子、片語建議）為**目標語言**，以保持 prompt 語言一致，避免目標語流入前階段。
- **強制**：Run B brief 輸出會經淨化（例如 `tl_instruction` 必須僅英文；若模型在 brief 欄位輸出目標語則替換為預設）。Run F 使用 `_tl_instruction_for_run_e()`，確保翻譯階段始終依正確目標語系輸出。

**Prompt 角色**（`model_prompts.csv`）：MAIN（main_group_translate）專注**原文（SOURCE）**；強在地化（如 Llama-Breeze2-8B）專注**目標語**、自然化／潤飾；弱在地化（如 Breeze-7B、custom-localization）使用 **STRICT** 格式；解析失敗時以 `raw_decode` 取第一個 `{...}`。

**音譯**：需在目標語音譯的人名或專有名詞由**在地化模型**負責；**主模型**（Run B）在 PACK 中以 `transliteration_requests`（字串陣列）提出。Run F Phase2（local_polish）會收到這些詞並在 polish prompt 中附加「Transliterate (音譯) in target language for these terms: …」，由 LOCAL 輸出音譯形式。

**CC／狀聲詞**：**主模型**（Run B）篩掉音效與狀聲詞（如 `[laughter]`、`[sigh]`、`*gasps*`）。可設 `omit_sfx: true` 且 `draft_tl` 為空（僅 SFX 的句）；對白＋SFX 時 `draft_tl` 僅含對白。Run F 在建好 draft_map 後套用 omit_sfx，該類句輸出為空。

**Run F 設定**（`config.py`）：`run_e_scheme`（UI：Run F scheme）、`group_translate_max_segments`（預設 4）、`local_polish_chunk_size`（預設 60）、`strip_punctuation`、`strip_punctuation_keep_decimal`、`strip_punctuation_keep_acronym`。

### 主要特性

- **單一模型載入**：任一時間只載入一個模型（音訊、推理、視覺或翻譯）
- **可續跑**：每個 run 將中間結果保存至 `./work/`（JSONL 格式）
- **錯誤容錯**：如果視覺/音訊失敗，流程會使用最佳可用 brief 繼續
- **進度追蹤**：進度條顯示當前步驟和完成百分比

---

## 影片輸入（FFmpeg 注意事項）

Gradio 內建的 **Video** 元件需要外部 **`ffmpeg`** 執行檔才能進行伺服器端處理。
如果沒有 `ffmpeg`，可能會出現 **"Executable 'ffmpeg' not found"** 錯誤。

為保持本專案**完全可攜式**（無需系統級安裝），本專案使用 **File** 輸入來處理影片。

- 您需要影片檔案用於 **Run A（音訊）** 和 **Run C/D（視覺）**。
- 使用 OpenCV (opencv-python) 擷取視覺畫面，使用 ffmpeg 提取音訊片段。
- **ffmpeg**：**Windows** – `install.bat` 若 PATH 中無 ffmpeg 會自動下載到 `runtime\ffmpeg`。**Linux / macOS** – `install.sh` 僅檢查 PATH 中是否有 ffmpeg；若缺失請手動安裝並參閱 **FFMPEG_INSTALL.md**。

若您使用的是仍使用 Video 元件的舊版 zip，請更新至最新版，或安裝 FFmpeg 並加入 PATH。

## 安裝與啟動（離線優先）

本專案採**離線優先**：先執行 **install** 一次（需網路），之後使用 **start** 隨時離線啟動。

**⚠️ 重要 - NVIDIA 顯卡使用者：**
- **請先安裝 CUDA Toolkit 12.9**（或 12.x）**再執行** `install.bat` / `install.sh`
- 下載：https://developer.nvidia.com/cuda-downloads
- 預建的 llama-cpp-python 輪檔需要先安裝 CUDA 才能啟用 GPU 加速

1. 將此資料夾解壓縮到任意位置（例如：`G:\Trajis SmartSRT`）。

2. **安裝（一次、需網路）** — 下載並安裝所有依賴與工具：
   - **Windows**：雙擊 `install.bat` → 可攜式 Python、venv、所有 Python 依賴（含音訊 Run A、視訊），若有 GPU 可選安裝 CUDA PyTorch，若 PATH 無 ffmpeg 則下載至 `runtime\ffmpeg`，Run A 音訊模型下載至 `models\audio`，預建 llama-cpp-python 輪檔、config、BOM。GGUF 模型需手動下載（見下方）。
     - **安裝後預計占用空間**：約 6-8 GB（僅 CPU：約 4-5 GB；含 CUDA PyTorch + ffmpeg：約 6-8 GB）
   - **Linux / macOS**：執行 `./install.sh`（同上：.venv、依賴、音訊模型、預建 llama-cpp-python 輪檔、config、BOM）。必要時：`chmod +x install.sh`
     - **安裝後預計占用空間**：約 4-5 GB（不含系統 Python 與 ffmpeg）

3. **啟動（離線）** — 不下載、不連網：
   - **Windows**：雙擊 `start.bat` → 檢查 .venv 與模型檔案後啟動 UI。
   - **Linux / macOS**：執行 `./start.sh`。必要時：`chmod +x start.sh`

- **解除安裝**：執行 `uninstall.bat`（Windows）或 `./uninstall.sh`（Linux/macOS）可刪除此資料夾內的執行環境、venv 與快取。必要時：`chmod +x uninstall.sh`

**顯卡支援：**

- **NVIDIA 顯卡**：支援 CUDA 12.x（建議 CUDA 12.9；RTX 20/30/40/50 系列、GTX 16 系列及更新型號）
- **AMD 顯卡**：ROCm 支援（實驗性，需手動設定）
- **Intel Arc 顯卡**：oneAPI 支援（實驗性，需手動設定）
- **CPU**：針對 Intel CPU 最佳化（無需 AVX-512 指令集），可在所有現代 x86-64 處理器上執行

**選用 – 僅安裝音訊依賴（Linux / macOS）：**

- 執行 `install.bat` 或 `install.sh` 已會安裝 Run A（音訊）依賴。僅在需要單獨重裝音訊依賴（torch、transformers、soundfile、scipy）而不做完整安裝時，可使用 `./scripts/install_audio_deps.sh`。需 Python 3，可選已啟用的 `.venv`。

所有內容都保留在此資料夾內（可攜式/隔離）。

---

## 模型相容性與目錄結構（必要）

本應用程式使用的**文字與視覺**模型必須為 **GGUF** 格式，且與 **llama-cpp-python** 相容。模型檔案由您自行準備，應用程式不會自動下載。

建立並使用以下目錄結構：

```
models/
  main/     ← 主推理模型（Run B）；一個或多個 .gguf 檔案
  local/    ← 在地化／翻譯模型（Run F）；一個或多個 .gguf 檔案
  vision/   ← 選用視覺模型（Run C/D）；主 .gguf + mmproj .gguf
  audio/    ← Run A 音訊模型（由安裝腳本或首次執行時下載）
```

### 相容性

- **主模型與在地化模型**：任何可在 llama-cpp-python 下運作的 **GGUF** 模型（具 chat 模板的 instruct/chat 模型）。將檔案放入 `./models/main/` 與 `./models/local/`。若量化為**分片**（多個 `.gguf`），請下載**全部分片**並放在同一資料夾。
- **視覺模型（選用）**：任何 llama-cpp-python 支援的 **GGUF 視覺**模型（主模型 + mmproj）。將兩個檔案放入 `./models/vision/`。應用程式會依檔名自動偵測類型。您可在 `config.json` 的 `vision.text_model` 與 `vision.mmproj_model` 指定確切檔名。
- **音訊（Run A）**：Run A 情緒模型於首次執行時從 Hugging Face Hub 下載（無本地 GGUF）。使用 Transformers `audio-classification`；依賴：`torch`、`transformers`、`soundfile`、`scipy`。

### 參數與量化（通用建議）

- **量化**：較輕量化（如 **Q4_K_M**）較省 VRAM、較快；較重量化（**Q5_K_M**、**Q6_K**、**Q8_0**）品質較好但需更多 VRAM 與磁碟。請依 GPU/RAM 選擇。
- **模型大小**：參數量越大（如 14B、7B）所需 VRAM/RAM 越多。模型**一次只載入一個**，故 VRAM 以**單一最大模型**為準。
- **上下文大小**：較大的 `n_ctx_*`（如 8192）有利長上下文，但會增加 VRAM（KV 快取）。若發生 OOM，可降低 `n_ctx_*` 或 `n_gpu_layers_*`。

### config.json 建議起點（依硬體調整）

- **16 GB VRAM**：`n_ctx_reason=8192`, `n_ctx_translate=4096`, `n_gpu_layers_reason=60`, `n_gpu_layers_translate=60`
- **12 GB VRAM**：`n_ctx_reason=4096`, `n_ctx_translate=2048`, `n_gpu_layers_reason=50`, `n_gpu_layers_translate=50`
- **8 GB VRAM**：`n_ctx_reason=2048`, `n_ctx_translate=2048`, `n_gpu_layers_reason=35`, `n_gpu_layers_translate=35`
- **僅 CPU／低 RAM**：建議使用 Q4_K_M（或更輕）與較小上下文；減少 `n_gpu_layers_*` 或設為 0 以全 CPU 執行。

---

## config.json

執行 `install.bat`（或 `install.sh`）時，若沒有 `config.json`，會執行 `scripts/plan_models.py` 建立一個最佳努力的 `config.json`。`start.bat` / `start.sh` 不會建立 config，僅負責啟動程式。
如果選擇不同的量化檔案，請編輯它。

關鍵欄位：

- `models_dir`：預設 `models`
- `reason_dir`：主模型目錄（預設：`./models/main/`）
- `translate_dir`：在地化模型目錄（預設：`./models/local/`）
- `vision_text_dir` / `vision_mmproj_dir`：視覺模型目錄（預設：`./models/vision/`）
- `audio.model_id_or_path`：Run A 使用的 Hugging Face 模型 id（預設：`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`）
- `audio.enabled`：啟用/停用音訊分析（預設：`true`）
- `audio.device`：`"auto"`（有 CUDA 則用）、`"cuda"` 或 `"cpu"`（預設：`auto`）
- `audio.device_index`：使用 CUDA 時的 GPU 索引（預設：`0`）
- `audio.batch_size`：情緒推論的批次大小；較大可提高 GPU 利用率（預設：`16`）
- `vision.enabled`：選用視覺 fallback
- `pipeline.n_frames`：多張影像視覺的幀數（預設：`3`）
- `pipeline.work_dir`：中間結果目錄（預設：`./work/`）
- `pipeline.run_e_scheme`：Run F 方案 — `"full"` | `"main_led"` | `"local_led"` | `"draft_first"`（預設：`"full"`）。詳見上方 **Run F 方案**。
- `pipeline.local_polish_chunk_size`：Run F local_polish 每批行數（預設：`60`）
- `pipeline.group_translate_max_segments`：Run F 主翻譯每子群組最大段數（預設：`4`）
- `pipeline.isolate_heavy_requests`：為 `true` 時，過重的請求（超過 token/行數/段數閾值）改以 one-shot 子行程執行，避免 OOM 拖死主程式（預設：`true`）
- `pipeline.isolate_heavy_timeout_sec`：隔離子行程逾時秒數（預設：`600`）
- `pipeline.strip_punctuation`：為 `true` 時，在 Run F 最終輸出時去除標點（預設：`true`）
- `pipeline.strip_punctuation_keep_decimal`：為 `true` 時，保護小數如 `3.14` 不被拆開（預設：`true`）
- `pipeline.strip_punctuation_keep_acronym`：為 `true` 時，保護縮寫如 `U.S.` 不被拆開（預設：`true`）

---

## 工作目錄（中間結果）

所有中間結果都保存至 `./work/` 目錄（JSONL 格式）：

- `audio_tags.jsonl` - Run A 結果（音訊情緒/語氣分析）
- `brief_work.jsonl` - 當前 brief（Run B 寫入；C/D/E 更新；Run F 讀取）
- `brief_v1.jsonl` - Run C 前 snapshot（更新前複製）
- `brief_v2.jsonl` - Run D 前 snapshot
- `brief_v3.jsonl` - Run E 前 snapshot
- `brief_v4.jsonl` - Run F 前 snapshot
- `vision_1frame.jsonl` - Run C 結果（單張影像視覺分析）
- `vision_multiframe.jsonl` - Run D 結果（多張影像視覺分析）
- `final_translations.jsonl` - Run F 結果（最終翻譯文字，新格式）

**JSONL 格式相容性：**

流程支援**舊格式**（使用 `idx` 對齊）和**新格式**（使用 `sub_id` 對齊）：

- **舊格式**：使用 `idx`（整數索引）識別字幕行
  - 範例：`{"idx": 0, "start_ms": 1000, "end_ms": 2000, ...}`
- **新格式**：使用 `sub_id`（基於 hash 的唯一識別符）確保資料對齊
  - 範例：`{"sub_id": "a1b2c3d4_0", "start_ms": 1000, "end_ms": 2000, ...}`
  - `sub_id` 由 `hash(start_ms, end_ms, text_raw)` 生成，確保跨 run 的一致性

流程會自動檢測格式並在需要時進行轉換。新的 run 將使用 `sub_id` 格式以確保更好的資料完整性。

**續跑功能**：如果 JSONL 檔案存在且條目數量正確，流程會自動載入並跳過該 run。流程支援從舊格式（`idx`）和新格式（`sub_id`）續跑。

**手動續跑**：您可以刪除特定 JSONL 檔案以僅重新執行那些步驟。

---

## UI 使用方式

1. **上傳檔案**：影片（MKV/MP4）和 SRT（英文字幕）
2. **選擇執行模式**：`all`（A→B→(C/D)→E→F，預設）| **A**（音訊）| **B**（brief）| **C**（單幀視覺）| **D**（多幀視覺）| **E**（上下文擴充）| **F**（翻譯）
3. **Run F scheme**（下拉選單）：依主模型／在地化模型強弱選擇 — **Full** | **MAIN-led** | **LOCAL-led** | **Draft-first**。詳見上方 **Run F 方案**。
4. **選用備援**（介面勾選）：
   - **啟用視覺備援（Run C/D）**：勾選且 brief 為 **need_vision**／**need_multi_frame_vision** 時執行單幀（C）或多幀（D）視覺並更新 brief。需本機 GGUF 視覺模型。
   - **啟用更多上下文備援（Run E）**：勾選時，具 **need_more_context** 的項目會在 Run F 前以前 3／後 3 句擴充並更新 brief。建議啟用。
   - **Max frames per subtitle (Run D)**：多幀視覺的幀數（預設 1–4）；**Frame offsets**：取樣位置。
5. **點擊「🚀 Translate」**並監控進度
6. **下載**翻譯完成的 SRT 檔案
7. **一鍵重設**：點擊 **「Reset」** 可清空所有輸入、輸出與日誌並恢復預設值，以便開始新的翻譯工作

**介面說明**：日誌區塊以**最新訊息在上方**顯示。`model_prompts.csv` 以 UTF-8（含 BOM）讀寫；`start.bat` / `start.sh` 啟動時會執行 `ensure_csv_bom.py` 以維持檔案編碼正確。

---

## 自訂模型提示詞（model_prompts.csv）

翻譯流程使用 `model_prompts.csv` 中定義的提示詞。每個模型的提示詞會依**模型檔名**自動匹配（不分大小寫的子字串匹配）。檔案應為 **UTF-8（含 BOM）**；`start.bat` 與 `start.sh` 啟動時會執行 `scripts/ensure_csv_bom.py` 以確保編碼正確。

### 模型官方 prompt 對齊

提示詞依各模型家族的**官方**聊天格式與建議設計，以維持行為可預測、相容：

- **Qwen2.5 (ChatML)**：System 角色 + User 角色；JSON Mode 做結構化輸出。模板使用 `chat_format=chatml` 與嚴格的「僅輸出有效 JSON、無 markdown」指示，符合官方 Qwen 用法。
- **Gemma 2（如 TranslateGemma）**：**無 system 角色**；所有指令在第一個 user turn。後端在 `chat_format=gemma` 時會將 system 內容合併進 user 訊息，模型只看到單一 user turn。
- **Mistral / Llama 2（如 Breeze、Llama-Breeze2）**：`[INST]` 風格；system prompt 會接在第一個 `[INST]` 前。用於 `local_polish`、`localization` 角色，必要時採 STRICT JSON 輸出。
- **Vision（Moondream、LLaVA）**：提示詞在程式內依 handler 套用；聊天格式依視覺模型檔名自動偵測。輸出一律為**英文**視覺描述（不產出字幕）。

CSV 的 **notes** 欄會標註該角色屬「Run A~D 全英文」或「Run F：僅輸出為目標語」，自訂列請維持相同語言邊界。

### 模型名稱匹配

- **運作方式**：應用程式提取模型檔名（例如，`my-main-model-q5_k_m.gguf`）並與 CSV `model_name` 欄位匹配。
- **匹配規則**：如果檔名**包含** CSV `model_name`（不分大小寫），則匹配成功。
  - 範例：`my-main-model-q5_k_m.gguf` 匹配 `my-main-model`
  - 範例：`my-local-model-00001-of-00002.gguf` 匹配 `my-local-model`
- **填寫內容**：使用出現在您模型檔名中的**唯一子字串**。通常基礎模型名稱（不含量化後綴）即可。

### CSV 欄位指南

| 欄位 | 說明 | 範例 |
|--------|-------------|---------|
| `model_name` | 在檔名中匹配的子字串（不分大小寫） | `my-main-model` |
| `role` | `main`（Run B）、`main_group_translate`（Run F 群組翻譯）、`main_assemble`（Run F 舊版組裝）、`localization`（Run F 片語建議）、`local_polish`（Run F 批次順口化）或 `vision`（Run C/D） | `localization` |
| `source_language` | 輸入語言（通常是 `English`） | `English` |
| `target_language` | 輸出語言（語言代碼：`en`、`zh-TW`、`zh-CN`、`ja-JP`、`es-ES`） | `zh-TW` |
| `chat_format` | 模型的聊天模板（`chatml`、`llama-3`、`mistral-instruct`、`moondream`） | `chatml` |
| `system_prompt_template` | 系統提示詞（角色定義） | 見下方範例 |
| `user_prompt_template` | 使用者提示詞（含佔位符） | 見下方範例 |
| `notes` | 說明（英文） | `Localization model for Traditional Chinese (Taiwan)` |

### 佔位符

在 `user_prompt_template` 中使用這些佔位符：

**Run B（main）佔位符：**
- `{line}` → 當前英文字幕行
- `{context}` → 完整上下文（Prev-1、Current、Next-1、視覺提示（如有））

**Run F（main_group_translate）** – MAIN 一次翻譯整組字幕；輸出依 sub_id 對齊：
- `{target_language}` → 語言代碼（如 zh-TW、ja-JP）
- `{tl_instruction}` → 目標語言指令（取自首段 PACK 或程式產生）
- `{segments_json}` → 每段 `{id, en, ms}` 的 JSON 陣列

**Run F（local_polish）** – LOCAL 批次順口化所有草稿（僅目標語；輸入不含英文）：
- `{tl_instruction}` → 目標語言指令
- `{lines_json}` → `{id, text}` 的 JSON 陣列（目標語草稿）

**Run F（localization）** – 僅用於片語建議（當該句有 idiom 槽位時）：
- `{tl_instruction}` → Run B/D PACK 中的目標語言指令
- `{requests_json}` → 片語請求（slot、meaning_tl、register、max_len）的 JSON
- `{target_language}` → 語言代碼（如 zh-TW、ja-JP）

**Run F（main_assemble）** – 舊版 Stage4 單行組裝（主模型潤飾 draft + 片語建議）：
- `{target_language}` → 語言代碼（如 zh-TW、ja-JP、es-ES）
- `{line_en}` → 原始英文字幕行
- `{ctx_brief}` → 上下文摘要（**英文**，來自 PACK）
- `{draft_prefilled}` → 已填入片語的 draft（&lt;I1&gt;/&lt;I2&gt; 已替換）
- `{suggestions_json}` → 使用的 slot→片語 JSON

詞彙表**不會**注入任何模型，僅在最終輸出端做強制替換。詞彙表條目可依目標語言指定替換（`targets` 依語言代碼，如 `zh-TW`、`ja-JP`），或使用單一 `to` / `zh` 作為備用。

**Run C/D（vision）佔位符：**
- `{line}` → 當前英文字幕行

### 提示詞風格：Base vs Instruct 模型

#### Base 模型（非 Instruct）
- **特徵**：較簡單、直接的提示詞，無結構化指令格式
- **使用時機**：您的模型是基礎/完成模型（未針對指令進行微調）
- **風格**：直接問題或簡單任務描述

#### Instruct 模型
- **特徵**：結構化指令格式，含編號規則和明確任務定義
- **使用時機**：您的模型經過指令微調（Instruct、Chat 等）
- **風格**：結構化，含規則、編號步驟、明確輸入/輸出定義

### CSV 中的範例

CSV 包含每個角色的範例行：

1. **`(custom-main-base)`** - Run B 的 Base 模型範例
2. **`(custom-main-instruct)`** - Run B 的 Instruct 模型範例
3. **`(custom-localization-base)`** - Run F 的 Base 模型範例
4. **`(custom-localization-instruct)`** - Run F 的 Instruct 模型範例
5. **`(custom-vision-base)`** - Vision 的 Base 模型範例
6. **`(custom-vision-instruct)`** - Vision 的 Instruct 模型範例

### 新增您自己的模型

1. **複製範例行**（例如，`(custom-main-instruct)`）
2. **更改 `model_name`** 以匹配您的模型檔名子字串
3. **設定 `role`**（`main`、`localization` 或 `vision`）
4. **設定 `target_language`** 為以下語言代碼之一：
   - `en` - 英文（適用於 Run B main 模型）
   - `zh-TW` - 繁體中文（台灣）
   - `zh-CN` - 簡體中文（大陸）
   - `ja-JP` - 日文
   - `es-ES` - 西班牙文
   - 或其他 IETF 語言代碼（視需要）
5. **設定 `chat_format`** 以匹配您模型的聊天模板
6. **撰寫 `system_prompt_template`**（角色定義）
7. **撰寫 `user_prompt_template`**（含佔位符的任務）
8. **填寫 `notes`**（英文說明）

### 重要注意事項

- **語言邊界**：**Run A–D**（音訊、brief v1/v2/v3、視覺）：prompt 與模型輸出必須**僅限英文**。**Run F**（main_group_translate、local_polish、localization、main_assemble）：prompt 為**英文**；僅**輸出**（翻譯句、片語建議）為目標語。請勿在 Run F 的 prompt 中寫入目標語指令（例如中文或日文）— 請使用英文（例如「Output ONLY the translated subtitle in the target language (locale: zh-TW).」），以免 prompt 與輸出語言混用。
- **提示詞語言**：Run B/C/D 使用僅限英文的 prompt；Run F 使用英文指令，預期模型輸出為目標語。
- **聊天格式**：必須匹配您模型的聊天模板。錯誤的格式可能導致輸出不佳或錯誤。
  - **視覺模型**：聊天格式由 `LocalVisionModel` 依模型檔名自動偵測。CSV 的 `chat_format` 欄位僅供說明。
- **佔位符**：始終使用確切的佔位符名稱（`{line}`、`{context}`、`{target_language}` 等）。它們會自動替換。
- **Run B/C/D 輸出（brief）**：必須請求 JSON 含 `target_language`、`tl_instruction`、`meaning_tl`、`draft_tl`、`idiom_requests`、`ctx_brief`、referents、tone_note、scene_brief — **全部為英文**（語言中立）。**每階段只輸出一個 need**：**v1** 僅輸出 **`need_vision`**；**v2** 僅輸出 **`need_multi_frame_vision`**；**v3** 僅輸出 **`need_more_context`**。可選 `plain_en`、`idiom_flag`、`transliteration_requests`、`omit_sfx`；`notes` 可含供 Run F 使用的 PACK。

---

## 疑難排解

### 「缺少必要的模型檔案」
執行 `start.bat`，它會開啟此 README 並告訴您缺少哪些檔案。

### GPU 未偵測到或效能緩慢
本專案已包含針對 NVIDIA GPU（CUDA 12.x）與 Intel CPU 最佳化的**預建 llama-cpp-python 輪檔**。
- **NVIDIA 顯卡**：請確保已安裝最新 GPU 驅動程式。應用程式會自動偵測並使用 CUDA。
- **Intel CPU**：CPU 版本針對現代 Intel 處理器最佳化，無需 AVX-512 指令集。
- **AMD/Intel Arc 顯卡**：實驗性支援，但需手動設定（預建輪檔不包含）。

### Windows「符號連結」警告
此警告來自 Hugging Face 快取。由於本專案不再自動下載模型，您可以忽略它。

### 音訊模型錯誤（Run A）
Run A 使用 Hugging Face 模型 **ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition**。請執行 `pip install -r requirements_base.txt`（torch、transformers、soundfile、scipy）。首次執行會從 Hub 下載模型。音訊提取需在 PATH 中有 ffmpeg。

### ffmpeg 未找到
- **Windows**：`install.bat` 會在 PATH 無 ffmpeg 時自動下載至 `runtime\ffmpeg`。若失敗請參閱 **FFMPEG_INSTALL.md** 手動下載與安裝（BtbN 建置、winget，或將 `runtime\ffmpeg\bin` 加入 PATH）。
- **Linux / macOS**：`install.sh` 僅檢查 PATH 中的 ffmpeg，不會自動下載；請用套件管理員安裝並參閱 **FFMPEG_INSTALL.md**。

---

## 授權/免責聲明

這是本地工具。您負責模型的授權和使用。
