# 翻譯流程詳細說明

本文件描述目前專案的**完整翻譯流程**（Run A → Run E），包含資料結構、模型載入策略、PACK 格式與 Run E 兩段式流程。

---

## 一、整體架構

- **輸入**：影片檔（MKV/MP4）、英文字幕 SRT、目標語言（如 zh-TW、ja-JP）、詞彙表（選用）。
- **輸出**：目標語言字幕 SRT、以及 `./work/` 下的中間結果（JSONL）。
- **對齊單位**：`sub_id`（由 `hash(start_ms, end_ms, text_raw)` 或序號產生），所有 Run 皆以 `dict[sub_id, SubtitleItem]` 傳遞，確保資料不錯位。
- **模型載入原則**：**任一時刻只載入一個模型**。使用 `model_mutex.hold_model(model_type)` 將「載入 → 推理 → 卸載」整段包在鎖內；若併發觸發兩個流程，後者會阻塞直到前者完整卸載，避免同時存在兩模型。

---

## 二、資料結構

### SubtitleItem（每條字幕）

| 欄位 | 說明 |
|------|------|
| `sub_id` | 固定識別符（不可變） |
| `start_ms`, `end_ms` | 時間區間（毫秒） |
| `text_raw`, `text_clean` | 原始／清理後英文字幕 |
| `audio_meta` | Run A 填入：情緒、語氣、強度、說話方式等 |
| `brief_v1`, `brief_v2`, `brief_v3` | Run B/C/D 填入：BriefMeta |
| `vision_desc_1`, `vision_desc_n` | Run C/D 填入：單張／多張影像描述 |
| `translated_text` | Run E 填入：最終翻譯文字 |

### BriefMeta（Brief 一筆）

| 欄位 | 說明 |
|------|------|
| `translation_brief` | 語意摘要（目標語言；對應 Stage2 的 meaning_tl 或 fallback） |
| `reasons` | 原始說明字串，**內含 `PACK:{...}` JSON**（供 Run E 解析） |
| `version` | `"v1"` / `"v2"` / `"v3"` |
| `need_vision` | v1：是否需視覺脈絡（單張）才能精準翻譯 |
| `need_multi_frame_vision` | v2：單張視覺是否不足、需多張視覺 |

**最佳 Brief**：`item.get_best_brief()` 依序回傳 `brief_v3` → `brief_v2` → `brief_v1`，若皆無則回傳 `None`。

---

## 三、Run A：音訊分析

- **目的**：為每條字幕提供音訊線索（情緒、語氣等），供後續 Brief 使用。
- **模型**：Hugging Face Wav2Vec2（ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition），首次執行時從 Hub 下載。
- **流程**：
  1. 使用 `model_mutex.hold_model("audio")`：載入音訊模型、執行分析、結束前卸載（`del model; gc.collect()`）。
  2. 若存在 `work_dir/full_audio.wav` 則跳過一次性全片音訊提取；否則從影片提取全片音訊，再依每條字幕時間切片段（快速模式）。
  3. 若無法使用快速模式，則改為每條字幕單獨從影片提取音訊片段。
  4. 對每個片段呼叫 `audio_model.analyze_emotion(...)`，結果寫入 `item.audio_meta`（AudioMeta）。
  5. Run 結束後卸載音訊模型。
- **輸出**：`items` 已填 `audio_meta`；另寫入 `work_dir/audio_tags.jsonl`（若實作有寫入）。

---

## 四、Run B：Brief 產生 v1（主模型，無視覺）

- **目的**：僅依**文字 + 音訊**產生第一版 brief（含 PACK），不使用影像。
- **模型**：主推理模型（如 Qwen2.5-14B-Instruct），`./models/main/`，對應 CSV `role=main`。
- **流程**：
  1. 使用 `model_mutex.hold_model("reason")`：載入主模型、批次推理、結束前卸載。
  2. 依 `start_ms` 排序後，對每條 `item` 依序呼叫 `stage2_reason_and_score`：
     - 輸入：`item.text_clean`、前一句／後一句、`item.get_audio_hint()`、`visual_hint=None`、`prompt_config`、`target_language`。
     - Prompt 來自 `model_prompts.csv`（main），替換 `{line}`、`{context}`、`{target_language}`、`{audio_hint}` 等。
  3. `stage2_reason_and_score` 回傳 `Stage2Result`：
     - `meaning_en`：實際存的是**目標語言**的語意（meaning_tl 或 draft_tl / plain_en 的 fallback）。
     - `notes`：內含 **`PACK:{...}`** 字串（見下節）。
     - v1 時含 `need_vision`；v2 時含 `need_multi_frame_vision`。
  4. 將結果寫入 `item.brief_v1`（BriefMeta），其中 `reasons = s2_result.notes`。
  5. 寫入 `work_dir/brief_v1.jsonl`，卸載主模型。

### Stage2 輸出與 PACK 格式

主模型被要求輸出 **JSON only**（無 Markdown），欄位包含（皆可為目標語言）：

- `target_language`：如 zh-TW、ja-JP。
- `tl_instruction`：給 Run E 在地化模型的指令（目標語言）。
- `meaning_tl`：該句簡短語意（目標語言）。
- `draft_tl`：一行草稿翻譯（目標語言），可含 slot `<I1>`、`<I2>` 等供片語填入。
- `idiom_requests`：陣列，每項 `{ "slot", "meaning_tl", "register", "max_len" }`，表示需要由在地化模型建議的片語。
- `ctx_brief`：簡短上下文（目標語言，≤120 字）。
- v1 時需輸出 `need_vision`（boolean）；v2 時需輸出 `need_multi_frame_vision`（boolean）。
- 可選：`plain_en`、`idiom_flag`、`idiom_span`、`referents`、`scene_brief` 等。

解析後會把「整份 JSON（不含 notes）」序列化成字串，前綴 `PACK:` 寫入 `notes`，例如：

`notes = "PACK:{\"target_language\":\"zh-TW\",\"tl_instruction\":\"...\",\"draft_tl\":\"嗯，<I1>。\",\"idiom_requests\":[...],\"ctx_brief\":\"...\",\"main_conf\":0.85}"`

該字串之後會存進 `BriefMeta.reasons`，Run E 用 `parse_pack_from_reasons(reasons)` 還原成 dict 使用。

---

## 五、Run C：單張影像 + Brief v2（條件觸發）

- **觸發條件**：`brief_v1.need_vision === true`（模型判斷需視覺脈絡才能精準翻譯）的項目才會進入 Run C。
- **目的**：對需視覺的項目擷取**一張**影像（字幕時間中點），用視覺模型產生描述，再**重新跑一次 Brief** 得到 v2。
- **模型**：視覺模型（如 Moondream2），`./models/vision/`，使用 `model_mutex.hold_model("vision")` 包住載入與推理。
- **流程**：
  1. 篩出 `target_sub_ids`（need_vision === true）。
  2. 載入視覺模型，對每個 target 擷取中點幀，呼叫 `vision_model.describe_with_grounding(...)`，結果寫入 `item.vision_desc_1`。
  3. 寫入 `work_dir/vision_1frame.jsonl`，卸載視覺模型。
  4. **再次載入主模型**，對同一批 target 呼叫 `run_brief_text(..., version="v2", vision_hint_map={sub_id: item.vision_desc_1})`，得到 `item.brief_v2`（reasons 內同樣含 PACK）。
  5. 寫入 `work_dir/brief_v2.jsonl`，卸載主模型。

---

## 六、Run D：多張影像 + Brief v3（條件觸發）

- **觸發條件**：`brief_v2.need_multi_frame_vision === true`（模型判斷單張視覺不足）的項目才會進入 Run D。
- **目的**：對需多張視覺的項目擷取**多張**影像（預設 3 張，等距），合併描述後再**重新跑一次 Brief** 得到 v3。
- **模型**：同 Run C 的視覺模型；之後同樣會再載入主模型。
- **流程**：
  1. 篩出 `target_sub_ids`（v2 need_multi_frame_vision === true）。
  2. 載入視覺模型，對每個 target 擷取 N 張幀，每張呼叫 `describe_with_grounding`，合併成 `item.vision_desc_n`。
  3. 寫入 `work_dir/vision_multiframe.jsonl`，卸載視覺模型。
  4. **再次載入主模型**，對同一批 target 呼叫 `run_brief_text(..., version="v3", vision_hint_map={sub_id: item.vision_desc_n})`，得到 `item.brief_v3`（reasons 內含 PACK）。
  5. 寫入 `work_dir/brief_v3.jsonl`，卸載主模型。

因此 **Run D 的「主產出」是 brief_v3**，其 `reasons` 中的 PACK 會被 Run E 使用。

---

## 七、Run E：最終翻譯（兩段式，一次只載入一個模型）

Run E 不再用「單一翻譯模型從頭翻到尾」，而是：

1. **Phase 1（LOCAL）**：僅在「至少一句有 idiom_requests」時載入**在地化模型**，對這些句產生 **slot→suggestion** JSON，存成 `suggestions_map[sub_id]`，然後**卸載 LOCAL**。
2. **Phase 2**：對每一句決定最終一行翻譯；若該句**有** idiom_requests，且 Phase 1 有跑，則**僅在此時**載入**主模型**，用 `draft_tl` + suggestions 組裝成一行後卸載 MAIN；若該句**沒有** idiom_requests，則直接用 PACK 的 `draft_tl`，不載入 MAIN。

詞彙表**不注入任何模型**，只在最後對輸出做**輸出端強制替換**（見下）。

### 7.1 前置：解析 PACK 與判斷是否需要組裝

- 依 `start_ms` 排序得到 `sorted_items`。
- 對每筆 `(sub_id, item)`：
  - `best_brief = item.get_best_brief()`（v3 > v2 > v1）。
  - `pack = parse_pack_from_reasons(best_brief.reasons)`（若無 best_brief 或 reasons 中無有效 PACK，則 `pack = None`）。
- 定義 **「有 idiom_requests」**：`_has_idiom_requests(pack)` 為 True 僅當 `pack` 存在且 `pack["idiom_requests"]` 為**非空 list**（避免空 list 或非 list 被誤判）。
- **need_assembly** = 是否存在任一句 `_has_idiom_requests(pack)` 為 True。

### 7.2 Phase 1：LOCAL 模型（僅在 need_assembly 時）

- 僅當 **need_assembly == True** 時：
  1. 使用 `model_mutex.hold_model("translate")` 載入**在地化模型**（`./models/local/`，CSV `role=localization`）。
  2. 對每筆 `(sub_id, item, pack)`：
     - 若 `_has_idiom_requests(pack)`：  
       呼叫 `stage3_suggest_local_phrases(translate_model, requests_json, tl_instruction, target_language, prompt_config)`。  
       - `requests_json = json.dumps(pack["idiom_requests"])`  
       - `tl_instruction` 來自 `pack["tl_instruction"]` 或 `_default_tl_instruction(target_language)`  
       - 回傳為 dict（slot→建議片語），解析失敗則回傳 `{}`。  
       結果存成 `suggestions_map[sub_id]`。
     - 否則：`suggestions_map[sub_id] = {}`。
  3. 離開 `hold_model("translate")` 前**卸載 LOCAL**：`del translate_model`、`gc.collect()`。
- 若 **need_assembly == False**：不載入 LOCAL，`suggestions_map` 全為 `{}`。

### 7.3 Phase 2：決定每句最終翻譯並寫回

- 若 **need_assembly == True**：使用 `model_mutex.hold_model("reason")` 載入**主模型**（與 Run B 相同），供底下「有 idiom_requests 的句子」組裝。
- 若 **need_assembly == False**：不載入主模型（`reason_model` 保持 None）。

對每筆 `(sub_id, item, pack)`（依序）：

1. **若無 pack**（無有效 PACK 或無 brief）：  
   `zh = (best_brief.translation_brief if best_brief else "").strip() or item.text_clean`

2. **若有 pack 但無 idiom_requests**（`not _has_idiom_requests(pack)`）：  
   `zh = (pack.get("draft_tl") or "").strip() or (best_brief.translation_brief if best_brief else "").strip() or item.text_clean`

3. **若有 pack 且有 idiom_requests**：  
   `zh = stage4_assemble_by_main(reason_model, item.text_clean, ctx_brief, draft_tl, suggestions_map.get(sub_id, {}), target_language, prompt_config=reason_assemble_prompt_config, role="main_assemble")`  
   - 內部會先用 `_fill_draft_with_suggestions(draft_tl, suggestions)` 把 `<I1>`、`<I2>` 等替換成 suggestion 的片語。  
   - 若 `reason_model is None`（理論上不應在此分支發生），直接回傳已填好的 draft。  
   - 否則呼叫主模型；system/user 提示詞可由 `model_prompts.csv` 的 `role=main_assemble`（依主模型檔名片段匹配）提供，placeholders：`{target_language}`、`{line_en}`、`{ctx_brief}`、`{draft_prefilled}`、`{suggestions_json}`；若無則 fallback 硬編碼。要求輸出一行最終字幕；回傳為單行字串，若為空則回傳已填好的 draft。

4. 若 3 得到空字串：  
   `zh = (pack.get("draft_tl") or "").strip() or item.text_clean`

5. **詞彙表（輸出端）**：  
   - `zh = apply_glossary_post(zh or "")`（還原 `{{GLOSS:...}}` 佔位符，若存在）。  
   - `zh = force_glossary_replace_output(zh or "", glossary or [], target_language)`：對 enabled 詞彙表條目，在輸出文字上做**整詞替換**（ASCII 整詞、非 ASCII 精確替換），不產生箭頭/列表、不呼叫模型；`target_language` 保留供日後擴充。

6. 寫入 `item.translated_text = zh`。

Phase 2 結束後，離開 `hold_model("reason")` 前卸載主模型：`del reason_model`、`gc.collect()`。

### 7.4 Run E 輸出與一致性

- 呼叫 `verify_subtitle_items_consistency(items, expected_sub_ids, "Run E")`。
- 將結果寫入 `work_dir/final_translations.jsonl`（每行一筆 JSON：`sub_id`, `start_ms`, `end_ms`, `translated_text`）。
- App 層再依 `items` 的 `translated_text` 寫回 SRT 並輸出檔案。

---

## 八、詞彙表（Glossary）

- **不注入任何模型**：Run B/D 的 main、Run E 的 LOCAL/MAIN 的 prompt 中**都不**包含詞彙表列表或術語表標籤。
- **僅在輸出端處理**（Run E Phase 2 每句得到 `zh` 之後）：
  1. **apply_glossary_post(zh)**：把文字中可能存在的 `{{GLOSS:xxx}}` 還原成 `xxx`（目前因未對輸入做 glossary 預替換，多為無操作）。
  2. **force_glossary_replace_output(zh, glossary, target_language)**：對詞彙表中 enabled 的條目，在 `zh` 上做強制替換（來源詞→目標詞），支援可變目標語言；邏輯為整詞（ASCII）或精確字串（非 ASCII），不產生箭頭/列表。

---

## 九、執行順序摘要（App 層）

1. **Run A**：`run_audio(items, video_path, work_dir, cfg, ...)` → 填 `audio_meta`。
2. **Run B**：`run_brief_text(items, work_dir, cfg, prompt_config, version="v1", ..., target_language)` → 填 `brief_v1`，寫 `brief_v1.jsonl`。
3. **Run C**（條件）：對 need_vision === true 的項目 `run_vision_single(...)` → 填 `vision_desc_1`；再 `run_brief_text(..., version="v2", vision_hint_map=...)` → 填 `brief_v2`，寫 `brief_v2.jsonl`。
4. **Run D**（條件）：對 v2 need_multi_frame_vision === true 的項目 `run_vision_multi(...)` → 填 `vision_desc_n`；再 `run_brief_text(..., version="v3", vision_hint_map=...)` → 填 `brief_v3`，寫 `brief_v3.jsonl`。
5. **Run E**：`run_final_translate(items, work_dir, cfg, glossary, target_language, prompt_config, reason_prompt_config=..., reason_assemble_prompt_config=..., ...)` → Phase 1（可選）LOCAL 建議片語 → Phase 2 每句組裝或直接用 draft_tl → 輸出端詞彙表替換 → 填 `translated_text`，寫 `final_translations.jsonl`。

全程遵守**一次只載入一個模型**，且 Run E 先完全卸載 LOCAL 後才可能載入 MAIN。

---

## 十、相關檔案

| 檔案 | 說明 |
|------|------|
| `src/subtitle_item.py` | SubtitleItem、BriefMeta、AudioMeta、get_best_brief |
| `src/pipeline_runs.py` | run_audio、run_brief_text、run_vision_single、run_vision_multi、run_final_translate |
| `src/pipeline.py` | stage2_reason_and_score、stage3_suggest_local_phrases、stage4_assemble_by_main、parse_pack_from_reasons、_fill_draft_with_suggestions、_default_tl_instruction |
| `src/glossary.py` | apply_glossary_post、force_glossary_replace_output |
| `src/model_lock.py` | get_model_mutex、hold_model、set_current_model_type |
| `model_prompts.csv` | Run A～D：main / vision 等 prompt 模板（依 model_name 匹配） |
| `model_prompts_run_f.csv` | **Run F 專用**：main_group_translate、main_assemble、local_polish、localization；優先於 model_prompts.csv 載入 |

以上即為目前**完整翻譯流程**的詳細敘述。
