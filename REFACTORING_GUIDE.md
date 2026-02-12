# 重構指南：統一 SubtitleItem 結構與模型互斥

## 已完成的部分

### 1. 核心資料結構 (`src/subtitle_item.py`)
- ✅ `SubtitleItem`：統一的字幕項目資料結構
- ✅ `AudioMeta`：音訊分析結果
- ✅ `BriefMeta`：Brief 生成結果
- ✅ `generate_sub_id()`：生成固定且不可變的 sub_id
- ✅ `verify_subtitle_items_consistency()`：一致性檢查函式

### 2. 模型互斥鎖 (`src/model_lock.py`)
- ✅ `ModelMutex`：全域單例，確保任何時刻只載入一個模型
- ✅ `hold_model(model_type)`：context manager，將「載入 → 推理 → 卸載」整段包在鎖內，併發時後者會阻塞直到前者卸載
- ✅ `ensure_model_unloaded()`：強制清理函式

### 3. 重構後的 Run 函式 (`src/pipeline_runs.py`)
- ✅ `run_audio()`：Run A，使用 SubtitleItem 和 sub_id 對齊
- ✅ `run_brief_text()`：Run B/C/D，使用 SubtitleItem 和 sub_id 對齊
- ✅ `run_vision_single()`：Run C，單張影像分析
- ✅ `run_vision_multi()`：Run D，多張影像分析
- ✅ `run_final_translate()`：Run E，最終翻譯

### 4. 共享工具模組
- ✅ `src/model_path_utils.py`：統一的模型路徑解析邏輯（與 app.py 共享）
- ✅ `src/jsonl_compat.py`：JSONL 格式相容性工具（舊格式 idx ↔ 新格式 sub_id）

## 關鍵改進

### 1. 資料不錯位
- **舊方式**：使用 `list[index]` 對齊，容易出錯
- **新方式**：使用 `dict[sub_id, SubtitleItem]` 對齊，sub_id 是固定且不可變的

### 2. 模型互斥
- **舊方式**：ModelManager 手動管理，容易遺漏
- **新方式**：使用 `model_mutex.hold_model(model_type)` 將載入、推理、卸載整段包在鎖內，自動保護；併發時後者會阻塞直到前者完整卸載

### 3. 一致性檢查
- 每個 run 結束後自動驗證 sub_id 集合
- 發現不一致立即 raise ValueError 並打印詳細資訊

## 使用範例

```python
from src.subtitle_item import create_subtitle_items_from_srt
from src.pipeline_runs import run_audio, run_brief_text
from src.model_lock import get_model_mutex

# 1. 從 SRT 創建 SubtitleItem 字典
items = create_subtitle_items_from_srt(subs)

# 2. Run A: 音訊分析
items = run_audio(
    items=items,
    video_path=video_path,
    work_dir=work_dir,
    cfg=cfg,
    log_lines=log_lines,
    progress_callback=progress_callback,
)

# 3. Run B: Brief v1 生成
items = run_brief_text(
    items=items,
    work_dir=work_dir,
    cfg=cfg,
    prompt_config=reason_prompt_config,
    version="v1",
    log_lines=log_lines,
    progress_callback=progress_callback,
    batch_size=batch_size,
)
```

## 整合與相容性

### 1. 模型路徑解析整合
- ✅ `app.py` 和 `pipeline_runs.py` 都使用 `src/model_path_utils.py` 中的共享函數
- ✅ 統一的 shard 模型選擇邏輯（優先選擇 `*-00001-of-*.gguf`）
- ✅ 支援目錄佈局和 legacy 檔名佈局

### 2. JSONL 格式相容性
- ✅ `src/jsonl_compat.py` 提供舊格式（`idx`）與新格式（`sub_id`）的雙向轉換
- ✅ 自動檢測 JSONL 格式並進行適當的對齊
- ✅ 所有 README（英文、繁體中文、簡體中文、日文、西班牙文）已更新說明格式相容性

### 3. 文檔一致性
- ✅ 所有 README 已更新，說明 JSONL 格式相容性
- ✅ `final_translations.jsonl` 已加入文檔
- ✅ CSV (`model_prompts.csv`) 與程式邏輯一致

## 待完成的工作（可選）

1. **遷移 app.py 到新管線**（可選）
   - 將 `translate_ui()` 改為使用新的 run 函式
   - 使用 `SubtitleItem` 和 `sub_id` 對齊
   - 使用 `jsonl_compat.py` 載入舊格式的 JSONL

2. **測試與驗證**
   - 確保所有 run 的一致性檢查通過
   - 驗證模型互斥鎖正常工作
   - 測試續跑功能（從 JSONL 載入，支援舊格式和新格式）

## 注意事項

- **sub_id 生成**：目前使用 `hash(start_ms, end_ms, text_raw)`，確保唯一性
- **模型載入**：每個 run 函式內部載入和釋放模型，使用互斥鎖保護
- **批次處理**：批次大小可調，但模型推理必須在互斥鎖內順序進行
- **錯誤處理**：任何不一致都會立即 raise，不會靜默失敗
