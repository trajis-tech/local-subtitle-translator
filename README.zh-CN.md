# 本地字幕翻译器（便携式、离线模型）

本项目使用 **llama-cpp-python** 运行本地字幕翻译流程。
设计为**便携式**：所有依赖项都在此文件夹内。

✅ **离线优先**：先执行 `install.bat`（Windows）或 `./install.sh`（Linux/macOS）**一次**（需网络）下载并安装所有依赖；之后使用 `start.bat` 或 `./start.sh` **离线启动**即可。

❗ **模型不会自动下载。** 您需要手动下载 GGUF 文件并放入 `./models/`。

---

## 翻译流程概览

翻译流程分为 **6 个执行阶段**（A→B→C/D→E→F），依序执行。**单一 brief**：目前仅一份 `./work/brief.jsonl`，每阶段更新；在 C/D/E 前会先复制为 `brief_v1.jsonl` / `brief_v2.jsonl` / `brief_v3.jsonl`（snapshot）。

- **Run A**：音频情绪/语气分析（所有字幕行）
  - 从影片中提取每条字幕对应的音频片段
  - 分析情绪、语气、强度、说话方式
  - 结果保存至 `./work/audio_tags.jsonl`
  - **注意**：音频模型已**预先打包**且与代码**紧密耦合**。**请勿修改或替换。**

- **Run B**：主模型产生翻译描述（所有字幕行）
  - 使用主推理模型（GGUF、llama-cpp-python）产生翻译指导。
  - 输入：英文字幕 + 前后各一句上下文 + 音频标签（皆英文）。
  - 输出：JSON — **全部为英文**（语言中立）：`meaning_tl`、`draft_tl`、`tl_instruction`、`idiom_requests`、`ctx_brief`、**`referents`**、**`tone_note`**、**`scene_brief`**，以及可选 **`disambiguation_note`**、**`rewrite_guidance`**。**每阶段只输出一个 need**：v1 仅输出 **`need_vision`**（布尔）。可选 `transliteration_requests`、`omit_sfx`；`reasons`（含供 Run F 使用的 PACK）。
  - 结果保存至 `./work/brief.jsonl`（单一当前 brief）。

- **Run C**：单张影像 fallback（选用、条件触发）
  - 当 **`need_vision === true`**（来自当前 brief）时触发。
  - 更新前：复制 `brief.jsonl` → `brief_v1.jsonl`。在字幕时间区间正中间提取一张影像；视觉模型分析场景/人物/动作。
  - 重新产生 brief（带单张视觉提示）→ 更新 `./work/brief.jsonl`。**每阶段只输出一个 need**：v2 仅输出 **`need_multi_frame_vision`**（布尔）每笔。

- **Run D**：多张影像 fallback（选用、条件触发）
  - 当 **`need_multi_frame_vision === true`**（来自当前 brief）时触发。
  - 更新前：复制 `brief.jsonl` → `brief_v2.jsonl`。在字幕时间区间内等距提取 N 张影像（可设定，默认：3）；视觉模型分析并合并描述。
  - 重新产生 brief（带多张影像提示）→ 更新 `./work/brief.jsonl`。**每阶段只输出一个 need**：v3 仅输出 **`need_more_context`**（布尔）；供 Run E（上下文扩充）使用。

- **Run E**：上下文扩充（条件）
  - 更新前：复制 `brief.jsonl` → `brief_v3.jsonl`。对 **`need_more_context === true`** 的项目以 **前 3／后 3 句** 再跑 stage2 更新 brief，写回 `./work/brief.jsonl`。仅更新 brief，不产出目标语。

- **Run F**：最终翻译（所有字幕行）
  - Run F 读取**当前 brief**（`brief.jsonl`，已由 E 更新）与 PACK 产出整集目标语字幕并写入 SRT。**配置**：`config.PipelineConfig.run_e_scheme`（默认 `"full"`）。**UI**：Run F scheme 下拉选单。**执行**：`pipeline_runs.run_final_translate()`；输出：`items.translated_text` 与 `work_dir/final_translations.jsonl`。**SRT 写回**：`app.py` 依 `(round(start_ms), round(end_ms))` 对齐，保留原文 `<i>` 标签。
  - **一次只载入一个模型**：同一时间仅载入 MAIN 或 LOCAL 其一；词汇表仅在输出端套用。
  - **稳健性**：所有 chat 调用经统一入口 **chat_dispatch**。heavy 请求可改以 one-shot 子进程执行；失败时 fallback 进程内。

### Run F 方案（依主模型／在地化模型强弱选择）

在界面中从 **Run F scheme** 下拉选单选择方案。选项：`full` | `main_led` | `local_led` | `draft_first`（无效时 fallback 为 `full`）。

| Scheme | 选择时机（主／地） | Phase1（初稿来源） | Phase2（润饰） |
|--------|---------------------|------------------------|-----------------|
| **Full** | 主强、地强 | MAIN 群组翻译 → draft_map | LOCAL 润饰 |
| **MAIN-led** | 主强、地弱 | MAIN 群组翻译 → draft_map | 无 |
| **LOCAL-led** | 主弱、地强 | PACK draft_tl → draft_map；可选 LOCAL 填 idiom slot | LOCAL 润饰 |
| **Draft-first** | 主弱、地弱 | PACK draft_tl → draft_map；可选 LOCAL 填 idiom slot | 无 |

- **Full**：Phase 1 — 载入 MAIN（reason），建立句子群组，依 `group_translate_max_segments`（默认 4）切 chunk；每 chunk 调用 `stage_main_group_translate`；**可部分采用 MAIN 输出**（依 **id** 对齐；缺的句用 PACK draft_tl 或 en_text）。Phase 2 — 载入 LOCAL，依 chunk 执行 `local_polish`（`local_polish_chunk_size` 默认 60）；仅接受请求内 key 且通过长度／污染检查的结果写入 draft_map。最终：词汇表 + strip_punctuation → `translated_text`。
- **MAIN-led**：Phase 1 — 同 Full（MAIN 群组翻译 → draft_map）。Phase 2 — **略过**。最终：词汇表 + strip_punctuation。
- **LOCAL-led**：不载入 MAIN。`draft_map = _build_draft_map_from_pack(...)`。若有项目含 `idiom_requests`，载入 LOCAL，逐行调用 `stage3_suggest_local_phrases`，以 `_fill_draft_with_suggestions` 填入 slot 并更新 draft_map。再载入 LOCAL 对全部行执行 `local_polish`。最终：词汇表 + strip_punctuation。
- **Draft-first**：不载入 MAIN。仅以 PACK 建立 draft_map；若有 idiom_requests 则载入 LOCAL 取得建议并填入；**不**润饰。弱在地化模型使用 **STRICT** prompt 与 raw_decode 后备解析，避免输出格式错误。

**对齐**：Run F 对齐一律依 **sub_id** 与**时间戳 (start_ms, end_ms)**，不依列表索引，以降低错位风险。

**语言政策（阶段间不混用）**：
- **Run A–E**：所有 prompt 与所有模型输出皆**仅限英文**。Run A（音频）、Run B/C/D（brief）、Run E（上下文扩充）不接收也不产出目标语；brief 为语言中立的英文，供 Run F 以单一英文接口翻译。
- **Run F**：所有**指令（prompt）为英文**；主模型与在地化模型的输入为**英文**（片段、tl_instruction、上下文）。仅**模型输出**（segment_texts[].text、润饰后句子、短语建议）为**目标语言**。
- **强制**：Run B brief 输出会经净化（例如 `tl_instruction` 必须仅英文）。Run F 使用 `_tl_instruction_for_run_e()`，确保翻译阶段始终依正确目标语系输出。

**Prompt 角色**（`model_prompts.csv`）：MAIN（main_group_translate）专注**原文（SOURCE）**；强在地化（如 Llama-Breeze2-8B）专注**目标语**、自然化／润饰；弱在地化（如 Breeze-7B、custom-localization）使用 **STRICT** 格式；解析失败时以 `raw_decode` 取第一个 `{...}`。

**音译**：需在目标语音译的人名或专有名词由**在地化模型**负责；**主模型**（Run B）在 PACK 中以 `transliteration_requests`（字符串数组）提出。Run F Phase2（local_polish）会收到这些词并在 polish prompt 中附加「Transliterate (音译) in target language for these terms: …」，由 LOCAL 输出音译形式。

**CC／状声词**：**主模型**（Run B）筛掉音效与状声词（如 `[laughter]`、`[sigh]`、`*gasps*`）。可设 `omit_sfx: true` 且 `draft_tl` 为空（仅 SFX 的句）；对白＋SFX 时 `draft_tl` 仅含对白。Run F 在建好 draft_map 后套用 omit_sfx，该类句输出为空。

**Run F 配置**（`config.py`）：`run_e_scheme`（UI：Run F scheme）、`group_translate_max_segments`（默认 4）、`local_polish_chunk_size`（默认 60）、`strip_punctuation`、`strip_punctuation_keep_decimal`、`strip_punctuation_keep_acronym`。

### 主要特性

- **单一模型加载**：任一时间只加载一个模型（音频、推理、视觉或翻译）
- **可续跑**：每个 run 将中间结果保存至 `./work/`（JSONL 格式）
- **错误容错**：如果视觉/音频失败，流程会使用最佳可用 brief 继续
- **进度追踪**：进度条显示当前步骤和完成百分比

---

## 视频输入（FFmpeg 注意事项）

Gradio 内置的 **Video** 组件需要外部 **`ffmpeg`** 可执行文件才能进行服务器端处理。
如果没有 `ffmpeg`，可能会出现 **"Executable 'ffmpeg' not found"** 错误。

为保持本项目**完全便携式**（无需系统级安装），本项目使用 **File** 输入来处理视频。

- 您需要视频文件用于 **Run A（音频）** 和 **Run C/D（视觉）**。
- 使用 OpenCV (opencv-python) 抓取视觉画面，使用 ffmpeg 提取音频片段。
- **ffmpeg**：**Windows** – `install.bat` 若 PATH 中无 ffmpeg 会自动下载到 `runtime\ffmpeg`。**Linux / macOS** – `install.sh` 仅检查 PATH 中是否有 ffmpeg；若缺失请手动安装并参阅 **FFMPEG_INSTALL.md**。

若您使用的是仍使用 Video 组件的旧版 zip，请更新至最新版，或安装 FFmpeg 并添加到 PATH。

## 安装与启动（离线优先）

本项目采**离线优先**：先执行 **install** 一次（需网络），之后使用 **start** 随时离线启动。

**⚠️ 重要 - NVIDIA 显卡用户：**
- **请先安装 CUDA Toolkit 12.9**（或 12.x）**再执行** `install.bat` / `install.sh`
- 下载：https://developer.nvidia.com/cuda-downloads
- 预构建的 llama-cpp-python 轮文件需要先安装 CUDA 才能启用 GPU 加速

1. 将此文件夹解压到任意位置（例如：`G:\local-subtitle-translator`）。

2. **安装（一次、需网络）** — 下载并安装所有依赖与工具：
   - **Windows**：双击 `install.bat` → 便携式 Python、venv、所有 Python 依赖（含音频 Run A、视频），若有 GPU 可选安装 CUDA PyTorch，若 PATH 无 ffmpeg 则下载至 `runtime\ffmpeg`，Run A 音频模型下载至 `models\audio`，预构建 llama-cpp-python 轮文件、config、BOM。GGUF 模型需手动下载（见下方）。
     - **安装后预计占用空间**：约 6-8 GB（仅 CPU：约 4-5 GB；含 CUDA PyTorch + ffmpeg：约 6-8 GB）
   - **Linux / macOS**：执行 `./install.sh`（同上：.venv、依赖、音频模型、预构建 llama-cpp-python 轮文件、config、BOM）。必要时：`chmod +x install.sh`
     - **安装后预计占用空间**：约 4-5 GB（不含系统 Python 与 ffmpeg）

3. **启动（离线）** — 不下载、不连网：
   - **Windows**：双击 `start.bat` → 检查 .venv 与模型文件后启动 UI。
   - **Linux / macOS**：执行 `./start.sh`。必要时：`chmod +x start.sh`

- **解除安装**：执行 `uninstall.bat`（Windows）或 `./uninstall.sh`（Linux/macOS）可删除此文件夹内的运行时环境、venv 与缓存。必要时：`chmod +x uninstall.sh`

**显卡支持：**

- **NVIDIA 显卡**：支持 CUDA 12.x（建议 CUDA 12.9；RTX 20/30/40/50 系列、GTX 16 系列及更新型号）
- **AMD 显卡**：ROCm 支持（实验性，需手动设置）
- **Intel Arc 显卡**：oneAPI 支持（实验性，需手动设置）
- **CPU**：针对 Intel CPU 优化（无需 AVX-512 指令集），可在所有现代 x86-64 处理器上运行

**选用的 – 仅安装音频依赖（Linux / macOS）：**

- 执行 `install.bat` 或 `install.sh` 已会安装 Run A（音频）依赖。仅在需要单独重装音频依赖（torch、transformers、soundfile、scipy）而不做完整安装时，可使用 `./scripts/install_audio_deps.sh`。需 Python 3，可选已启用的 `.venv`。

所有内容都保留在此文件夹内（便携式/隔离）。

---

## 模型兼容性与目录结构（必需）

本应用使用的**文字与视觉**模型必须为 **GGUF** 格式，且与 **llama-cpp-python** 兼容。模型文件由您自行准备，应用不会自动下载。

创建并使用以下目录结构：

```
models/
  main/     ← 主推理模型（Run B）；一个或多个 .gguf 文件
  local/    ← 本地化/翻译模型（Run E）；一个或多个 .gguf 文件
  vision/   ← 可选视觉模型（Run C/D）；主 .gguf + mmproj .gguf
  audio/    ← Run A 音频模型（由安装脚本或首次运行时下载）
```

### 兼容性

- **主模型与本地化模型**：任何可在 llama-cpp-python 下运行的 **GGUF** 模型（具 chat 模板的 instruct/chat 模型）。将文件放入 `./models/main/` 与 `./models/local/`。若量化为**分片**（多个 `.gguf`），请下载**全部分片**并放在同一文件夹。
- **视觉模型（可选）**：任何 llama-cpp-python 支持的 **GGUF 视觉**模型（主模型 + mmproj）。将两个文件放入 `./models/vision/`。应用会按文件名自动检测类型。您可在 `config.json` 的 `vision.text_model` 与 `vision.mmproj_model` 指定确切文件名。
- **音频（Run A）**：Run A 情绪模型于首次运行时从 Hugging Face Hub 下载（无本地 GGUF）。使用 Transformers `audio-classification`；依赖：`torch`、`transformers`、`soundfile`、`scipy`。

### 参数与量化（通用建议）

- **量化**：较轻量化（如 **Q4_K_M**）更省 VRAM、更快；较重量化（**Q5_K_M**、**Q6_K**、**Q8_0**）质量更好但需更多 VRAM 与磁盘。请按 GPU/RAM 选择。
- **模型大小**：参数量越大（如 14B、7B）所需 VRAM/RAM 越多。模型**一次只加载一个**，故 VRAM 以**单一最大模型**为准。
- **上下文大小**：较大的 `n_ctx_*`（如 8192）有利于长上下文，但会增加 VRAM（KV 缓存）。若发生 OOM，可降低 `n_ctx_*` 或 `n_gpu_layers_*`。

### config.json 建议起点（按硬件调整）

- **16 GB VRAM**：`n_ctx_reason=8192`, `n_ctx_translate=4096`, `n_gpu_layers_reason=60`, `n_gpu_layers_translate=60`
- **12 GB VRAM**：`n_ctx_reason=4096`, `n_ctx_translate=2048`, `n_gpu_layers_reason=50`, `n_gpu_layers_translate=50`
- **8 GB VRAM**：`n_ctx_reason=2048`, `n_ctx_translate=2048`, `n_gpu_layers_reason=35`, `n_gpu_layers_translate=35`
- **仅 CPU / 低 RAM**：建议使用 Q4_K_M（或更轻）与较小上下文；减少 `n_gpu_layers_*` 或设为 0 以全 CPU 运行。

---

## config.json

执行 `install.bat`（或 `install.sh`）时，若没有 `config.json`，会运行 `scripts/plan_models.py` 创建一个最佳努力的 `config.json`。`start.bat` / `start.sh` 不会创建 config，仅负责启动程序。
如果选择不同的量化文件，请编辑它。

关键字段：

- `models_dir`：默认 `models`
- `reason_dir`：主模型目录（默认：`./models/main/`）
- `translate_dir`：本地化模型目录（默认：`./models/local/`）
- `vision_text_dir` / `vision_mmproj_dir`：视觉模型目录（默认：`./models/vision/`）
- `audio.model_id_or_path`：Run A 使用的 Hugging Face 模型 id（默认：`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`）
- `audio.enabled`：启用/停用音频分析（默认：`true`）
- `audio.device`：`"auto"`（有 CUDA 则用）、`"cuda"` 或 `"cpu"`（默认：`auto`）
- `audio.device_index`：使用 CUDA 时的 GPU 索引（默认：`0`）
- `audio.batch_size`：情绪推理的批次大小；较大可提高 GPU 利用率（默认：`16`）
- `vision.enabled`：可选视觉 fallback
- `pipeline.n_frames`：多张影像视觉的帧数（默认：`3`）
- `pipeline.work_dir`：中间结果目录（默认：`./work/`）
- `pipeline.run_e_scheme`：Run F 方案 — `"full"` | `"main_led"` | `"local_led"` | `"draft_first"`（默认：`"full"`）。详见上方 **Run F 方案**。
- `pipeline.local_polish_chunk_size`：Run F local_polish 每批行数（默认：`60`）
- `pipeline.group_translate_max_segments`：Run F 主翻译每子群组最大段数（默认：`4`）
- `pipeline.isolate_heavy_requests`：为 `true` 时，过重的请求（超过 token/行数/段数阈值）改以 one-shot 子进程执行，避免 OOM 拖死主程序（默认：`true`）
- `pipeline.isolate_heavy_timeout_sec`：隔离子进程超时秒数（默认：`600`）
- `pipeline.strip_punctuation`：为 `true` 时，在 Run F 最终输出时去除标点（默认：`true`）
- `pipeline.strip_punctuation_keep_decimal`：为 `true` 时，保护小数如 `3.14` 不被拆开（默认：`true`）
- `pipeline.strip_punctuation_keep_acronym`：为 `true` 时，保护缩写如 `U.S.` 不被拆开（默认：`true`）

---

## 工作目录（中间结果）

所有中间结果都保存至 `./work/` 目录（JSONL 格式）：

- `audio_tags.jsonl` - Run A 结果（音频情绪/语气分析）
- `brief.jsonl` - 当前 brief（Run B 写入；C/D/E 更新；Run F 读取）
- `brief_v1.jsonl` - Run C 前 snapshot（更新前复制）
- `brief_v2.jsonl` - Run D 前 snapshot
- `brief_v3.jsonl` - Run E 前 snapshot
- `vision_1frame.jsonl` - Run C 结果（单张影像视觉分析）
- `vision_multiframe.jsonl` - Run D 结果（多张影像视觉分析）
- `final_translations.jsonl` - Run F 结果（最终翻译文字，新格式）

**JSONL 格式兼容性：**

流程支持**旧格式**（使用 `idx` 对齐）和**新格式**（使用 `sub_id` 对齐）：

- **旧格式**：使用 `idx`（整数索引）识别字幕行
  - 示例：`{"idx": 0, "start_ms": 1000, "end_ms": 2000, ...}`
- **新格式**：使用 `sub_id`（基于 hash 的唯一标识符）确保数据对齐
  - 示例：`{"sub_id": "a1b2c3d4_0", "start_ms": 1000, "end_ms": 2000, ...}`
  - `sub_id` 由 `hash(start_ms, end_ms, text_raw)` 生成，确保跨 run 的一致性

流程会自动检测格式并在需要时进行转换。新的 run 将使用 `sub_id` 格式以确保更好的数据完整性。

**续跑功能**：如果 JSONL 文件存在且条目数量正确，流程会自动加载并跳过该 run。流程支持从旧格式（`idx`）和新格式（`sub_id`）续跑。

**手动续跑**：您可以删除特定 JSONL 文件以仅重新执行那些步骤。

---

## UI 使用方式

1. **上传文件**：视频（MKV/MP4）和 SRT（英文字幕）
2. **选择执行模式**：`all`（A→B→(C/D)→E→F，默认）| **A**（音频）| **B**（brief）| **C**（单帧视觉）| **D**（多帧视觉）| **E**（上下文扩充）| **F**（翻译）
3. **Run F scheme**（下拉选单）：依主模型／在地化模型强弱选择 — **Full** | **MAIN-led** | **LOCAL-led** | **Draft-first**。详见上方 **Run F 方案**。
4. **选用备援**（界面勾选）：
   - **启用视觉备援（Run C/D）**：勾选且 brief 为 **need_vision**／**need_multi_frame_vision** 时执行单帧（C）或多帧（D）视觉并更新 brief。需本地 GGUF 视觉模型。
   - **启用更多上下文备援（Run E）**：勾选时，具 **need_more_context** 的项目会在 Run F 以前 3／后 3 句扩充并更新 brief。建议启用。
   - **Max frames per subtitle (Run D)**：多帧视觉的帧数（默认 1–4）；**Frame offsets**：取样位置。
5. **点击「🚀 Translate」**并监控进度
6. **下载**翻译完成的 SRT 文件
7. **重置**：点击 **「Reset」** 可清空所有输入、输出与日志并恢复默认值，以便开始新的翻译工作

**界面说明**：日志区以**最新条目在上方**显示。`model_prompts.csv` 以 UTF-8（含 BOM）读写；`start.bat` / `start.sh` 启动时会执行 `ensure_csv_bom.py` 以保持文件编码正确。

---

## 自定义模型提示词（model_prompts.csv）

翻译流程使用 `model_prompts.csv` 中定义的提示词。每个模型的提示词会根据**模型文件名**自动匹配（不区分大小写的子字符串匹配）。文件应为 **UTF-8（含 BOM）**；`start.bat` 与 `start.sh` 启动时会执行 `scripts/ensure_csv_bom.py` 以确保编码正确。

### 模型官方 prompt 对齐

提示词依各模型家族的**官方**聊天格式与建议设计，以维持行为可预测、兼容：

- **Qwen2.5 (ChatML)**：System 角色 + User 角色；JSON Mode 做结构化输出。模板使用 `chat_format=chatml` 与严格的「仅输出有效 JSON、无 markdown」指示，符合官方 Qwen 用法。
- **Gemma 2（如 TranslateGemma）**：**无 system 角色**；所有指令在第一个 user turn。后端在 `chat_format=gemma` 时会将 system 内容合并进 user 消息，模型只看到单一 user turn。
- **Mistral / Llama 2（如 Breeze、Llama-Breeze2）**：`[INST]` 风格；system prompt 会接在第一个 `[INST]` 前。用于 `local_polish`、`localization` 角色，必要时采 STRICT JSON 输出。
- **Vision（Moondream、LLaVA）**：提示词在程序内依 handler 套用；聊天格式依视觉模型文件名自动检测。输出一律为**英文**视觉描述（不产出字幕）。

CSV 的 **notes** 列会标注该角色属「Run A~D 全英文」或「Run F：仅输出为目标语」，自定义行请维持相同语言边界。

### 模型名称匹配

- **工作原理**：应用程序提取模型文件名（例如，`my-main-model-q5_k_m.gguf`）并与 CSV `model_name` 列匹配。
- **匹配规则**：如果文件名**包含** CSV `model_name`（不区分大小写），则匹配成功。
  - 示例：`my-main-model-q5_k_m.gguf` 匹配 `my-main-model`
  - 示例：`my-local-model-00001-of-00002.gguf` 匹配 `my-local-model`
- **填写内容**：使用出现在您模型文件名中的**唯一子字符串**。通常基础模型名称（不含量化后缀）即可。

### CSV 列指南

| 列 | 说明 | 示例 |
|--------|-------------|---------|
| `model_name` | 在文件名中匹配的子字符串（不区分大小写） | `my-main-model` |
| `role` | `main`（Run B）、`main_group_translate`（Run F 群组翻译）、`main_assemble`（Run F 旧版组装）、`localization`（Run F 短语建议）、`local_polish`（Run F 批次顺口化）或 `vision`（Run C/D） | `localization` |
| `source_language` | 输入语言（通常是 `English`） | `English` |
| `target_language` | 输出语言（语言代码：`en`、`zh-TW`、`zh-CN`、`ja-JP`、`es-ES`） | `zh-TW` |
| `chat_format` | 模型的聊天模板（`chatml`、`llama-3`、`mistral-instruct`、`moondream`） | `chatml` |
| `system_prompt_template` | 系统提示词（角色定义） | 见下方示例 |
| `user_prompt_template` | 用户提示词（含占位符） | 见下方示例 |
| `notes` | 说明（英文） | `Localization model for Traditional Chinese (Taiwan)` |

### 占位符

在 `user_prompt_template` 中使用这些占位符：

**Run B（main）占位符：**
- `{line}` → 当前英文字幕行
- `{context}` → 完整上下文（Prev-1、Current、Next-1、Prev-More、Next-More、Visual Hint）

**Run F（localization）占位符：**
- `{tl_instruction}`、`{requests_json}`、`{target_language}`（片语建议）

**Run F（main_assemble）** – 旧版 Stage4 单行组装：
- `{target_language}`、`{line_en}`、`{ctx_brief}`、`{draft_prefilled}`、`{suggestions_json}`

**Run C/D（vision）占位符：**
- `{line}` → 当前英文字幕行

### 提示词风格：Base vs Instruct 模型

#### Base 模型（非 Instruct）
- **特征**：较简单、直接的提示词，无结构化指令格式
- **使用时机**：您的模型是基础/完成模型（未针对指令进行微调）
- **风格**：直接问题或简单任务描述
- **示例**（Run B）：
  ```
  Analyze this subtitle line and explain what it really means in plain English.
  
  Subtitle: {line}
  Context: {context}
  
  Explain the meaning, including any idioms, jokes, tone, or implied meaning.
  ```

#### Instruct 模型
- **特征**：结构化指令格式，含编号规则和明确任务定义
- **使用时机**：您的模型经过指令微调（Instruct、Chat 等）
- **风格**：结构化，含规则、编号步骤、明确输入/输出定义
- **示例**（Run B）：
  ```
  You are stage 2 (reasoning) in a multi-stage subtitle translation pipeline.
  - Input: one English subtitle line plus nearby context.
  - Output: ENGLISH ONLY: a clear, unambiguous explanation...
  - Do NOT translate to any target language here.
  
  Subtitle line: {line}
  Context (previous/next lines): {context}
  ```

### CSV 中的示例

CSV 包含每个角色的示例行：

1. **`(custom-main-base)`** - Run B 的 Base 模型示例
2. **`(custom-main-instruct)`** - Run B 的 Instruct 模型示例
3. **`(custom-localization-base)`** - Run E 的 Base 模型示例
4. **`(custom-localization-instruct)`** - Run E 的 Instruct 模型示例
5. **`(custom-vision-base)`** - Vision 的 Base 模型示例
6. **`(custom-vision-instruct)`** - Vision 的 Instruct 模型示例

### 添加您自己的模型

1. **复制示例行**（例如，`(custom-main-instruct)`）
2. **更改 `model_name`** 以匹配您的模型文件名子字符串
3. **设置 `role`**（`main`、`localization` 或 `vision`）
4. **设置 `target_language`** 为以下语言代码之一：
   - `en` - 英语（适用于 Run B main 模型）
   - `zh-TW` - 繁体中文（台湾）
   - `zh-CN` - 简体中文（大陆）
   - `ja-JP` - 日语
   - `es-ES` - 西班牙语
   - 或其他 IETF 语言代码（根据需要）
5. **设置 `chat_format`** 以匹配您模型的聊天模板：
   - `chatml` - 许多现代 instruct/chat 模型
   - `llama-3` - Llama 3 模型
   - `mistral-instruct` - Mistral 模型
   - `moondream` - 部分视觉模型
6. **编写 `system_prompt_template`**（角色定义，通常 1-2 句）
   - 对于本地化模型：如果您想在提示词中泛泛提及目标语言，可使用 `[target_language]` 作为占位符
7. **编写 `user_prompt_template`**（含占位符的任务）
   - Base 模型使用 Base 风格
   - 指令微调模型使用 Instruct 风格
   - 对于本地化模型：在提示词文本中将 `[target_language]` 替换为实际的语言代码（例如，`zh-TW`、`ja-JP`）
8. **填写 `notes`**（英文说明）

### 重要注意事项

- **语言边界**：**Run A–D**（音频、brief v1/v2/v3、视觉）：prompt 与模型输出必须**仅限英文**。**Run E**（main_group_translate、local_polish、localization、main_assemble）：prompt 为**英文**；仅**输出**（翻译句、短语建议）为目标语。请勿在 Run E 的 prompt 中写入目标语指令（例如中文或日文）— 请使用英文（例如「Output ONLY the translated subtitle in the target language (locale: zh-TW).」），以免 prompt 与输出语言混用。
- **提示词语言**：Run B/C/D 使用仅限英文的 prompt；Run E 使用英文指令，预期模型输出为目标语。
- **聊天格式**：必须匹配您模型的聊天模板。错误的格式可能导致输出不佳或错误。
  - **视觉模型**：聊天格式由 `LocalVisionModel` 根据模型文件名自动检测。CSV 的 `chat_format` 字段仅供说明。
- **占位符**：始终使用确切的占位符名称（`{line}`、`{context}`、`{target_language}` 等）。它们会自动替换。
- **Run B/C/D 输出（brief）**：必须请求 JSON 含 `target_language`、`tl_instruction`、`meaning_tl`、`draft_tl`、`idiom_requests`、`ctx_brief`、referents、tone_note、scene_brief — **全部为英文**（语言中立）。**每阶段只输出一个 need**：**v1** 仅输出 **`need_vision`**；**v2** 仅输出 **`need_multi_frame_vision`**；**v3** 仅输出 **`need_more_context`**。可选 `plain_en`、`idiom_flag`、`transliteration_requests`、`omit_sfx`；`notes` 可含供 Run E 使用的 PACK。

---

## 故障排除

### 「缺少必需的模型文件」
运行 `start.bat`，它会打开此 README 并告诉您缺少哪些文件。

### GPU 未检测到或性能缓慢
本项目已包含针对 NVIDIA GPU（CUDA 12.x）与 Intel CPU 优化的**预构建 llama-cpp-python 轮文件**。
- **NVIDIA 显卡**：请确保已安装最新 GPU 驱动程序。应用程序会自动检测并使用 CUDA。
- **Intel CPU**：CPU 版本针对现代 Intel 处理器优化，无需 AVX-512 指令集。
- **AMD/Intel Arc 显卡**：实验性支持，但需手动设置（预构建轮文件不包含）。

### Windows「符号链接」警告
此警告来自 Hugging Face 缓存。由于本项目不再自动下载模型，您可以忽略它。

### 音频模型错误（Run A）
Run A 使用 Hugging Face 模型 **ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition**。请执行 `pip install -r requirements_base.txt`（torch、transformers、soundfile、scipy）。首次运行会从 Hub 下载模型。音频提取需在 PATH 中有 ffmpeg。

### ffmpeg 未找到
- **Windows**：`install.bat` 在 PATH 中无 ffmpeg 时会自动下载到 `runtime\ffmpeg`。若失败请参阅 **FFMPEG_INSTALL.md** 手动下载与安装（BtbN 构建、winget，或将 `runtime\ffmpeg\bin` 加入 PATH）。
- **Linux / macOS**：`install.sh` 仅检查 PATH 中的 ffmpeg，不会自动下载；请用包管理器安装并参阅 **FFMPEG_INSTALL.md**。

---

## 许可证/免责声明

这是本地工具。您负责模型的许可和使用。
