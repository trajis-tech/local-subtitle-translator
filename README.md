## Trajis SmartSRT (Portable, Offline Models)

Trajis SmartSRT runs a local subtitle translation pipeline using llama-cpp-python. It is designed to be portable: all dependencies live inside this folder. It first builds an English, language-neutral translation brief (optionally assisted by audio/visual/context signals), then generates the final subtitles and enforces a glossary at the output—delivering more consistent, controllable results than one-shot translation.

| Aspect                                          | Typical online translators (Google/DeepL-like) | Typical subtitle translators (single-pass / batch tools) | **Trajis SmartSRT**                                                                                                                  |
| ----------------------------------------------- | ---------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Offline / privacy                               | Usually **no**                                 | Sometimes                                                | **Yes — offline-first & portable**, models live in `./models/`                                                                       |
| Subtitle-aware signals (tone / emotion / scene) | Mostly **none**                                | Limited                                                  | **Run A audio tone/emotion tags** + optional **Run C/D visual assist** → more context-aware subtitles                                |
| Long-video consistency                          | Can drift across chunks                        | Often drifts                                             | **A→F pipeline with an English “brief”** + optional **Run E context expansion** when needed                                          |
| Mix-language / drift control                    | Hard to control                                | Hard to control                                          | **Strict language boundary:** A–E are **English-only** (incl. brief), **only F outputs target language**                             |
| Terminology consistency                         | Manual fixes                                   | Basic find/replace                                       | **Glossary lock:** not injected into prompts; **forced at output stage**, can be **locale-specific** (zh-TW / zh-CN / ja-JP / es-ES) |
| Strategy for multi-model setups                 | N/A                                            | N/A                                                      | **Run F schemes** (e.g., main-led / local-led / draft-first) to mix “main” + “localization” models                                   |
| Alignment safety                                | Not SRT-aware                                  | Often index-based → can shift                            | **sub_id + timestamps alignment** to reduce offset/shift issues                                                                      |

✅ **Offline-first**: Run `install.bat` (Windows) or `./install.sh` (Linux/macOS) **once** with network to download and install everything; then use `start.bat` or `./start.sh` for **offline launch** anytime.

❗ **Models are NOT auto-downloaded.** You download GGUF files manually and place them into `./models/`.

![trajis-smartsrt-1 0 0-1](https://github.com/user-attachments/assets/b5376104-e259-4ded-9116-392d626fe29a)
![trajis-smartsrt-1 0 0-2](https://github.com/user-attachments/assets/6e6a34a8-ea7c-4145-bb1d-70be389b7b29)

---


## Translation Pipeline Overview

The translation pipeline is split into **6 runs** (A→B→C/D→E→F), executed sequentially. **Single brief**: one current file `./work/brief_work.jsonl` is updated each stage; before C/D/E/F the current brief is copied to `brief_v1.jsonl` / `brief_v2.jsonl` / `brief_v3.jsonl` / `brief_v4.jsonl` (snapshots).

- **Run A**: Audio emotion/tone analysis (all subtitle lines)
  - Extracts audio segments from video for each subtitle
  - Analyzes emotion, tone, intensity, and speaking style
  - Results saved to `./work/audio_tags.jsonl`
  - **Note**: Audio model is **pre-packaged** and **tightly coupled** with the code. **Do NOT modify or replace it.**

- **Run B**: Main model generates translation brief (all subtitle lines)
  - Uses the main reasoning model (GGUF, llama-cpp-python) to generate translation guidance.
  - Input: English subtitle + previous/next context (one line each side) + audio tags (all English).
  - Output: JSON — **all brief content in English only** (language-neutral): `meaning_tl`, `draft_tl`, `tl_instruction`, `idiom_requests`, `ctx_brief`, **`referents`**, **`tone_note`**, **`scene_brief`**, and optionally **`disambiguation_note`**, **`rewrite_guidance`**. **One need per stage**: v1 outputs **`need_vision`** only (boolean). Optional: `transliteration_requests`, `omit_sfx`; `reasons` (includes PACK for Run F).
  - Results saved to `./work/brief_work.jsonl` (single current brief).

- **Run C**: Single-frame vision fallback (optional, conditional)
  - Triggered when **`need_vision === true`** (from current brief).
  - Before updating: copy `brief_work.jsonl` → `brief_v1.jsonl`. Extracts one frame at the midpoint of subtitle time range; vision model analyzes scene/characters/actions.
  - Regenerates brief with single-frame vision hint → updates `./work/brief_work.jsonl`. **One need per stage**: v2 outputs **`need_multi_frame_vision`** only (boolean) per item.

- **Run D**: Multi-frame vision fallback (optional, conditional)
  - Triggered when **`need_multi_frame_vision === true`** (from current brief).
  - Before updating: copy `brief_work.jsonl` → `brief_v2.jsonl`. Extracts N frames (configurable, default: 3); vision model analyzes and merges descriptions.
  - Regenerates brief with multi-frame vision hint → updates `./work/brief_work.jsonl`. **One need per stage**: v3 outputs **`need_more_context`** only (boolean); used by Run E (context expansion).

- **Run E**: Context expansion (conditional)
  - Before updating: copy `brief_work.jsonl` → `brief_v3.jsonl`. For items with **`need_more_context === true`**, runs stage2 with **prev-3/next-3** context and updates their brief; writes `./work/brief_work.jsonl`. No model output in target language; brief only.

- **Run F**: Final translation (all subtitle lines)
  - Before translating, Run F copies the current brief to `brief_v4.jsonl` as a snapshot. Run F then reads **current brief** (`brief_work.jsonl`, updated by E) and PACK to produce full target-language subtitles and write the SRT. **Config**: `config.PipelineConfig.run_e_scheme` (default `"full"`) — **Run F scheme** in UI. **Execution**: `pipeline_runs.run_final_translate()`; output: `items.translated_text` and `work_dir/final_translations.jsonl`. **SRT write-back**: `app.py` aligns by `(round(start_ms), round(end_ms))`, keeps original `<i>` tags.
  - **One model at a time**: Only one of MAIN or LOCAL is loaded at a time; glossary is applied on output only.
  - **Robustness**: All chat calls go through **chat_dispatch**. Heavy requests can run in a one-shot subprocess when `pipeline.isolate_heavy_requests` is on; on failure, fallback to in-process with reduced max_tokens or CPU reload.

### Run F schemes (choose by main / localization model strength)

In the UI, select the scheme from the **Run F scheme** dropdown. Options: `full` | `main_led` | `local_led` | `draft_first` (invalid value falls back to `full`).

| Scheme | When to use (main／local) | Phase1 (draft source) | Phase2 (polish) |
|--------|---------------------|------------------------|-----------------|
| **Full** | main strong, local strong | MAIN group translate → draft_map | LOCAL polish |
| **MAIN-led** | main strong, local weak | MAIN group translate → draft_map | none |
| **LOCAL-led** | main weak, local strong | PACK draft_tl → draft_map; optional LOCAL fill idiom slots | LOCAL polish |
| **Draft-first** | main weak, local weak | PACK draft_tl → draft_map; optional LOCAL fill idiom slots | none |

- **Full**: Phase 1 — load MAIN (reason), build sentence groups (`build_sentence_groups`), split by `group_translate_max_segments` (default 4). For each chunk call `stage_main_group_translate`; **partial MAIN output is accepted** (match by **id** only; missing segments use PACK draft_tl or en_text). If MAIN returns nothing usable, use `safe_fallback_split`. Phase 2 — load LOCAL, `local_polish` in chunks (`local_polish_chunk_size`, default 60); only keys from the request and passing length/pollution checks are applied to draft_map. Final: glossary + strip_punctuation → `translated_text`.
- **MAIN-led**: Phase 1 — same as Full (MAIN group translate → draft_map). Phase 2 — **skipped**. Final: glossary + strip_punctuation.
- **LOCAL-led**: No MAIN. `draft_map = _build_draft_map_from_pack(...)`. If any item has `idiom_requests`, load LOCAL, call `stage3_suggest_local_phrases` per line, fill slots with `_fill_draft_with_suggestions`, update draft_map. Then load LOCAL and run `local_polish` in chunks. Final: glossary + strip_punctuation.
- **Draft-first**: No MAIN. Build draft_map from PACK only; if idiom_requests exist, load LOCAL, get suggestions, fill slots; **no** polish. Weak-localization models use **STRICT** prompts and `raw_decode` fallback to avoid format errors.

**Alignment**: All Run F alignment is by **sub_id** and **timestamps (start_ms, end_ms)**; never by list index, to avoid misalignment.

**Language policy (no mixing between stages)**:
- **Run A–E**: All prompts and all model outputs are **English only**. Run A (audio), Run B/C/D (brief), and Run E (context expansion) never receive or produce target-language text; the brief is language-neutral English so that Run F can translate from a single English interface.
- **Run F**: All instructions (prompts) are in **English**; input to the main and localization models is **English** (segments, tl_instruction, context). Only the **model output** (segment_texts[].text, polished lines, phrase suggestions) is in the **target language**.
- **Enforcement**: Run B brief output is sanitized (e.g. `tl_instruction` must be English-only). Run F uses `_tl_instruction_for_run_e()` so the translation stage always gets the correct target locale regardless of brief content.

**Prompt roles** (`model_prompts.csv`): MAIN (main_group_translate) receives English input and outputs target-language draft; naturalization is done by LOCAL. Strong localization (e.g. Llama-Breeze2-8B): focus on **TARGET LANGUAGE**, naturalize/polish; JSON output, no STRICT format. Weak localization (e.g. Breeze-7B, custom-localization): **STRICT** — output only a single JSON, no preamble/markdown; parse failure triggers `raw_decode` to extract first `{...}`.

**Transliteration**: Names or proper nouns that should be transliterated in the target language are the **localization model’s job**; the **main model** (Run B) proposes them in PACK as `transliteration_requests` (array of strings). Run F Phase2 (local_polish) receives these terms and appends “Transliterate in target language for these terms: …” to the polish prompt so the LOCAL model outputs transliterated forms.

**CC / SFX (狀聲詞)**: The **main model** (Run B) filters out sound effects and onomatopoeia (e.g. `[laughter]`, `[sigh]`, `*gasps*`). It can set `omit_sfx: true` and `draft_tl` to empty for SFX-only lines; for dialogue + SFX it puts only the dialogue in `draft_tl`. Run F applies `omit_sfx` after building the draft map so those lines get empty output.

**Run F config** (`config.py`): `run_e_scheme` (UI: Run F scheme), `group_translate_max_segments` (default 4), `local_polish_chunk_size` (default 60), `strip_punctuation`, `strip_punctuation_keep_decimal`, `strip_punctuation_keep_acronym`.

#### Run F: detailed logic, model I/O, and mandatory requirements

- **Entry**: `run_final_translate()` in `src/pipeline_runs.py`. **Config**: `cfg.pipeline.run_e_scheme` (`"full"` | `"main_led"` | `"local_led"` | `"draft_first"`).
- **Common**: `build_sentence_groups(items)`, `_build_protected_terms()`; `draft_map` keyed by **sub_id**. **omit_sfx**: if PACK has `omit_sfx === true`, set `draft_map[sub_id] = ""`. **Final**: glossary → optional strip_punctuation → `item.translated_text`; **name_mappings** for names CSV.

**Phase1 (MAIN group translate, schemes full / main_led only)**

- **Input**: `groups = build_sentence_groups(items)` (adjacent segments merged by gap &lt; 250 ms, sentence-end rules, short fragments). Each group is chunked by `group_translate_max_segments` (default 4). Per chunk: `segments_json` = array of `{id, en, ms}` (English text); `target_language` (locale code); `tl_instruction` from first segment PACK or default (**in English**).
- **Call**: `stage_main_group_translate(reason_model, target_language, segments_json, tl_instruction, ...)` with **json_mode=True**.
- **MAIN (main_group_translate) mandatory (CSV system_prompt_template)**:
  - Output **only** valid JSON; no markdown or extra text; use JSON Mode / grammar.
  - Role: translate each segment to target-language draft; naturalization is LOCAL’s job.
  - **Person names / proper nouns**: keep **original** in segment text; list them in **transliteration_requests** (array of strings).
  - **Subtitle format**: segment text may use only `?` `!` `-` `...` `""`; standard movie subtitle style; no markdown/asterisk emphasis.
  - **Required keys**: `target_language` (string), **segment_texts** (array of `{id, text}`), one entry per input segment, same order; each `text` one line.
  - **Optional**: `transliteration_requests` (array of strings).
- **Output handling**: Parse JSON (root array or object; accept segment array under `segment_texts` / `segments` / `translations` / `items` / `results` / `output` / `data`; per-item text from `text` / `draft_tl` / `tl` / `content` / `value`). Match by **id** to `expected_ids_chunk`; write to `draft_map`. If parsing fails or **id_to_text** is empty → **safe_fallback_split**(group_text, sub_segments, protected_terms) for that chunk.

**local_led / draft_first: draft from PACK, optional LOCAL idiom fill**

- No Phase1: `draft_map = _build_draft_map_from_pack()` (draft_tl or translation_brief / text_clean).
- If any PACK has **idiom_requests**: load LOCAL, call **stage3_suggest_local_phrases**(translate_model, requests_json, tl_instruction, target_language, ...) per line; **json_mode=True**. Fill slots with `_fill_draft_with_suggestions`, update draft_map.
- **LOCAL (localization) mandatory**: Output **only** valid JSON; **only** phrase suggestions (slot → phrase string), e.g. `{"I1":"phrase","I2":""}`; no full-sentence translation or extra text.

**Phase2 (LOCAL polish, schemes full / local_led only)**

- **Input**: `lines_for_polish = [{id, text: draft_map[sub_id]}, ...]` in sorted order; chunked by `local_polish_chunk_size` (default 60). **local_polish_mode** (`cfg.pipeline.local_polish_mode`, default `"strong"`): **weak** = strip punctuation on each line before sending; **strong** = keep punctuation.
- **Call**: `stage3_polish_local_lines(..., transliteration_terms, local_polish_mode)`; **json_mode=True**. User prompt includes tl_instruction and “Transliterate in target language for these terms: …” when transliteration_terms present.
- **Injected by mode (prepended to system)**:
  - **weak**: “Input text has punctuation removed. Your task: make each sentence more fluent and transliterate person names where listed.”
  - **strong**: “Input text includes punctuation. Your task: make each sentence more fluent and more localized; transliterate person names where listed. Output rule: the first character of each line must NOT be preceded by any punctuation.”
- **LOCAL (local_polish) mandatory (CSV)**: Output **only** valid JSON; keys = input sub_id, value = polished **single-line** string; preserve meaning; no retranslation; subtitle format: only `?` `!` `-` `...` `""`; optional **name_translations**: array of `{original, translated}`.
- **Acceptance**: Only keys in this chunk’s ids; value must be non-empty, single-line, length 0.3×–3× original, no prompt pollution; merge into draft_map. Parse **name_translations** for name_mappings CSV.

**Final output (all schemes)**

- Glossary → optional strip_punctuation (keep_decimal, keep_acronym) → `item.translated_text`. **name_mappings**: from Phase2 name_translations + PACK transliteration_requests (fill missing with empty translated).

**Run F config summary**

| Setting | Default | Description |
|---------|---------|-------------|
| run_e_scheme | "full" | full / main_led / local_led / draft_first |
| local_polish_mode | "strong" | weak = strip punctuation before polish input; strong = keep punctuation, output first char not preceded by punctuation |
| group_translate_max_segments | 4 | Phase1 max segments per chunk |
| local_polish_chunk_size | 60 | Phase2 lines per polish chunk |
| strip_punctuation | true | Strip punctuation on final output |
| strip_punctuation_keep_decimal | true | Preserve decimals (e.g. 3.14) |
| strip_punctuation_keep_acronym | true | Preserve acronyms (e.g. U.S.) |

**Model / CSV role mapping**

| Stage | Model | CSV role | Notes |
|-------|--------|----------|-------|
| Phase1 group translate | MAIN (reason) | main_group_translate | EN → target draft; keep names in text, list in transliteration_requests |
| local_led/draft_first idiom fill | LOCAL (translate) | localization | Output I1/I2… phrase suggestions only |
| Phase2 polish | LOCAL (translate) | local_polish | Draft → polished; weak/strong by local_polish_mode |

**“MAIN JSON failed or empty, using safe_fallback_split”**

- **When**: A Phase1 chunk’s MAIN response cannot be parsed into at least one segment with **id** in expected set and non-empty **text** → `id_to_text` empty → that chunk uses **safe_fallback_split** (PACK draft_tl or en_text split by protected_terms).
- **Causes**: Model returns nothing / not JSON; JSON truncated (max_tokens/EOS); trailing comma or prefix text; wrong root key or segment key; all segments dropped (missing id or empty text); ids not matching chunk. **Cannot be fully avoided** (model randomness, length limits, implementation differences).
- **Mitigations**: **json_mode** and strict prompts; prefix stripping and multi-step parsing (trailing comma fix, raw_decode, regex for segment_texts); accept multiple root and segment keys. When still no valid segment, **safe_fallback_split** ensures the chunk still gets a conservative translation.

#### Key Features

- **Single Model Loading**: Only one model is loaded at a time (audio, reason, vision, or translate)
- **Resumable**: Each run saves intermediate results to `./work/` (JSONL format)
- **Error Resilient**: If vision/audio fails, the pipeline continues with the best available brief
- **Progress Tracking**: Progress bar shows current step and completion percentage

---

## Video Input (FFmpeg note)

Gradio's built-in **Video** component performs server-side processing that requires an external **`ffmpeg`** executable.
If `ffmpeg` is not available, you can get errors like **"Executable 'ffmpeg' not found"**.

To keep this project **fully portable** (no system-wide installs), this repo uses a **File** input for the video instead.

- You need a video file for **Run A (audio)** and **Run C/D (vision)**.
- OpenCV (opencv-python) is used to grab frames for Vision, and ffmpeg is used to extract audio segments.
- **ffmpeg**: **Windows** – `install.bat` auto-downloads a portable ffmpeg to `runtime\ffmpeg` if not in PATH. **Linux / macOS** – `install.sh` only checks for ffmpeg in PATH; install it manually and see **FFMPEG_INSTALL.md** if missing.

If you are using an older zip that still uses the Video component, update to the latest zip, or install FFmpeg and add it to PATH.

## Installation and launch (offline-first)

This project is designed for **offline use**: run **install** once (with network), then use **start** anytime (no network).

**⚠️ IMPORTANT - For NVIDIA GPU users:**
- **Install CUDA Toolkit 12.9** (or 12.x) **BEFORE** running `install.bat` / `install.sh`
- Download: https://developer.nvidia.com/cuda-downloads
- The prebuilt llama-cpp-python wheel requires CUDA to be installed first for GPU acceleration

1. Extract this folder anywhere (example: `G:\Trajis SmartSRT`).

2. **Install (one-time, with network)** — downloads and installs everything:
   - **Windows**: Double-click `install.bat` → portable Python, venv, all Python deps (base + audio Run A + video), optional CUDA PyTorch if GPU present, ffmpeg to `runtime\ffmpeg` if not in PATH, Run A audio model to `models\audio`, prebuilt llama-cpp-python wheel, config, BOM. GGUF models are manual (see below).
     - **Estimated disk usage after install**: ~6-8 GB (CPU only: ~4-5 GB; with CUDA PyTorch + ffmpeg: ~6-8 GB)
   - **Linux / macOS**: Run `./install.sh` (same idea: .venv, deps, audio model, prebuilt llama-cpp-python wheel, config, BOM). If needed: `chmod +x install.sh`
     - **Estimated disk usage after install**: ~4-5 GB (excluding system Python and ffmpeg)

3. **Start (offline launch)** — no downloads, no network:
   - **Windows**: Double-click `start.bat` → checks .venv and model files, then launches the UI.
   - **Linux / macOS**: Run `./start.sh`. If needed: `chmod +x start.sh`

- **Uninstall**: Run `uninstall.bat` (Windows) or `./uninstall.sh` (Linux/macOS) to remove runtime, venv, and caches inside this folder. If needed: `chmod +x uninstall.sh`

**GPU Support:**

- **NVIDIA GPUs**: CUDA 12.x support (CUDA 12.9 recommended; RTX 20/30/40/50 series, GTX 16 series, and newer)
- **AMD GPUs**: ROCm support (experimental, requires manual setup)
- **Intel Arc GPUs**: oneAPI support (experimental, requires manual setup)
- **CPU**: Optimized for Intel CPUs (no AVX-512 required), works on all modern x86-64 processors

**Optional – install audio deps only (Linux/macOS):**

- Running `install.bat` or `install.sh` already installs Run A (audio) dependencies. Use `./scripts/install_audio_deps.sh` only if you need to reinstall audio deps without full setup (torch, transformers, soundfile, scipy). Requires Python 3 and optionally an active `.venv`.

Everything stays inside this folder (portable / isolated).

---

## Model compatibility and layout (required)

All **text and vision** models used by this app must be **GGUF** format and compatible with **llama-cpp-python**. You provide the model files; the app does not download them.

Create and use this folder layout:

```
models/
  main/     ← Main reasoning model (Run B); one or more .gguf files
  local/    ← Localization / translation model (Run F); one or more .gguf files
  vision/   ← Optional vision model (Run C/D); main .gguf + mmproj .gguf
  audio/    ← Run A audio model (downloaded by install script or on first run)
```

### Compatibility

- **Main & localization models**: Any **GGUF** model that works with llama-cpp-python (instruct/chat models with a chat template). Place files in `./models/main/` and `./models/local/` respectively. If a quantization is **sharded** (multiple `.gguf` files), download **all shards** and put them in the same folder.
- **Vision models (optional)**: Any **GGUF vision** model supported by llama-cpp-python (main model + mmproj). Place both files in `./models/vision/`. The app auto-detects model type from the filename. You can set exact filenames in `config.json` under `vision.text_model` and `vision.mmproj_model`.
- **Audio (Run A)**: The Run A emotion model is loaded from Hugging Face Hub on first run (no local GGUF). It uses Transformers `audio-classification`; dependencies: `torch`, `transformers`, `soundfile`, `scipy`.

### Parameter and quantization (generic guidance)

- **Quantization**: Lighter quants (e.g. **Q4_K_M**) use less VRAM and are faster; heavier quants (**Q5_K_M**, **Q6_K**, **Q8_0**) improve quality but need more VRAM and disk. Choose according to your GPU/RAM.
- **Model size**: Larger parameter counts (e.g. 14B, 7B) need more VRAM and RAM. Models are loaded **one at a time**, so VRAM is determined by the **largest single model** you use.
- **Context size**: Larger `n_ctx_*` (e.g. 8192) improves long-context handling but increases VRAM (KV cache). Reduce `n_ctx_*` or `n_gpu_layers_*` if you hit OOM.

### Suggested config.json starting points (tune for your hardware)

- **16 GB VRAM**: `n_ctx_reason=8192`, `n_ctx_translate=4096`, `n_gpu_layers_reason=60`, `n_gpu_layers_translate=60`
- **12 GB VRAM**: `n_ctx_reason=4096`, `n_ctx_translate=2048`, `n_gpu_layers_reason=50`, `n_gpu_layers_translate=50`
- **8 GB VRAM**: `n_ctx_reason=2048`, `n_ctx_translate=2048`, `n_gpu_layers_reason=35`, `n_gpu_layers_translate=35`
- **CPU only / low RAM**: Prefer Q4_K_M (or lighter) and smaller context; reduce `n_gpu_layers_*` or set to 0 for full CPU.

---

## config.json

When you run `install.bat` (or `install.sh`), it runs `scripts/plan_models.py` to create `config.json` if you don't have one. `start.bat` / `start.sh` do not create config; they only launch the app.
In normal use you **do not need to maintain config.json**. Models are discovered from `./models` directories. Edit config only if you need advanced tuning (VRAM, batch sizes, fallback settings).

---

## Work Directory (Intermediate Results)

All intermediate results are saved to `./work/` directory in JSONL format:

- `audio_tags.jsonl` - Run A results (audio emotion/tone analysis)
- `brief_work.jsonl` - Current brief (Run B writes; C/D/E update; Run F reads)
- `brief_v1.jsonl` - Snapshot before Run C (copy of brief_work.jsonl before C updates)
- `brief_v2.jsonl` - Snapshot before Run D (copy before D updates)
- `brief_v3.jsonl` - Snapshot before Run E (copy before E context expansion)
- `brief_v4.jsonl` - Snapshot before Run F (copy before final translation)
- `vision_1frame.jsonl` - Run C results (single-frame vision analysis)
- `vision_multiframe.jsonl` - Run D results (multi-frame vision analysis)
- `final_translations.jsonl` - Run F results (final translated text, new format)

**JSONL Format Compatibility:**

The pipeline supports both **legacy format** (using `idx` for alignment) and **new format** (using `sub_id` for alignment):

- **Legacy format**: Uses `idx` (integer index) to identify subtitle lines
  - Example: `{"idx": 0, "start_ms": 1000, "end_ms": 2000, ...}`
- **New format**: Uses `sub_id` (hash-based unique identifier) to ensure data alignment
  - Example: `{"sub_id": "a1b2c3d4_0", "start_ms": 1000, "end_ms": 2000, ...}`
  - `sub_id` is generated from `hash(start_ms, end_ms, text_raw)` to ensure consistency across runs

The pipeline automatically detects the format and handles conversion when needed. New runs will use the `sub_id` format for better data integrity.

**Resume functionality**: If a JSONL file exists and has the correct number of entries, the pipeline will automatically load it and skip that run. The pipeline supports resuming from both legacy (`idx`) and new (`sub_id`) formats.

**Manual resume**: You can delete specific JSONL files to re-run only those steps.

---

## UI Usage

1. **Upload files**: Video (MKV/MP4) and SRT (English subtitles)
2. **Select run mode**: `all` (A→B→(C/D)→E→F, default) | **A** (audio) | **B** (brief) | **C** (1-frame vision) | **D** (multi-frame vision) | **E** (context expansion) | **F** (translation)
3. **Run F scheme** (dropdown): choose by main / localization model strength — **Full** | **MAIN-led** | **LOCAL-led** | **Draft-first**. See **Run F schemes** in Translation Pipeline Overview.
4. **Optional fallbacks** (checkboxes in UI):
   - **Enable vision fallback (Run C/D)**: When checked, and the brief sets **need_vision** / **need_multi_frame_vision**, the pipeline runs single-frame (C) or multi-frame (D) vision and updates the brief. Requires a local GGUF vision model. Leave unchecked if you have no vision model.
   - **Enable context expansion fallback (Run E)**: When checked, items with **need_more_context** get prev-3/next-3 context and an updated brief before Run F. Recommended on; turn off to skip Run E (e.g. to save time when no lines need extra context).
5. **Click "🚀 Translate"** and monitor progress
6. **Download** the translated SRT file when complete
7. **Reset**: Click **"Reset"** to clear all inputs, outputs, and log and restore defaults so you can start a new translation job

**UI details**: The log panel shows the **newest entries at the top**. `model_prompts.csv` is read/written as UTF-8 with BOM; `start.bat` / `start.sh` run `ensure_csv_bom.py` on launch to keep the file encoding correct.

---

## Customizing Model Prompts (model_prompts.csv)

The translation pipeline uses prompts defined in `model_prompts.csv`. Each model's prompt is automatically matched by **model filename** (case-insensitive substring match). The file is expected to be **UTF-8 with BOM**; `start.bat` and `start.sh` run `scripts/ensure_csv_bom.py` on launch to ensure this.

### Model official prompt alignment

Prompts are designed to follow each model family’s **official** chat format and recommendations so that behaviour stays predictable and compatible:

- **Qwen2.5 (ChatML)**: System role + user role; JSON Mode for structured output. Templates use `chat_format=chatml` and strict “valid JSON only, no markdown” instructions as in the official Qwen usage.
- **Gemma 2 (e.g. TranslateGemma)**: **No system role**; all instruction is in the first user turn. The backend merges system content into the user message when `chat_format=gemma` so the model sees a single user turn.
- **Mistral / Llama 2 (e.g. Breeze, Llama-Breeze2)**: `[INST]`-style; system prompt is prepended to the first `[INST]` block. Used for `local_polish` and `localization` roles with STRICT JSON output where required.
- **Vision (Moondream, LLaVA)**: Prompts are applied in code per handler; chat format is auto-detected from the vision model filename. Output is always **English** visual description only (no subtitles).

The CSV **notes** column documents which role is “Run A~D all-English” vs “Run F: output target language only” so that custom rows keep the same language boundaries.

### Model Name Matching

- **How it works**: The app extracts the model filename (e.g., `my-main-model-q5_k_m.gguf`) and matches it against CSV `model_name` column.
- **Matching rule**: If the filename **contains** the CSV `model_name` (case-insensitive), it's a match.
  - Example: `my-main-model-q5_k_m.gguf` matches `my-main-model`
  - Example: `my-local-model-00001-of-00002.gguf` matches `my-local-model`
- **What to fill**: Use a **unique substring** that appears in your model filename. Usually the base model name without quantization suffix works.

### CSV Column Guide

| Column | Description | Example |
|--------|-------------|---------|
| `model_name` | Substring to match in filename (case-insensitive) | `my-main-model` |
| `role` | `main` (Run B), `main_group_translate` (Run F group translate), `main_assemble` (Run F legacy), `localization` (Run F phrase suggestions), `local_polish` (Run F batch polish), or `vision` (Run C/D) | `localization` |
| `source_language` | Input language (usually `English`) | `English` |
| `target_language` | Output language (Locale code: `en`, `zh-TW`, `zh-CN`, `ja-JP`, `es-ES`) | `zh-TW` |
| `chat_format` | Model's chat template (`chatml`, `llama-3`, `mistral-instruct`, `moondream`, `llava`) | `chatml` |
| `system_prompt_template` | System prompt (role definition) | See examples below |
| `user_prompt_template` | User prompt with placeholders | See examples below |
| `notes` | Description (English) | `Localization model for Traditional Chinese (Taiwan)` |

### Placeholders

Use these placeholders in `user_prompt_template`:

**Run B (main) placeholders:**
- `{line}` → Current English subtitle line
- `{context}` → Full context (Prev-1, Current, Next-1, Visual Hint if available)

**Run F (main_group_translate)** – MAIN translates a group of segments; **input all English**, output target-language; aligned per sub_id:
- `{target_language}` → Locale code (e.g. zh-TW, ja-JP)
- `{tl_instruction}` → Instruction **in English** (from first segment PACK or default)
- `{segments_json}` → JSON array of `{id, en, ms}` per segment (English text)

**Run F (local_polish)** – LOCAL batch-polishes draft lines; **all instructions in English**. Input may be English (translate+polish) or target language (polish only); output is target language:
- `{tl_instruction}` → Instruction **in English**
- `{lines_json}` → JSON array of `{id, text}` (draft text; may be English or target language)

**Run F (localization)** – used only for phrase suggestions (when a line has idiom slots); **all instructions in English**:
- `{tl_instruction}` → Instruction **in English** from Run B/D PACK
- `{requests_json}` → Idiom requests (slot, meaning_tl in English, register, max_len) as JSON
- `{target_language}` → Locale code (e.g. zh-TW, ja-JP)

**Run F (main_assemble)** – legacy Stage4 one-line assembly (MAIN model polishes draft + suggestions):
- `{target_language}` → Locale code (e.g. zh-TW, ja-JP, es-ES)
- `{line_en}` → Original English subtitle line
- `{ctx_brief}` → Context brief (**English**, from PACK)
- `{draft_prefilled}` → Draft with &lt;I1&gt;/&lt;I2&gt; already filled by suggestions
- `{suggestions_json}` → JSON of slot→phrase used

Glossary is **not** injected into any model; it is applied on the final output only (output-side force replace). Glossary entries can specify per-target-language replacements (`targets` by locale, e.g. `zh-TW`, `ja-JP`) or a single `to` / `zh` fallback.

**Run C/D (vision) placeholders:**
- `{line}` → Current English subtitle line

### Prompt Styles: Base vs Instruct Models

#### Base Models (Non-Instruct)
- **Characteristics**: Simpler, direct prompts without structured instruction format
- **Use when**: Your model is a base/completion model (not fine-tuned for instructions)
- **Style**: Direct questions or simple task descriptions

#### Instruct Models
- **Characteristics**: Structured instruction format with numbered rules and clear task definition
- **Use when**: Your model is instruction-tuned (Instruct, Chat, etc.)
- **Style**: Structured with rules, numbered steps, clear input/output definitions

### Examples in CSV

The CSV includes example rows for each role:

1. **`(custom-main-base)`** - Base model example for Run B
2. **`(custom-main-instruct)`** - Instruct model example for Run B
3. **`(custom-localization-base)`** - Base model example for Run F
4. **`(custom-localization-instruct)`** - Instruct model example for Run F
5. **`(custom-vision-base)`** - Base model example for Vision
6. **`(custom-vision-instruct)`** - Instruct model example for Vision

### Adding Your Own Model

1. **Copy an example row** (e.g., `(custom-main-instruct)`)
2. **Change `model_name`** to match your model filename substring
3. **Set `role`** (`main`, `localization`, or `vision`)
4. **Set `target_language`** to one of these locale codes:
   - `en` - English (for Run B main models)
   - `zh-TW` - Traditional Chinese (Taiwan)
   - `zh-CN` - Simplified Chinese (Mainland)
   - `ja-JP` - Japanese
   - `es-ES` - Spanish
   - Or other IETF locale codes as needed
5. **Set `chat_format`** to match your model's chat template
   - For vision models: This field is mainly for documentation. The actual chat format is **auto-detected** by `LocalVisionModel` based on the model filename. You can use any supported format (moondream, llava, etc.) as a placeholder.
6. **Write `system_prompt_template`** (role definition)
7. **Write `user_prompt_template`** (task with placeholders)
8. **Fill `notes`** (English description)

### Important Notes

- **Language boundaries**: **Run A–E** (audio, brief, vision, context expansion): prompts and model output must be **English only**. **Run F** (main_group_translate, local_polish, localization, main_assemble): prompts are in **English**; only the **output** (translated lines, phrase suggestions) is in the target language. Do not put target-language instructions (e.g. Chinese or Japanese) into Run F prompts—use English (e.g. “Output ONLY the translated subtitle in the target language (locale: zh-TW).”) so that prompt language never mixes with output language.
- **Prompt language**: For Run B/C/D use English-only prompts; for Run F use English instructions and expect target-language output from the model.
- **Chat format**: Must match your model's chat template. Wrong format can cause poor output or errors.
  - **For vision models**: The chat format is automatically detected by `LocalVisionModel` based on the model filename. The CSV `chat_format` field is mainly for documentation purposes.
- **Placeholders**: Always use the exact placeholder names (`{line}`, `{context}`, `{target_language}`, etc.). They are replaced automatically.
- **Run B/C/D output (brief)**: Must request JSON with `target_language`, `tl_instruction`, `meaning_tl`, `draft_tl`, `idiom_requests`, `ctx_brief`, referents, tone_note, scene_brief — **all brief content in English only** (language-neutral). **One need per stage**: **v1** outputs **`need_vision`** only; **v2** outputs **`need_multi_frame_vision`** only; **v3** outputs **`need_more_context`** only. Optional: `plain_en`, `idiom_flag`, `transliteration_requests`, `omit_sfx`; `notes` may contain PACK for Run F.

---

## Troubleshooting

### "Required model files are missing"
Run `start.bat`, it will open this README and tell you which files are missing.

### GPU not detected or slow performance
This project includes **prebuilt llama-cpp-python wheels** optimized for NVIDIA GPUs (CUDA 12.x) and Intel CPUs.
- **NVIDIA GPUs**: Ensure you have the latest GPU driver installed. The app will automatically use CUDA if available.
- **Intel CPUs**: The CPU version is optimized for modern Intel processors and does not require AVX-512 instruction set.
- **AMD/Intel Arc GPUs**: Experimental support available but requires manual setup (not included in prebuilt wheels).

### Windows "symlink" warning
This warning comes from Hugging Face caching. Since this project does not auto-download models anymore, you can ignore it.

### Audio model errors (Run A)
Run A uses the Hugging Face model **ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition**. Install dependencies with `pip install -r requirements_base.txt` (torch, transformers, soundfile, scipy). First run will download the model from Hugging Face Hub. Audio extraction requires ffmpeg in PATH.

### ffmpeg not found
- **Windows**: `install.bat` auto-downloads ffmpeg to `runtime\ffmpeg` when not in PATH. If that failed, see **FFMPEG_INSTALL.md** for manual download URL and install steps (BtbN builds, winget, or add `runtime\ffmpeg\bin` to PATH).
- **Linux / macOS**: `install.sh` does not auto-download ffmpeg; install it via your package manager and see **FFMPEG_INSTALL.md** if needed.

---

## License / Disclaimer

This is a local tool. You are responsible for model licensing and usage.
