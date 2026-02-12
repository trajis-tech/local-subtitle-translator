# ローカル字幕翻訳ツール（ポータブル、オフラインモデル）

このプロジェクトは **llama-cpp-python** を使用してローカル字幕翻訳パイプラインを実行します。
**ポータブル**に設計されています：すべての依存関係がこのフォルダ内にあります。

✅ **オフライン優先**：まず `install.bat`（Windows）または `./install.sh`（Linux/macOS）を**1回**（ネットワーク必要）実行してすべてをダウンロード・インストール；その後は `start.bat` または `./start.sh` で**オフライン起動**できます。

❗ **モデルは自動ダウンロードされません。** GGUF ファイルを手動でダウンロードして `./models/` に配置してください。

---

## 翻訳パイプライン概要

翻訳パイプラインは **6 つの実行段階**（A→B→C/D→E→F）に分割され、順次実行されます。**単一 brief**：現在は `./work/brief.jsonl` の 1 つのみを各段階で更新；C/D/E の前に `brief_v1.jsonl` / `brief_v2.jsonl` / `brief_v3.jsonl` にコピー（スナップショット）。

- **Run A**：音声感情/トーン分析（すべての字幕行）
  - 動画から各字幕に対応する音声セグメントを抽出
  - 感情、トーン、強度、話し方を分析
  - 結果を `./work/audio_tags.jsonl` に保存
  - **注意**：音声モデルは**事前パッケージ化**されており、コードと**密結合**しています。**変更や置換しないでください。**

- **Run B**：メインモデルが翻訳ブリーフを生成（すべての字幕行）
  - メイン推論モデル（GGUF、llama-cpp-python）で翻訳ガイダンスを生成。
  - 入力：英語字幕 + 前後 1 行ずつの文脈 + 音声タグ（すべて英語）。
  - 出力：JSON — **すべて英語のみ**（言語中立）：`meaning_tl`、`draft_tl`、`tl_instruction`、`idiom_requests`、`ctx_brief`、**`referents`**、**`tone_note`**、**`scene_brief`**、および任意で **`disambiguation_note`**、**`rewrite_guidance`**。**段階ごとに need は 1 つのみ**：v1 は **`need_vision`** のみ（真偽値）。任意で `transliteration_requests`、`omit_sfx`；`reasons`（Run F 用 PACK 含む）。
  - 結果を `./work/brief.jsonl` に保存（単一の現在ブリーフ）。

- **Run C**：単一フレーム視覚フォールバック（オプション、条件付きトリガー）
  - **`need_vision === true`**（現在の brief から）の場合にトリガー。
  - 更新前：`brief.jsonl` → `brief_v1.jsonl` にコピー。字幕時間範囲の中点で 1 フレームを抽出；視覚モデルがシーン/キャラクター/アクションを分析。
  - ブリーフを単一フレーム視覚ヒント付きで再生成 → `./work/brief.jsonl` を更新。**段階ごとに need は 1 つのみ**：v2 は **`need_multi_frame_vision`** のみ（真偽値）を各項目に出力。

- **Run D**：マルチフレーム視覚フォールバック（オプション、条件付きトリガー）
  - **`need_multi_frame_vision === true`**（現在の brief から）の場合にトリガー。
  - 更新前：`brief.jsonl` → `brief_v2.jsonl` にコピー。字幕時間範囲内で等間隔に N フレーム（設定可能、デフォルト：3）を抽出；視覚モデルが分析して説明をマージ。
  - ブリーフをマルチフレーム視覚ヒント付きで再生成 → `./work/brief.jsonl` を更新。**段階ごとに need は 1 つのみ**：v3 は **`need_more_context`** のみ（真偽値）；Run E（コンテキスト拡張）で使用。

- **Run E**：コンテキスト拡張（条件付き）
  - 更新前：`brief.jsonl` → `brief_v3.jsonl` にコピー。**`need_more_context === true`** の項目に対して **前 3／後 3 行**で stage2 を再実行してブリーフを更新し、`./work/brief.jsonl` に書き戻す。目標言語は出力しない；ブリーフのみ。

- **Run F**：最終翻訳（すべての字幕行）
  - Run F は**現在の brief**（`brief.jsonl`、E で更新済み）と PACK を読み、全編目標言語字幕を生成して SRT に書き出す。**設定**：`config.PipelineConfig.run_e_scheme`（デフォルト `"full"`）。**UI**：Run F scheme ドロップダウン。**実行**：`pipeline_runs.run_final_translate()`；出力：`items.translated_text` と `work_dir/final_translations.jsonl`。**SRT 書き戻し**：`app.py` が `(round(start_ms), round(end_ms))` でアラインし、原文 `<i>` タグを保持。
  - **同時にロードするモデルは 1 つのみ**：MAIN と LOCAL のいずれか一方のみ；用語集は出力時のみ適用。
  - **堅牢性**：すべての chat 呼び出しは **chat_dispatch** を経由。heavy リクエストは one-shot サブプロセスで実行可能；失敗時はプロセス内 fallback。

### Run F スキーム（メイン／ローカライゼーションモデルの強弱で選択）

UI の **Run F scheme** ドロップダウンでスキームを選択。オプション：`full` | `main_led` | `local_led` | `draft_first`（無効時は `full` にフォールバック）。

| Scheme | 選択タイミング（主／地） | Phase1（ドラフト元） | Phase2（ポリッシュ） |
|--------|---------------------|------------------------|-----------------|
| **Full** | 主強・地強 | MAIN グループ翻訳 → draft_map | LOCAL ポリッシュ |
| **MAIN-led** | 主強・地弱 | MAIN グループ翻訳 → draft_map | なし |
| **LOCAL-led** | 主弱・地強 | PACK draft_tl → draft_map；任意で LOCAL で idiom slot を埋める | LOCAL ポリッシュ |
| **Draft-first** | 主弱・地弱 | PACK draft_tl → draft_map；任意で LOCAL で idiom slot を埋める | なし |

- **Full**：Phase 1 — MAIN（reason）をロード、文グループを構築、`group_translate_max_segments`（デフォルト 4）で chunk に分割；各 chunk で `stage_main_group_translate` を呼び出し；**MAIN 出力を部分採用可**（**id** でマッチ；不足分は PACK draft_tl または en_text）。Phase 2 — LOCAL をロード、chunk で `local_polish`（`local_polish_chunk_size` デフォルト 60）；リクエスト内 key かつ長度・汚染チェック通過分のみ draft_map に反映。最終：用語集 + strip_punctuation → `translated_text`。
- **MAIN-led**：Phase 1 — Full と同じ（MAIN グループ翻訳 → draft_map）。Phase 2 — **スキップ**。最終：用語集 + strip_punctuation。
- **LOCAL-led**：MAIN は使わない。`draft_map = _build_draft_map_from_pack(...)`。`idiom_requests` がある項目があれば LOCAL をロードし、該当行で `stage3_suggest_local_phrases` を呼び、`_fill_draft_with_suggestions` で slot を埋めて draft_map を更新。その後 LOCAL で全行 `local_polish`。最終：用語集 + strip_punctuation。
- **Draft-first**：MAIN は使わない。PACK のみで draft_map を構築；idiom_requests があれば LOCAL で提案取得・埋め；**ポリッシュなし**。弱ローカライゼーションモデルは **STRICT** プロンプトと raw_decode フォールバックで出力フォーマットエラーを回避。

**アライメント**：Run F のアライメントはすべて **sub_id** と**タイムスタンプ (start_ms, end_ms)** に基づき、リストインデックスは使わない。

**言語ポリシー（段階間で混在させない）**：
- **Run A–E**：すべての prompt とモデル出力は **英語のみ**。Run A（音声）、Run B/C/D（brief）、Run E（コンテキスト拡張）は目標言語を受け取らず産出しない；brief は言語中立の英語で、Run F が単一の英語インターフェースから翻訳する。
- **Run F**：すべての**指示（prompt）は英語**；メインおよびローカライゼーションモデルへの入力は**英語**（セグメント、tl_instruction、コンテキスト）。**モデル出力**（segment_texts[].text、ポリッシュ済み行、フレーズ提案）のみが**目標言語**。
- **強制**：Run B の brief 出力はサニタイズされる（例：`tl_instruction` は英語のみ）。Run F は `_tl_instruction_for_run_e()` により、翻訳段で常に正しい目標ロケールで出力する。

**Prompt 役割**（`model_prompts.csv`）：MAIN（main_group_translate）は**原文（SOURCE）**に専念；強ローカライゼーション（例：Llama-Breeze2-8B）は**目標言語**に専念・自然化／ポリッシュ；弱ローカライゼーション（例：Breeze-7B、custom-localization）は **STRICT** 形式；パース失敗時は `raw_decode` で最初の `{...}` を取得。

**音訳**：目標言語で音訳すべき人名・固有名詞は**ローカライゼーションモデル**の担当；**メインモデル**（Run B）は PACK の `transliteration_requests`（文字列配列）で指定。Run F Phase2（local_polish）でこれらの語を受け取り、polish プロンプトに「Transliterate (音譯) in target language for these terms: …」を付加し、LOCAL が音訳形を出力。

**CC／効果音**：**メインモデル**（Run B）が効果音・擬音（例：`[laughter]`、`[sigh]`、`*gasps*`）を除外。SFX のみの行は `omit_sfx: true` かつ `draft_tl` を空に；対話＋SFX の行は `draft_tl` に対話のみ。Run F は draft_map 構築後に omit_sfx を適用し、該当行は空出力。

**Run F 設定**（`config.py`）：`run_e_scheme`（UI：Run F scheme）、`group_translate_max_segments`（デフォルト 4）、`local_polish_chunk_size`（デフォルト 60）、`strip_punctuation`、`strip_punctuation_keep_decimal`、`strip_punctuation_keep_acronym`。

### 主な機能

- **単一モデル読み込み**：一度に1つのモデルのみ読み込まれます（音声、推論、視覚、または翻訳）
- **再開可能**：各 run は中間結果を `./work/` に保存します（JSONL 形式）
- **エラー耐性**：視覚/音声が失敗した場合、パイプラインは利用可能な最適なブリーフで続行します
- **進捗追跡**：プログレスバーに現在のステップと完了パーセンテージが表示されます

---

## 動画入力（FFmpeg に関する注意）

Gradio の組み込み **Video** コンポーネントは、外部の **`ffmpeg`** 実行ファイルを必要とするサーバー側処理を実行します。
`ffmpeg` が利用できない場合、**"Executable 'ffmpeg' not found"** のようなエラーが発生する可能性があります。

このプロジェクトを**完全にポータブル**（システム全体のインストール不要）に保つため、このリポジトリでは代わりに **File** 入力を動画に使用します。

- **Run A（音声）** と **Run C/D（視覚）** に動画ファイルが必要です。
- OpenCV (opencv-python) を使用して Vision のフレームを取得し、ffmpeg を使用して音声セグメントを抽出します。
- **ffmpeg**：**Windows** – PATH に ffmpeg が無い場合、`install.bat` が `runtime\ffmpeg` にポータブル ffmpeg を自動ダウンロード。**Linux / macOS** – `install.sh` は PATH 内の ffmpeg のみチェック；未インストールの場合は手動でインストールし **FFMPEG_INSTALL.md** を参照。

Video コンポーネントを使用している古い zip を使用している場合は、最新の zip に更新するか、FFmpeg をインストールして PATH に追加してください。

## インストールと起動（オフライン優先）

このプロジェクトは**オフライン優先**です：まず **install** を1回（ネットワーク必要）実行し、その後は **start** でいつでもオフライン起動します。

**⚠️ 重要 - NVIDIA GPU ユーザー：**
- **CUDA Toolkit 12.9**（または 12.x）を `install.bat` / `install.sh` **実行前に**インストールしてください
- ダウンロード：https://developer.nvidia.com/cuda-downloads
- ビルド済み llama-cpp-python ホイールは GPU アクセラレーションのために CUDA の事前インストールが必要です

1. このフォルダを任意の場所に展開します（例：`G:\local-subtitle-translator`）。

2. **インストール（1回、ネットワーク必要）** — すべてをダウンロード・インストール：
   - **Windows**：`install.bat` をダブルクリック → ポータブル Python、venv、全 Python 依存（音声 Run A、動画含む）、GPU があれば CUDA PyTorch、PATH に ffmpeg が無ければ `runtime\ffmpeg` にダウンロード、Run A 音声モデルを `models\audio` にダウンロード、ビルド済み llama-cpp-python ホイール、config、BOM。GGUF モデルは手動ダウンロード（下記参照）。
     - **インストール後の推定ディスク使用量**：約 6-8 GB（CPU のみ：約 4-5 GB；CUDA PyTorch + ffmpeg 含む：約 6-8 GB）
   - **Linux / macOS**：`./install.sh` を実行（同様：.venv、依存、音声モデル、ビルド済み llama-cpp-python ホイール、config、BOM）。必要に応じて：`chmod +x install.sh`
     - **インストール後の推定ディスク使用量**：約 4-5 GB（システム Python と ffmpeg を除く）

3. **起動（オフライン）** — ダウンロードなし、ネットワーク不要：
   - **Windows**：`start.bat` をダブルクリック → .venv とモデルファイルをチェックして UI 起動。
   - **Linux / macOS**：`./start.sh` を実行。必要に応じて：`chmod +x start.sh`

- **アンインストール**：`uninstall.bat`（Windows）または `./uninstall.sh`（Linux/macOS）を実行するとこのフォルダ内のランタイム、venv、キャッシュを削除。必要に応じて：`chmod +x uninstall.sh`

**GPU サポート：**

- **NVIDIA GPU**：CUDA 12.x サポート（CUDA 12.9 推奨；RTX 20/30/40/50 シリーズ、GTX 16 シリーズ以降）
- **AMD GPU**：ROCm サポート（実験的、手動設定が必要）
- **Intel Arc GPU**：oneAPI サポート（実験的、手動設定が必要）
- **CPU**：Intel CPU 向けに最適化（AVX-512 命令セット不要）、すべての現代的な x86-64 プロセッサで動作

**オプション – 音声依存のみインストール（Linux / macOS）：**

- `install.bat` または `install.sh` を実行すれば Run A（音声）の依存は既にインストールされます。音声依存（torch、transformers、soundfile、scipy）のみを再インストールし、フルセットアップを省略したい場合にのみ `./scripts/install_audio_deps.sh` を使用。Python 3 が必要；既存の `.venv` を有効にしても可。

すべてがこのフォルダ内に保持されます（ポータブル/分離）。

---

## モデル互換性とディレクトリ構成（必須）

本アプリで使用する**テキスト・視覚**モデルは **GGUF** 形式で、**llama-cpp-python** と互換性がある必要があります。モデルファイルはユーザーが用意し、アプリは自動ダウンロードしません。

以下のディレクトリ構成を作成/使用してください：

```
models/
  main/     ← メイン推論モデル（Run B）；1つ以上の .gguf ファイル
  local/    ← ローカライゼーション/翻訳モデル（Run E）；1つ以上の .gguf ファイル
  vision/   ← オプション視覚モデル（Run C/D）；メイン .gguf + mmproj .gguf
  audio/    ← Run A 音声モデル（インストールスクリプトまたは初回実行時にダウンロード）
```

### 互換性

- **メイン・ローカライゼーションモデル**：llama-cpp-python で動作する **GGUF** モデル（チャットテンプレート付き instruct/chat モデル）。ファイルを `./models/main/` と `./models/local/` に配置。**シャード**（複数 .gguf）の場合は**全シャード**をダウンロードし同一フォルダに配置。
- **視覚モデル（オプション）**：llama-cpp-python がサポートする **GGUF 視覚**モデル（メイン + mmproj）。両ファイルを `./models/vision/` に配置。アプリはファイル名からモデルタイプを自動検出。`config.json` の `vision.text_model` と `vision.mmproj_model` で正確なファイル名を指定可能。
- **音声（Run A）**：Run A 感情モデルは初回実行時に Hugging Face Hub からダウンロード（ローカル GGUF なし）。Transformers `audio-classification` 使用；依存：`torch`、`transformers`、`soundfile`、`scipy`。

### パラメータと量子化（一般的な目安）

- **量子化**：軽い量子化（**Q4_K_M** など）は VRAM 節約・高速；重い量子化（**Q5_K_M**、**Q6_K**、**Q8_0**）は品質向上だが VRAM・ディスクを多く使用。GPU/RAM に応じて選択。
- **モデルサイズ**：パラメータ数が大きい（14B、7B など）ほど VRAM/RAM が必要。モデルは**一度に1つ**しか読み込まないため、VRAM は**使用する最大の単一モデル**で決まる。
- **コンテキストサイズ**：`n_ctx_*` を大きく（例 8192）すると長文対応が良くなるが VRAM（KV キャッシュ）が増える。OOM の場合は `n_ctx_*` または `n_gpu_layers_*` を下げる。

### config.json の推奨開始点（ハードウェアに合わせて調整）

- **16 GB VRAM**：`n_ctx_reason=8192`, `n_ctx_translate=4096`, `n_gpu_layers_reason=60`, `n_gpu_layers_translate=60`
- **12 GB VRAM**：`n_ctx_reason=4096`, `n_ctx_translate=2048`, `n_gpu_layers_reason=50`, `n_gpu_layers_translate=50`
- **8 GB VRAM**：`n_ctx_reason=2048`, `n_ctx_translate=2048`, `n_gpu_layers_reason=35`, `n_gpu_layers_translate=35`
- **CPU のみ / 低 RAM**：Q4_K_M（またはより軽い）と小さいコンテキストを推奨；`n_gpu_layers_*` を減らすか 0 で全 CPU 実行。

---

## config.json

`install.bat`（または `install.sh`）実行時、`config.json` がない場合は `scripts/plan_models.py` が実行され、最善の `config.json` が作成されます。`start.bat` / `start.sh` は config を作成せず、起動のみ行います。
異なる量子化ファイルを選択した場合は、それを編集してください。

主要フィールド：

- `models_dir`：デフォルト `models`
- `reason_dir`：主モデルのディレクトリ（デフォルト：`./models/main/`）
- `translate_dir`：ローカライゼーションモデルのディレクトリ（デフォルト：`./models/local/`）
- `vision_text_dir` / `vision_mmproj_dir`：視覚モデルのディレクトリ（デフォルト：`./models/vision/`）
- `audio.model_id_or_path`：Run A 用 Hugging Face モデル id（デフォルト：`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`）
- `audio.enabled`：音声分析の有効/無効（デフォルト：`true`）
- `audio.device`：`"auto"`（CUDA 利用時）、`"cuda"` または `"cpu"`（デフォルト：`auto`）
- `audio.device_index`：CUDA 使用時の GPU インデックス（デフォルト：`0`）
- `audio.batch_size`：感情推論のバッチサイズ；大きいほど GPU 利用率向上（デフォルト：`16`）
- `vision.enabled`：オプションの視覚フォールバック
- `pipeline.n_frames`：マルチフレーム視覚のフレーム数（デフォルト：`3`）
- `pipeline.work_dir`：中間結果ディレクトリ（デフォルト：`./work/`）
- `pipeline.run_e_scheme`：Run F スキーム — `"full"` | `"main_led"` | `"local_led"` | `"draft_first"`（デフォルト：`"full"`）。上記 **Run F スキーム** を参照。
- `pipeline.local_polish_chunk_size`：Run E local_polish の 1 チャンク行数（デフォルト：`60`）
- `pipeline.group_translate_max_segments`：Run E メイン翻訳のサブグループ最大セグメント数（デフォルト：`4`）
- `pipeline.isolate_heavy_requests`：`true` の場合、heavy リクエスト（token/行数/セグメントが閾値超過）は one-shot サブプロセスで実行し OOM でメインプロセスが落ちるのを防ぐ（デフォルト：`true`）
- `pipeline.isolate_heavy_timeout_sec`：隔離ワーカーのタイムアウト秒（デフォルト：`600`）
- `pipeline.strip_punctuation`：`true` の場合、Run E 最終出力で句読点を除去（デフォルト：`true`）
- `pipeline.strip_punctuation_keep_decimal`：`true` の場合、`3.14` などの小数を保護（デフォルト：`true`）
- `pipeline.strip_punctuation_keep_acronym`：`true` の場合、`U.S.` などの略語を保護（デフォルト：`true`）

---

## 作業ディレクトリ（中間結果）

すべての中間結果は `./work/` ディレクトリに JSONL 形式で保存されます：

- `audio_tags.jsonl` - Run A 結果（音声感情/トーン分析）
- `brief.jsonl` - 現在のブリーフ（Run B が書き込み；C/D/E が更新；Run F が読み取り）
- `brief_v1.jsonl` - Run C 前スナップショット（更新前コピー）
- `brief_v2.jsonl` - Run D 前スナップショット
- `brief_v3.jsonl` - Run E 前スナップショット
- `vision_1frame.jsonl` - Run C 結果（単一フレーム視覚分析）
- `vision_multiframe.jsonl` - Run D 結果（複数フレーム視覚分析）
- `final_translations.jsonl` - Run F 結果（最終翻訳テキスト、新形式）

**JSONL 形式の互換性：**

パイプラインは**旧形式**（`idx` を使用した整列）と**新形式**（`sub_id` を使用した整列）の両方をサポートします：

- **旧形式**：`idx`（整数インデックス）を使用して字幕行を識別
  - 例：`{"idx": 0, "start_ms": 1000, "end_ms": 2000, ...}`
- **新形式**：`sub_id`（ハッシュベースの一意識別子）を使用してデータ整列を保証
  - 例：`{"sub_id": "a1b2c3d4_0", "start_ms": 1000, "end_ms": 2000, ...}`
  - `sub_id` は `hash(start_ms, end_ms, text_raw)` から生成され、run 間の一貫性を保証

パイプラインは形式を自動検出し、必要に応じて変換を処理します。新しい run は、より優れたデータ整合性のために `sub_id` 形式を使用します。

**再開機能**：JSONL ファイルが存在し、エントリ数が正しい場合、パイプラインは自動的に読み込んでその run をスキップします。パイプラインは旧形式（`idx`）と新形式（`sub_id`）の両方からの再開をサポートします。

**手動再開**：特定の JSONL ファイルを削除して、それらのステップのみを再実行できます。

---

## UI の使用方法

1. **ファイルをアップロード**：動画（MKV/MP4）と SRT（英語字幕）
2. **実行モードを選択**：`all`（A→B→(C/D)→E→F、デフォルト）| **A**（音声）| **B**（ブリーフ）| **C**（単一フレーム視覚）| **D**（マルチフレーム視覚）| **E**（コンテキスト拡張）| **F**（翻訳）
3. **Run F scheme**（ドロップダウン）：メイン／ローカライゼーションモデルの強弱で選択 — **Full** | **MAIN-led** | **LOCAL-led** | **Draft-first**。上記 **Run F スキーム** を参照。
4. **オプションのフォールバック**（UI でチェック）：
   - **視覚フォールバックを有効にする（Run C/D）**：チェック時、brief が **need_vision**／**need_multi_frame_vision** なら単一フレーム（C）またはマルチフレーム（D）視覚を実行してブリーフを更新。ローカル GGUF 視覚モデルが必要。
   - **コンテキスト拡張フォールバックを有効にする（Run E）**：チェック時、**need_more_context** の項目は Run F の前に前 3／後 3 行で拡張してブリーフを更新。推奨。
   - **Max frames per subtitle (Run D)**：マルチフレーム視覚のフレーム数（デフォルト：1–4）；**Frame offsets**：サンプル位置。
5. **「🚀 Translate」をクリック**して進捗を監視
6. **翻訳が完了したら** SRT ファイルをダウンロード
7. **リセット**：**「Reset」**をクリックすると、すべての入力・出力・ログをクリアし、既定値に戻して新しい翻訳を開始できます

**UI の詳細**：ログは**新しいエントリが上**に表示されます。`model_prompts.csv` は UTF-8（BOM 付き）で読み書きされます。`start.bat` / `start.sh` 起動時に `ensure_csv_bom.py` が実行され、ファイルエンコーディングが維持されます。

---

## モデルプロンプトのカスタマイズ（model_prompts.csv）

翻訳パイプラインは `model_prompts.csv` で定義されたプロンプトを使用します。各モデルのプロンプトは**モデルファイル名**によって自動的にマッチングされます（大文字小文字を区別しない部分文字列マッチ）。ファイルは **UTF-8（BOM 付き）** である必要があります。`start.bat` と `start.sh` 起動時に `scripts/ensure_csv_bom.py` が実行され、エンコーディングが確保されます。

### モデル公式プロンプトとの整合

プロンプトは各モデルファミリーの**公式**チャット形式と推奨に沿って設計され、動作の予測可能性と互換性を保ちます：

- **Qwen2.5 (ChatML)**：System ロール + User ロール；JSON Mode で構造化出力。テンプレートは `chat_format=chatml` と厳格な「有効な JSON のみ、markdown なし」指示を使用（公式 Qwen 用法に準拠）。
- **Gemma 2（例：TranslateGemma）**：**System ロールなし**；すべての指示は最初の user turn に。バックエンドは `chat_format=gemma` 時に system 内容を user メッセージにマージし、モデルは単一の user turn のみを見る。
- **Mistral / Llama 2（例：Breeze、Llama-Breeze2）**：`[INST]` スタイル；system prompt は最初の `[INST]` ブロック前に付加。`local_polish`・`localization` ロールで使用；必要に応じて STRICT JSON 出力。
- **Vision（Moondream、LLaVA）**：プロンプトはコード内でハンドラごとに適用；チャット形式は視覚モデルのファイル名から自動検出。出力は常に**英語**の視覚説明のみ（字幕は出さない）。

CSV の **notes** 列で、そのロールが「Run A~D はすべて英語」か「Run E：出力は目標言語のみ」かを記載し、カスタム行でも同じ言語境界を保つ。

### モデル名マッチング

- **動作方法**：アプリケーションはモデルファイル名（例：`my-main-model-q5_k_m.gguf`）を抽出し、CSV `model_name` 列と照合します。
- **マッチングルール**：ファイル名が CSV `model_name` を**含む**場合（大文字小文字を区別しない）、マッチします。
  - 例：`my-main-model-q5_k_m.gguf` は `my-main-model` とマッチ
  - 例：`my-local-model-00001-of-00002.gguf` は `my-local-model` とマッチ
- **入力内容**：モデルファイル名に現れる**一意の部分文字列**を使用してください。通常、量子化サフィックスなしの基本モデル名で機能します。

### CSV 列ガイド

| 列 | 説明 | 例 |
|--------|-------------|---------|
| `model_name` | ファイル名でマッチする部分文字列（大文字小文字を区別しない） | `my-main-model` |
| `role` | `main`（Run B）、`main_assemble`（Run E Stage4 組立て）、`localization`（Run E）、または `vision`（Run C/D） | `localization` |
| `source_language` | 入力言語（通常は `English`） | `English` |
| `target_language` | 出力言語（ロケールコード：`en`、`zh-TW`、`zh-CN`、`ja-JP`、`es-ES`） | `zh-TW` |
| `chat_format` | モデルのチャットテンプレート（`chatml`、`llama-3`、`mistral-instruct`、`moondream`） | `chatml` |
| `system_prompt_template` | システムプロンプト（役割定義） | 以下の例を参照 |
| `user_prompt_template` | ユーザープロンプト（プレースホルダー付き） | 以下の例を参照 |
| `notes` | 説明（英語） | `Localization model for Traditional Chinese (Taiwan)` |

### プレースホルダー

`user_prompt_template` でこれらのプレースホルダーを使用してください：

**Run B（main）プレースホルダー：**
- `{line}` → 現在の英文字幕行
- `{context}` → 完全なコンテキスト（Prev-1、Current、Next-1、Prev-More、Next-More、Visual Hint）

**Run E（localization）プレースホルダー：**
- `{tl_instruction}`、`{requests_json}`、`{target_language}`（フレーズ提案用）

**Run E（main_assemble）** – Stage4 一行組立て：
- `{target_language}`、`{line_en}`、`{ctx_brief}`、`{draft_prefilled}`、`{suggestions_json}`

**Run C/D（vision）プレースホルダー：**
- `{line}` → 現在の英文字幕行

### プロンプトスタイル：Base vs Instruct モデル

#### Base モデル（非 Instruct）
- **特徴**：構造化された命令形式のない、より単純で直接的なプロンプト
- **使用時**：モデルがベース/完成モデル（命令用に微調整されていない）の場合
- **スタイル**：直接的な質問または単純なタスク説明
- **例**（Run B）：
  ```
  Analyze this subtitle line and explain what it really means in plain English.
  
  Subtitle: {line}
  Context: {context}
  
  Explain the meaning, including any idioms, jokes, tone, or implied meaning.
  ```

#### Instruct モデル
- **特徴**：番号付きルールと明確なタスク定義を持つ構造化された命令形式
- **使用時**：モデルが命令調整されている場合（Instruct、Chat など）
- **スタイル**：ルール、番号付きステップ、明確な入力/出力定義を持つ構造化
- **例**（Run B）：
  ```
  You are stage 2 (reasoning) in a multi-stage subtitle translation pipeline.
  - Input: one English subtitle line plus nearby context.
  - Output: ENGLISH ONLY: a clear, unambiguous explanation...
  - Do NOT translate to any target language here.
  
  Subtitle line: {line}
  Context (previous/next lines): {context}
  ```

### CSV の例

CSV には各役割の例行が含まれています：

1. **`(custom-main-base)`** - Run B の Base モデル例
2. **`(custom-main-instruct)`** - Run B の Instruct モデル例
3. **`(custom-localization-base)`** - Run E の Base モデル例
4. **`(custom-localization-instruct)`** - Run E の Instruct モデル例
5. **`(custom-vision-base)`** - Vision の Base モデル例
6. **`(custom-vision-instruct)`** - Vision の Instruct モデル例

### 独自のモデルを追加する

1. **例行をコピー**（例：`(custom-main-instruct)`）
2. **`model_name` を変更**して、モデルファイル名の部分文字列に一致させる
3. **`role` を設定**（`main`、`localization`、または `vision`）
4. **`target_language` を設定**して、以下のロケールコードのいずれかにする：
   - `en` - 英語（Run B main モデル用）
   - `zh-TW` - 繁体中国語（台湾）
   - `zh-CN` - 簡体中国語（大陸）
   - `ja-JP` - 日本語
   - `es-ES` - スペイン語
   - または必要に応じて他の IETF ロケールコード
5. **`chat_format` を設定**して、モデルのチャットテンプレートに一致させる：
   - `chatml` - 多くの現代 instruct/chat モデル
   - `llama-3` - Llama 3 モデル
   - `mistral-instruct` - Mistral モデル
   - `moondream` - 一部の視覚モデル
6. **`system_prompt_template` を記述**（役割定義、通常 1-2 文）
   - ローカライゼーションモデルの場合：プロンプトで目標言語を一般的に言及したい場合は、`[target_language]` をプレースホルダーとして使用
7. **`user_prompt_template` を記述**（プレースホルダー付きタスク）
   - ベースモデルには Base スタイルを使用
   - 命令調整モデルには Instruct スタイルを使用
   - ローカライゼーションモデルの場合：プロンプトテキストで `[target_language]` を実際のロケールコード（例：`zh-TW`、`ja-JP`）に置き換える
8. **`notes` を記入**（英語の説明）

### 重要な注意事項

- **言語境界**：**Run A–D**（音声、brief v1/v2/v3、視覚）：プロンプトとモデル出力は **英語のみ** にすること。**Run E**（main_group_translate、local_polish、localization、main_assemble）：プロンプトは**英語**；**出力**（翻訳行、フレーズ提案）のみが目標言語。Run E のプロンプトに目標言語の指示（例：中国語や日本語）を書かないこと— 英語で書く（例：「Output ONLY the translated subtitle in the target language (locale: zh-TW).」）。プロンプト言語と出力言語を混在させない。
- **プロンプト言語**：Run B/C/D は英語のみのプロンプト；Run E は英語の指示を使い、モデル出力は目標言語を想定。
- **チャット形式**：モデルのチャットテンプレートに一致する必要があります。間違った形式は、出力不良やエラーを引き起こす可能性があります。
  - **視覚モデル**：チャット形式は `LocalVisionModel` がモデルファイル名から自動検出します。CSV の `chat_format` 列は主にドキュメント用です。
- **プレースホルダー**：常に正確なプレースホルダー名（`{line}`、`{context}`、`{target_language}` など）を使用してください。これらは自動的に置き換えられます。
- **Run B/C/D 出力（brief）**：JSON で `target_language`、`tl_instruction`、`meaning_tl`、`draft_tl`、`idiom_requests`、`ctx_brief`、referents、tone_note、scene_brief — **すべて英語のみ**（言語中立）を要求。**段階ごとに need は 1 つのみ**：**v1** は **`need_vision`** のみ；**v2** は **`need_multi_frame_vision`** のみ；**v3** は **`need_more_context`** のみ。任意で `plain_en`、`idiom_flag`、`transliteration_requests`、`omit_sfx`；`notes` に Run E 用 PACK を含めてよい。

---

## トラブルシューティング

### 「必要なモデルファイルがありません」
`start.bat` を実行すると、この README が開き、どのファイルが不足しているかが表示されます。

### GPU が検出されないまたはパフォーマンスが遅い
このプロジェクトには NVIDIA GPU（CUDA 12.x）と Intel CPU 向けに最適化された**ビルド済み llama-cpp-python ホイール**が含まれています。
- **NVIDIA GPU**：最新の GPU ドライバーがインストールされていることを確認してください。アプリケーションは自動的に CUDA を検出して使用します。
- **Intel CPU**：CPU 版は現代的な Intel プロセッサ向けに最適化されており、AVX-512 命令セットは不要です。
- **AMD/Intel Arc GPU**：実験的サポートがありますが、手動設定が必要です（ビルド済みホイールには含まれていません）。

### Windows「シンボリックリンク」警告
この警告は Hugging Face キャッシュから発生します。このプロジェクトはモデルを自動ダウンロードしなくなったため、無視できます。

### 音声モデルエラー（Run A）
Run A は Hugging Face モデル **ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition** を使用します。`pip install -r requirements_base.txt`（torch、transformers、soundfile、scipy）で依存関係をインストールしてください。初回実行時に Hugging Face Hub からモデルがダウンロードされます。音声抽出には PATH に ffmpeg が必要です。

### ffmpeg が見つからない
- **Windows**：PATH に ffmpeg が無い場合、`install.bat` が `runtime\ffmpeg` に自動ダウンロードします。失敗した場合は **FFMPEG_INSTALL.md** を参照して手動でインストール（BtbN ビルド、winget、または `runtime\ffmpeg\bin` を PATH に追加）。
- **Linux / macOS**：`install.sh` は ffmpeg を自動ダウンロードしません。パッケージマネージャでインストールし、**FFMPEG_INSTALL.md** を参照してください。

---

## ライセンス/免責事項

これはローカルツールです。モデルのライセンスと使用について責任を負います。
