# Trajis SmartSRT (Modelos Portátiles, Sin Conexión)

Trajis SmartSRT ejecuta una canalización de traducción de subtítulos local usando **llama-cpp-python**.
Está diseñado para ser **portátil**: todas las dependencias viven dentro de esta carpeta.

✅ **Offline-first**: ejecute `install.bat` (Windows) o `./install.sh` (Linux/macOS) **una vez** con red para descargar e instalar todo; luego use `start.bat` o `./start.sh` para **arranque offline** cuando quiera.

❗ **Los modelos NO se descargan automáticamente.** Debe descargar manualmente los archivos GGUF y colocarlos en `./models/`.

---

## Resumen de la Canalización de Traducción

La canalización de traducción se divide en **6 ejecuciones** (A→B→C/D→E→F), ejecutadas secuencialmente. **Brief único**: un solo archivo actual `./work/brief.jsonl` se actualiza en cada etapa; antes de C/D/E se copia a `brief_v1.jsonl` / `brief_v2.jsonl` / `brief_v3.jsonl` (snapshots).

- **Run A**: Análisis de emoción/tono de audio (todas las líneas de subtítulos)
  - Extrae segmentos de audio del video para cada subtítulo
  - Analiza emoción, tono, intensidad y estilo de habla
  - Los resultados se guardan en `./work/audio_tags.jsonl`
  - **Nota**: El modelo de audio está **preempaquetado** y **estrechamente acoplado** con el código. **NO lo modifique ni reemplace.**

- **Run B**: El modelo principal genera el brief de traducción (todas las líneas de subtítulos)
  - Usa el modelo de razonamiento principal (GGUF, llama-cpp-python) para generar guía de traducción.
  - Entrada: Subtítulo en inglés + una línea de contexto anterior y siguiente + etiquetas de audio (todo en inglés).
  - Salida: JSON — **todo en inglés** (neutral): `meaning_tl`, `draft_tl`, `tl_instruction`, `idiom_requests`, `ctx_brief`, **`referents`**, **`tone_note`**, **`scene_brief`**, y opcionalmente **`disambiguation_note`**, **`rewrite_guidance`**. **Un need por etapa**: v1 solo **`need_vision`** (booleano). Opcional: `transliteration_requests`, `omit_sfx`; `reasons` (incluye PACK para Run F).
  - Los resultados se guardan en `./work/brief.jsonl` (brief actual único).

- **Run C**: Fallback de visión de un solo fotograma (opcional, activado condicionalmente)
  - Se activa cuando **`need_vision === true`** (del brief actual).
  - Antes de actualizar: copiar `brief.jsonl` → `brief_v1.jsonl`. Extrae un fotograma en el punto medio del rango de tiempo del subtítulo; el modelo de visión analiza escena/personajes/acciones.
  - Regenera el brief con pista de un solo fotograma → actualiza `./work/brief.jsonl`. **Un need por etapa**: v2 solo **`need_multi_frame_vision`** (booleano) por ítem.

- **Run D**: Fallback de visión de múltiples fotogramas (opcional, activado condicionalmente)
  - Se activa cuando **`need_multi_frame_vision === true`** (del brief actual).
  - Antes de actualizar: copiar `brief.jsonl` → `brief_v2.jsonl`. Extrae N fotogramas (configurable, predeterminado: 3) espaciados uniformemente; el modelo de visión analiza y fusiona descripciones.
  - Regenera el brief con pista de múltiples fotogramas → actualiza `./work/brief.jsonl`. **Un need por etapa**: v3 solo **`need_more_context`** (booleano); usado por Run E (expansión de contexto).

- **Run E**: Expansión de contexto (condicional)
  - Antes de actualizar: copiar `brief.jsonl` → `brief_v3.jsonl`. Para ítems con **`need_more_context === true`**, ejecuta stage2 con contexto **prev-3/next-3** y actualiza su brief; escribe `./work/brief.jsonl`. No hay salida del modelo en idioma objetivo; solo brief.

- **Run F**: Traducción final (todas las líneas de subtítulos)
  - Run F lee el **brief actual** (`brief.jsonl`, actualizado por E) y el PACK para producir los subtítulos completos en el idioma objetivo y escribir el SRT. **Config**: `config.PipelineConfig.run_e_scheme` (por defecto `"full"`). **UI**: menú desplegable Run F scheme. **Ejecución**: `pipeline_runs.run_final_translate()`; salida: `items.translated_text` y `work_dir/final_translations.jsonl`. **Escritura SRT**: `app.py` alinea por `(round(start_ms), round(end_ms))`, conserva etiquetas `<i>` del original.
  - **Un modelo a la vez**: solo MAIN o LOCAL está cargado; el glosario se aplica solo en la salida.
  - **Robustez**: todas las llamadas chat pasan por **chat_dispatch**. Las peticiones heavy pueden ejecutarse en un subproceso one-shot; en fallo, fallback in-process.

### Esquemas Run F (elegir por fuerza del modelo principal / modelo de localización)

En la UI seleccione el esquema en el menú **Run F scheme**. Opciones: `full` | `main_led` | `local_led` | `draft_first` (valor inválido hace fallback a `full`).

| Scheme | Cuándo usar (principal/local) | Phase1 (origen borrador) | Phase2 (pulido) |
|--------|---------------------|------------------------|-----------------|
| **Full** | Principal fuerte, local fuerte | Traducción por grupos MAIN → draft_map | Pulido LOCAL |
| **MAIN-led** | Principal fuerte, local débil | Traducción por grupos MAIN → draft_map | ninguno |
| **LOCAL-led** | Principal débil, local fuerte | PACK draft_tl → draft_map; opcional LOCAL rellenar slots idiom | Pulido LOCAL |
| **Draft-first** | Principal débil, local débil | PACK draft_tl → draft_map; opcional LOCAL rellenar slots idiom | ninguno |

- **Full**: Fase 1 — cargar MAIN (reason), construir grupos de oraciones, dividir por `group_translate_max_segments` (por defecto 4); por cada chunk llamar `stage_main_group_translate`; **se acepta salida MAIN parcial** (emparejar por **id**; segmentos faltantes usan PACK draft_tl o en_text). Fase 2 — cargar LOCAL, `local_polish` por chunks (`local_polish_chunk_size`, por defecto 60); solo claves de la petición que pasen comprobaciones de longitud/contaminación se aplican a draft_map. Final: glosario + strip_punctuation → `translated_text`.
- **MAIN-led**: Fase 1 — igual que Full (traducción por grupos MAIN → draft_map). Fase 2 — **omitida**. Final: glosario + strip_punctuation.
- **LOCAL-led**: Sin MAIN. `draft_map = _build_draft_map_from_pack(...)`. Si algún ítem tiene `idiom_requests`, cargar LOCAL, llamar `stage3_suggest_local_phrases` por línea, rellenar slots con `_fill_draft_with_suggestions` y actualizar draft_map. Luego cargar LOCAL y ejecutar `local_polish` por chunks. Final: glosario + strip_punctuation.
- **Draft-first**: Sin MAIN. Construir draft_map solo desde PACK; si hay idiom_requests, cargar LOCAL, obtener sugerencias y rellenar; **sin** pulido. Los modelos de localización débiles usan prompts **STRICT** y fallback raw_decode para evitar errores de formato.

**Alineación**: Toda la alineación en Run F es por **sub_id** y **marcas de tiempo (start_ms, end_ms)**; nunca por índice de lista.

**Política de idioma (sin mezclar entre etapas)**:
- **Run A–E**: Todos los prompts y todas las salidas del modelo son **solo en inglés**. Run A (audio), Run B/C/D (brief) y Run E (expansión de contexto) no reciben ni producen texto en idioma objetivo; el brief es inglés neutro para que Run F traduzca desde una única interfaz en inglés.
- **Run F**: Todas las **instrucciones (prompts) en inglés**; la entrada a los modelos principal y de localización es **inglés** (segmentos, tl_instruction, contexto). Solo la **salida del modelo** (segment_texts[].text, líneas pulidas, sugerencias de frases) está en el **idioma objetivo**.
- **Aplicación**: La salida del brief de Run B se sanitiza (p. ej. `tl_instruction` debe ser solo inglés). Run F usa `_tl_instruction_for_run_e()` para que la etapa de traducción siempre obtenga la locale objetivo correcta.

**Roles de prompt** (`model_prompts.csv`): MAIN (main_group_translate) se centra en el **ORIGEN (SOURCE)**; localización fuerte (p. ej. Llama-Breeze2-8B) se centra en el **idioma objetivo**, naturalizar/pulir; localización débil (p. ej. Breeze-7B, custom-localization) usa formato **STRICT**; en fallo de parse se usa raw_decode para extraer el primer `{...}`.

**Transliteración (音譯)**: Los nombres o términos que deben transliterarse en el idioma objetivo son tarea del **modelo de localización**; el **modelo principal** (Run B) los propone en PACK como `transliteration_requests` (array de cadenas). Run F Fase 2 (local_polish) recibe estos términos y añade al prompt de pulido «Transliterate (音譯) in target language for these terms: …» para que LOCAL emita las formas transliteradas.

**CC / SFX (狀聲詞)**: El **modelo principal** (Run B) filtra efectos de sonido y onomatopeyas (p. ej. `[laughter]`, `[sigh]`, `*gasps*`). Puede poner `omit_sfx: true` y `draft_tl` vacío para líneas solo SFX; para diálogo+SFX pone solo el diálogo en `draft_tl`. Run F aplica omit_sfx tras construir draft_map, por lo que esas líneas quedan con salida vacía.

**Config Run F** (`config.py`): `run_e_scheme` (UI: Run F scheme), `group_translate_max_segments` (por defecto 4), `local_polish_chunk_size` (por defecto 60), `strip_punctuation`, `strip_punctuation_keep_decimal`, `strip_punctuation_keep_acronym`.

### Características Principales

- **Carga de Modelo Único**: Solo se carga un modelo a la vez (audio, razón, visión o traducción)
- **Reanudable**: Cada ejecución guarda resultados intermedios en `./work/` (formato JSONL)
- **Resistente a Errores**: Si falla la visión/audio, la canalización continúa con el mejor brief disponible
- **Seguimiento de Progreso**: La barra de progreso muestra el paso actual y el porcentaje de finalización

---

## Entrada de video (nota sobre FFmpeg)

El componente **Video** incorporado de Gradio realiza procesamiento del lado del servidor que requiere un ejecutable externo **`ffmpeg`**.
Si `ffmpeg` no está disponible, puede obtener errores como **"Executable 'ffmpeg' not found"**.

Para mantener este proyecto **completamente portátil** (sin instalaciones a nivel del sistema), este repositorio usa una entrada **File** para el video en su lugar.

- Necesita un archivo de video para **Run A (audio)** y **Run C/D (visión)**.
- Se usa OpenCV (opencv-python) para capturar fotogramas para Vision, y ffmpeg se usa para extraer segmentos de audio.
- **ffmpeg**: **Windows** – si ffmpeg no está en PATH, `install.bat` descarga una versión portátil en `runtime\ffmpeg`. **Linux / macOS** – `install.sh` solo comprueba ffmpeg en PATH; instálelo manualmente y vea **FFMPEG_INSTALL.md** si falta.

Si está usando un zip anterior que aún usa el componente Video, actualice al zip más reciente o instale FFmpeg y agréguelo a PATH.

## Instalación y arranque (offline-first)

Este proyecto está pensado para **uso offline**: ejecute **install** una vez (con red), luego use **start** cuando quiera (sin red).

**⚠️ IMPORTANTE - Usuarios de GPU NVIDIA:**
- **Instale CUDA Toolkit 12.9** (o 12.x) **ANTES** de ejecutar `install.bat` / `install.sh`
- Descarga: https://developer.nvidia.com/cuda-downloads
- La rueda precompilada de llama-cpp-python requiere que CUDA esté instalado primero para la aceleración por GPU

1. Extraiga esta carpeta en cualquier lugar (ejemplo: `G:\Trajis SmartSRT`).

2. **Instalar (una vez, con red)** — descarga e instala todo:
   - **Windows**: doble clic en `install.bat` → Python portátil, venv, todas las dependencias Python (base + audio Run A + vídeo), opcional CUDA PyTorch si hay GPU, ffmpeg en `runtime\ffmpeg` si no está en PATH, modelo de audio Run A en `models\audio`, rueda precompilada llama-cpp-python, config, BOM. Los modelos GGUF son manuales (véase abajo).
     - **Uso estimado de disco tras instalar**: ~6-8 GB (solo CPU: ~4-5 GB; con CUDA PyTorch + ffmpeg: ~6-8 GB)
   - **Linux / macOS**: ejecute `./install.sh` (misma idea: .venv, deps, modelo de audio, rueda precompilada llama-cpp-python, config, BOM). Si hace falta: `chmod +x install.sh`
     - **Uso estimado de disco tras instalar**: ~4-5 GB (excluyendo Python del sistema y ffmpeg)

3. **Arrancar (offline)** — sin descargas, sin red:
   - **Windows**: doble clic en `start.bat` → comprueba .venv y archivos de modelo, luego lanza la UI.
   - **Linux / macOS**: ejecute `./start.sh`. Si hace falta: `chmod +x start.sh`

- **Desinstalar**: ejecute `uninstall.bat` (Windows) o `./uninstall.sh` (Linux/macOS) para eliminar entorno, venv y cachés dentro de esta carpeta. Si hace falta: `chmod +x uninstall.sh`

**Soporte de GPU:**

- **GPUs NVIDIA**: soporte CUDA 12.x (CUDA 12.9 recomendado; series RTX 20/30/40/50, serie GTX 16 y más recientes)
- **GPUs AMD**: soporte ROCm (experimental, requiere configuración manual)
- **GPUs Intel Arc**: soporte oneAPI (experimental, requiere configuración manual)
- **CPU**: optimizado para CPUs Intel (no requiere conjunto de instrucciones AVX-512), funciona en todos los procesadores x86-64 modernos

**Opcional – instalar solo dependencias de audio (Linux / macOS):**

- Ejecutar `install.bat` o `install.sh` ya instala las dependencias de Run A (audio). Use `./scripts/install_audio_deps.sh` solo si necesita reinstalar dependencias de audio (torch, transformers, soundfile, scipy) sin hacer la instalación completa. Requiere Python 3 y opcionalmente un `.venv` activo.

Todo permanece dentro de esta carpeta (portátil/aislado).

---

## Compatibilidad de modelos y estructura de carpetas (requerido)

Todos los modelos de **texto y visión** que usa esta aplicación deben ser **GGUF** y compatibles con **llama-cpp-python**. Usted proporciona los archivos; la aplicación no los descarga.

Cree y use esta estructura de carpetas:

```
models/
  main/     ← Modelo de razonamiento principal (Run B); uno o más archivos .gguf
  local/    ← Modelo de localización/traducción (Run E); uno o más archivos .gguf
  vision/   ← Modelo de visión opcional (Run C/D); .gguf principal + mmproj .gguf
  audio/    ← Modelo de audio Run A (descargado por el script de instalación o en la primera ejecución)
```

### Compatibilidad

- **Modelos principal y de localización**: Cualquier modelo **GGUF** que funcione con llama-cpp-python (modelos instruct/chat con plantilla de chat). Coloque los archivos en `./models/main/` y `./models/local/` respectivamente. Si la cuantización está **fragmentada** (varios .gguf), descargue **todos los fragmentos** y colóquelos en la misma carpeta.
- **Modelos de visión (opcional)**: Cualquier modelo de **visión GGUF** soportado por llama-cpp-python (modelo principal + mmproj). Coloque ambos archivos en `./models/vision/`. La aplicación detecta el tipo por el nombre del archivo. Puede fijar nombres exactos en `config.json` en `vision.text_model` y `vision.mmproj_model`.
- **Audio (Run A)**: El modelo de emoción Run A se descarga de Hugging Face Hub en la primera ejecución (sin GGUF local). Usa Transformers `audio-classification`; dependencias: `torch`, `transformers`, `soundfile`, `scipy`.

### Parámetros y cuantización (guía genérica)

- **Cuantización**: Cuantos más ligeros (p. ej. **Q4_K_M**) menos VRAM y más rápido; más pesados (**Q5_K_M**, **Q6_K**, **Q8_0**) mejor calidad pero más VRAM y disco. Elija según su GPU/RAM.
- **Tamaño del modelo**: Más parámetros (p. ej. 14B, 7B) requieren más VRAM y RAM. Los modelos se cargan **uno a la vez**, así que la VRAM la define el **modelo único más grande** que use.
- **Contexto**: Un `n_ctx_*` mayor (p. ej. 8192) mejora el contexto largo pero aumenta la VRAM (caché KV). Si hay OOM, reduzca `n_ctx_*` o `n_gpu_layers_*`.

### Puntos de partida sugeridos para config.json (ajuste según su hardware)

- **16 GB VRAM**: `n_ctx_reason=8192`, `n_ctx_translate=4096`, `n_gpu_layers_reason=60`, `n_gpu_layers_translate=60`
- **12 GB VRAM**: `n_ctx_reason=4096`, `n_ctx_translate=2048`, `n_gpu_layers_reason=50`, `n_gpu_layers_translate=50`
- **8 GB VRAM**: `n_ctx_reason=2048`, `n_ctx_translate=2048`, `n_gpu_layers_reason=35`, `n_gpu_layers_translate=35`
- **Solo CPU / poca RAM**: Prefiera Q4_K_M (o más ligero) y contexto pequeño; reduzca `n_gpu_layers_*` o póngalo a 0 para usar solo CPU.

---

## config.json

Cuando ejecuta `install.bat` (o `install.sh`), se ejecuta `scripts/plan_models.py` para crear `config.json` si no existe. `start.bat` / `start.sh` no crean config; solo arrancan la aplicación.
En uso normal **no es necesario mantener config.json**; los modelos se detectan desde `./models`. Edítelo solo si necesita ajustes avanzados (VRAM, tamaños de lote, fallbacks).

---

## Directorio de Trabajo (Resultados Intermedios)

Todos los resultados intermedios se guardan en el directorio `./work/` en formato JSONL:

- `audio_tags.jsonl` - Resultados de Run A (análisis de emoción/tono de audio)
- `brief.jsonl` - Brief actual (Run B escribe; C/D/E actualizan; Run F lee)
- `brief_v1.jsonl` - Snapshot antes de Run C (copia antes de que C actualice)
- `brief_v2.jsonl` - Snapshot antes de Run D
- `brief_v3.jsonl` - Snapshot antes de Run E (expansión de contexto)
- `vision_1frame.jsonl` - Resultados de Run C (análisis de visión de un solo fotograma)
- `vision_multiframe.jsonl` - Resultados de Run D (análisis de visión de múltiples fotogramas)
- `final_translations.jsonl` - Resultados de Run F (texto traducido final, nuevo formato)

**Compatibilidad de Formato JSONL:**

La canalización admite tanto el **formato antiguo** (usando `idx` para alineación) como el **formato nuevo** (usando `sub_id` para alineación):

- **Formato antiguo**: Usa `idx` (índice entero) para identificar líneas de subtítulos
  - Ejemplo: `{"idx": 0, "start_ms": 1000, "end_ms": 2000, ...}`
- **Formato nuevo**: Usa `sub_id` (identificador único basado en hash) para garantizar la alineación de datos
  - Ejemplo: `{"sub_id": "a1b2c3d4_0", "start_ms": 1000, "end_ms": 2000, ...}`
  - `sub_id` se genera a partir de `hash(start_ms, end_ms, text_raw)` para garantizar la consistencia entre runs

La canalización detecta automáticamente el formato y maneja la conversión cuando es necesario. Los nuevos runs usarán el formato `sub_id` para garantizar una mejor integridad de datos.

**Funcionalidad de reanudación**: Si existe un archivo JSONL y tiene el número correcto de entradas, la canalización lo cargará automáticamente y omitirá ese run. La canalización admite reanudación tanto desde el formato antiguo (`idx`) como desde el nuevo (`sub_id`).

**Reanudación manual**: Puede eliminar archivos JSONL específicos para volver a ejecutar solo esos pasos.

---

## Uso de la UI

1. **Subir archivos**: Video (MKV/MP4) y SRT (subtítulos en inglés)
2. **Seleccionar modo de ejecución**: `all` (A→B→(C/D)→E→F, predeterminado) | **A** (audio) | **B** (brief) | **C** (visión 1 fotograma) | **D** (visión múltiples fotogramas) | **E** (expansión de contexto) | **F** (traducción)
3. **Run F scheme** (menú desplegable): elija por fuerza del modelo principal / de localización — **Full** | **MAIN-led** | **LOCAL-led** | **Draft-first**. Véase **Esquemas Run F** en Resumen de la Canalización.
4. **Fallbacks opcionales** (casillas en la UI):
   - **Habilitar fallback de visión (Run C/D)**: cuando esté marcado y el brief tenga **need_vision** / **need_multi_frame_vision**, se ejecuta visión de un fotograma (C) o múltiples (D) y se actualiza el brief. Requiere modelo de visión GGUF local.
   - **Habilitar fallback de expansión de contexto (Run E)**: cuando esté marcado, los ítems con **need_more_context** obtienen contexto prev-3/next-3 y brief actualizado antes de Run F. Recomendado.
   - **Max frames per subtitle (Run D)** / **Frame offsets**: número y posiciones de fotogramas (predeterminado: 1–4).
5. **Hacer clic en "🚀 Translate"** y monitorear el progreso
6. **Descargar** el archivo SRT traducido cuando esté completo
7. **Restablecer**: Haga clic en **"Reset"** para borrar todas las entradas, salidas y registro y restaurar los valores predeterminados para comenzar una nueva traducción

**Detalles de la UI**: El panel de registro muestra las **entradas más recientes arriba**. `model_prompts.csv` se lee/escribe en UTF-8 con BOM; `start.bat` / `start.sh` ejecutan `ensure_csv_bom.py` al iniciar para mantener la codificación correcta.

---

## Personalización de Prompts de Modelo (model_prompts.csv)

La canalización de traducción usa prompts definidos en `model_prompts.csv`. El prompt de cada modelo se empareja automáticamente por **nombre de archivo del modelo** (coincidencia de subcadena que no distingue mayúsculas y minúsculas). El archivo debe estar en **UTF-8 con BOM**; `start.bat` y `start.sh` ejecutan `scripts/ensure_csv_bom.py` al iniciar para asegurarlo.

### Alineación oficial de prompts del modelo

Los prompts están diseñados para seguir el formato de chat **oficial** y las recomendaciones de cada familia de modelos, de modo que el comportamiento sea predecible y compatible:

- **Qwen2.5 (ChatML)**: Rol system + rol user; JSON Mode para salida estructurada. Las plantillas usan `chat_format=chatml` e instrucciones estrictas de «solo JSON válido, sin markdown» según el uso oficial de Qwen.
- **Gemma 2 (p. ej. TranslateGemma)**: **Sin rol system**; toda la instrucción va en el primer turno de user. El backend fusiona el contenido de system en el mensaje de user cuando `chat_format=gemma`, de modo que el modelo solo ve un turno de user.
- **Mistral / Llama 2 (p. ej. Breeze, Llama-Breeze2)**: Estilo `[INST]`; el system prompt se antepone al primer bloque `[INST]`. Se usa en los roles `local_polish` y `localization` con salida STRICT JSON cuando se requiere.
- **Vision (Moondream, LLaVA)**: Los prompts se aplican en código por handler; el formato de chat se detecta automáticamente por el nombre del archivo del modelo de visión. La salida es siempre descripción visual **en inglés** únicamente (no subtítulos).

La columna **notes** del CSV documenta si el rol es «Run A~D todo en inglés» o «Run E: salida solo en idioma objetivo» para que las filas personalizadas mantengan las mismas fronteras de idioma.

### Coincidencia de Nombre de Modelo

- **Cómo funciona**: La aplicación extrae el nombre de archivo del modelo (p. ej., `my-main-model-q5_k_m.gguf`) y lo compara con la columna CSV `model_name`.
- **Regla de coincidencia**: Si el nombre de archivo **contiene** el CSV `model_name` (sin distinguir mayúsculas y minúsculas), es una coincidencia.
  - Ejemplo: `my-main-model-q5_k_m.gguf` coincide con `my-main-model`
  - Ejemplo: `my-local-model-00001-of-00002.gguf` coincide con `my-local-model`
- **Qué completar**: Use una **subcadena única** que aparezca en el nombre de archivo de su modelo. Por lo general, el nombre del modelo base sin sufijo de cuantización funciona.

### Guía de Columnas CSV

| Columna | Descripción | Ejemplo |
|--------|-------------|---------|
| `model_name` | Subcadena para coincidir en el nombre de archivo (sin distinguir mayúsculas y minúsculas) | `my-main-model` |
| `role` | `main` (Run B), `main_assemble` (Run E Stage4), `localization` (Run E), o `vision` (Run C/D) | `localization` |
| `source_language` | Idioma de entrada (generalmente `English`) | `English` |
| `target_language` | Idioma de salida (Código de localización: `en`, `zh-TW`, `zh-CN`, `ja-JP`, `es-ES`) | `zh-TW` |
| `chat_format` | Plantilla de chat del modelo (`chatml`, `llama-3`, `mistral-instruct`, `moondream`) | `chatml` |
| `system_prompt_template` | Prompt del sistema (definición de rol) | Ver ejemplos a continuación |
| `user_prompt_template` | Prompt del usuario con marcadores de posición | Ver ejemplos a continuación |
| `notes` | Descripción (inglés) | `Localization model for Traditional Chinese (Taiwan)` |

### Marcadores de Posición

Use estos marcadores de posición en `user_prompt_template`:

**Marcadores de posición Run B (main):**
- `{line}` → Línea de subtítulo en inglés actual
- `{context}` → Contexto completo (Prev-1, Current, Next-1, Prev-More, Next-More, Visual Hint)

**Marcadores de posición Run E (localization):**
- `{tl_instruction}`, `{requests_json}`, `{target_language}` (sugerencias de frases idiomáticas)

**Marcadores de posición Run E (main_assemble)** – Stage4 ensamblado en una línea:
- `{target_language}`, `{line_en}`, `{ctx_brief}`, `{draft_prefilled}`, `{suggestions_json}`

**Marcadores de posición Run C/D (vision):**
- `{line}` → Línea de subtítulo en inglés actual

### Estilos de Prompt: Modelos Base vs Instruct

#### Modelos Base (No Instruct)
- **Características**: Prompts más simples y directos sin formato de instrucción estructurado
- **Cuándo usar**: Su modelo es un modelo base/completado (no ajustado para instrucciones)
- **Estilo**: Preguntas directas o descripciones de tareas simples
- **Ejemplo** (Run B):
  ```
  Analyze this subtitle line and explain what it really means in plain English.
  
  Subtitle: {line}
  Context: {context}
  
  Explain the meaning, including any idioms, jokes, tone, or implied meaning.
  ```

#### Modelos Instruct
- **Características**: Formato de instrucción estructurado con reglas numeradas y definición de tarea clara
- **Cuándo usar**: Su modelo está ajustado para instrucciones (Instruct, Chat, etc.)
- **Estilo**: Estructurado con reglas, pasos numerados, definiciones claras de entrada/salida
- **Ejemplo** (Run B):
  ```
  You are stage 2 (reasoning) in a multi-stage subtitle translation pipeline.
  - Input: one English subtitle line plus nearby context.
  - Output: ENGLISH ONLY: a clear, unambiguous explanation...
  - Do NOT translate to any target language here.
  
  Subtitle line: {line}
  Context (previous/next lines): {context}
  ```

### Ejemplos en CSV

El CSV incluye filas de ejemplo para cada rol:

1. **`(custom-main-base)`** - Ejemplo de modelo Base para Run B
2. **`(custom-main-instruct)`** - Ejemplo de modelo Instruct para Run B
3. **`(custom-localization-base)`** - Ejemplo de modelo Base para Run E
4. **`(custom-localization-instruct)`** - Ejemplo de modelo Instruct para Run E
5. **`(custom-vision-base)`** - Ejemplo de modelo Base para Vision
6. **`(custom-vision-instruct)`** - Ejemplo de modelo Instruct para Vision

### Agregar Su Propio Modelo

1. **Copie una fila de ejemplo** (p. ej., `(custom-main-instruct)`)
2. **Cambie `model_name`** para que coincida con la subcadena del nombre de archivo de su modelo
3. **Establezca `role`** (`main`, `localization`, o `vision`)
4. **Establezca `target_language`** a uno de estos códigos de localización:
   - `en` - Inglés (para modelos main Run B)
   - `zh-TW` - Chino tradicional (Taiwán)
   - `zh-CN` - Chino simplificado (Continental)
   - `ja-JP` - Japonés
   - `es-ES` - Español
   - U otros códigos de localización IETF según sea necesario
5. **Establezca `chat_format`** para que coincida con la plantilla de chat de su modelo:
   - `chatml` - muchos modelos instruct/chat modernos
   - `llama-3` - Modelos Llama 3
   - `mistral-instruct` - Modelos Mistral
   - `moondream` - algunos modelos de visión
6. **Escriba `system_prompt_template`** (definición de rol, generalmente 1-2 oraciones)
   - Para modelos de localización: Use `[target_language]` como marcador de posición si desea que el prompt mencione el idioma objetivo de manera genérica
7. **Escriba `user_prompt_template`** (tarea con marcadores de posición)
   - Use estilo Base para modelos base
   - Use estilo Instruct para modelos ajustados para instrucciones
   - Para modelos de localización: Reemplace `[target_language]` con el código de localización real (p. ej., `zh-TW`, `ja-JP`) en el texto de su prompt
8. **Complete `notes`** (descripción en inglés)

### Notas Importantes

- **Fronteras de idioma**: **Run A–D** (audio, brief v1/v2/v3, visión): los prompts y la salida del modelo deben ser **solo en inglés**. **Run E** (main_group_translate, local_polish, localization, main_assemble): los prompts están en **inglés**; solo la **salida** (líneas traducidas, sugerencias de frases) está en el idioma objetivo. No ponga instrucciones en idioma objetivo (p. ej. chino o japonés) en los prompts de Run E—use inglés (p. ej. «Output ONLY the translated subtitle in the target language (locale: zh-TW).») para que el idioma del prompt no se mezcle con el de la salida.
- **Idioma del prompt**: Para Run B/C/D use prompts solo en inglés; para Run E use instrucciones en inglés y espere salida en idioma objetivo del modelo.
- **Formato de chat**: Debe coincidir con la plantilla de chat de su modelo. El formato incorrecto puede causar salida deficiente o errores.
  - **Para modelos de visión**: El formato de chat lo detecta automáticamente `LocalVisionModel` según el nombre del archivo del modelo. El campo `chat_format` del CSV es sobre todo para documentación.
- **Marcadores de posición**: Siempre use los nombres exactos de los marcadores de posición (`{line}`, `{context}`, `{target_language}`, etc.). Se reemplazan automáticamente.
- **Salida Run B/C/D (brief)**: Debe solicitar JSON con `target_language`, `tl_instruction`, `meaning_tl`, `draft_tl`, `idiom_requests`, `ctx_brief`, referents, tone_note, scene_brief — **todo en inglés** (brief neutral). **Un need por etapa**: **v1** solo **`need_vision`**; **v2** solo **`need_multi_frame_vision`**; **v3** solo **`need_more_context`**. Opcional: `plain_en`, `idiom_flag`, `transliteration_requests`, `omit_sfx`; `notes` puede contener PACK para Run E.

---

## Solución de Problemas

### "Faltan archivos de modelo requeridos"
Ejecute `start.bat`, abrirá este README y le dirá qué archivos faltan.

### GPU no detectada o rendimiento lento
Este proyecto incluye **ruedas precompiladas de llama-cpp-python** optimizadas para GPUs NVIDIA (CUDA 12.x) y CPUs Intel.
- **GPUs NVIDIA**: asegúrese de tener instalado el controlador de GPU más reciente. La aplicación detectará y usará CUDA automáticamente.
- **CPUs Intel**: la versión de CPU está optimizada para procesadores Intel modernos y no requiere el conjunto de instrucciones AVX-512.
- **GPUs AMD/Intel Arc**: soporte experimental disponible pero requiere configuración manual (no incluido en ruedas precompiladas).

### Advertencia de "symlink" de Windows
Esta advertencia proviene del caché de Hugging Face. Como este proyecto ya no descarga modelos automáticamente, puede ignorarla.

### Errores del modelo de audio (Run A)
Run A usa el modelo de Hugging Face **ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition**. Ejecute `pip install -r requirements_base.txt` (torch, transformers, soundfile, scipy). La primera ejecución descargará el modelo del Hub. La extracción de audio requiere ffmpeg en PATH.

### ffmpeg no encontrado
- **Windows**: `install.bat` descarga ffmpeg en `runtime\ffmpeg` cuando no está en PATH. Si falla, vea **FFMPEG_INSTALL.md** para descarga manual e instalación (builds BtbN, winget, o agregar `runtime\ffmpeg\bin` a PATH).
- **Linux / macOS**: `install.sh` no descarga ffmpeg automáticamente; instálelo con su gestor de paquetes y vea **FFMPEG_INSTALL.md** si hace falta.

---

## Licencia/Descargo de Responsabilidad

Esta es una herramienta local. Usted es responsable de la licencia y el uso del modelo.
