# vlm-eval

Evaluate proprietary and open Vision Language Models (VLMs) through APIs or local Hugging Face checkpoints on historical Swedish handwriting datasets.

The project expects each sample to have:
- A page image (`.jpg`, `.png`, `.tif`, etc.)
- A corresponding PAGE XML file containing ground-truth transcription (`TextEquiv/Unicode`)

Evaluation metrics implemented:
- Bag-of-Words Precision
- Bag-of-Words Recall
- Bag-of-Words F1

## Why this architecture

This implementation uses provider SDKs directly for OpenAI-compatible endpoints, Anthropic, Gemini, and DeepSeek, and uses Transformers directly for local Hugging Face VLM checkpoints, rather than an abstraction layer like LangChain. That keeps the evaluation path transparent while still using the vendors' and Hugging Face's maintained clients.

Supported providers out of the box:
- `openai_compatible` (OpenAI and compatible hosted/self-hosted APIs)
- `anthropic`
- `gemini`
- `deepseek`
- `huggingface` (local/open LVLM checkpoints via `transformers`, for example Gemma 3 and Qwen2.5-VL)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

To evaluate local Hugging Face LVLM checkpoints, install the optional Transformers stack:

```bash
pip install -e .[dev,huggingface]
```

## Dataset layout

The evaluator matches files by stem between archive-level images and PAGE XMLs found in a `page/` subfolder. XML files in sibling folders (such as `alto/`) are ignored.

```text
data/
  archive_1/
    page_0001.jpg
    page/
      page_0001.xml
    alto/
      page_0001.xml  # ignored
  archive_2/
    page_0002.tif
    page/
      page_0002.xml
```

## Configure models

Copy and edit the example config:

```bash
cp examples/config.example.yaml config.yaml
```

Set API keys in your environment, for example:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export DEEPSEEK_API_KEY=...
export LOCAL_VLM_API_KEY=dummy
export HF_TOKEN=...  # optional, for gated/private Hugging Face models such as Gemma
```


### Hugging Face open LVLMs

Use `provider: huggingface` to load a local Transformers checkpoint directly from Hugging Face. Public models can omit `api_key_env`; gated or private models can set `api_key_env: HF_TOKEN`. Generation uses the repo prompt plus the page image as a multimodal chat message.

Examples:

```yaml
models:
  - name: qwen2.5-vl-3b
    provider: huggingface
    model: Qwen/Qwen2.5-VL-3B-Instruct
    max_tokens: 2048
    temperature: 0
    device_map: auto
    torch_dtype: bfloat16

  - name: gemma-3-4b
    provider: huggingface
    model: google/gemma-3-4b-it
    api_key_env: HF_TOKEN
    max_tokens: 2048
    temperature: 0
    device_map: auto
    torch_dtype: bfloat16
```

Optional Hugging Face settings:
- `device_map`: passed to `from_pretrained` (defaults to `auto`). Use `cpu` for CPU-only smoke tests.
- `torch_dtype`: passed to `from_pretrained` (defaults to `auto`). Common GPU values are `bfloat16` or `float16`.
- `trust_remote_code`: defaults to `false`; set only for checkpoints that require custom model code.
- `model_kwargs`: extra keyword arguments forwarded to the model `from_pretrained` call, for example `attn_implementation: sdpa`.

## Run evaluation

```bash
vlm-eval run --config config.yaml
```

Outputs:
- Progress lines to stdout for each page and each folder (precision/recall/F1 by model)
- One CSV per configured model, derived from `output_csv` (for example `results_gpt-4-1-mini.csv`), each including page-level, folder-level, and entire-testset rows (`level`/`scope_id`)
- JSON summary printed to stdout (entire-testset precision/recall/F1 by model)

## Notes on PAGE XML parsing

Ground truth is extracted from all `TextEquiv/Unicode` nodes, joined by line breaks.

## Development

```bash
pytest
```

## Line-level Gemini CER evaluation script

For the `data/line_predictions.csv` format (line image path + TrOCR + ground truth), use:

```bash
python scripts/gemini_line_tasks.py \
  --input-csv data/line_predictions.csv \
  --guidelines-pdf data/tridis.pdf \
  --output-csv outputs/line_predictions_gemini_tasks.csv \
  --model gemini-3.1-pro
```

This runs six tasks per row:
1. Transcribe from image only.
2. Correct TrOCR using image + TrOCR text.
3. Correct TrOCR using TrOCR text only.
4. Correct TrOCR using TrOCR text + `tridis.pdf` guidelines.
5. Correct TrOCR using image + TrOCR text + `tridis.pdf` guidelines.
6. Transcribe from image + `tridis.pdf` guidelines.

The output CSV keeps all original columns and appends:
- `gemini_task{1..6}_transcription`
- `gemini_task{1..6}_cer` (per-line CER)
- A final `__SUMMARY__` row with `gemini_task{1..6}_dataset_cer`.

Notes:
- Set `GEMINI_API_KEY` (or use `--api-key-env`).
- If image paths in the CSV are absolute paths from another machine, use `--image-path-prefix` with a local folder containing the cropped line images.
