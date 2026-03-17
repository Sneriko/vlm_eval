# vlm-eval

Evaluate proprietary and open Vision Language Models (VLMs) through their APIs on historical Swedish handwriting datasets.

The project expects each sample to have:
- A page image (`.jpg`, `.png`, `.tif`, etc.)
- A corresponding PAGE XML file containing ground-truth transcription (`TextEquiv/Unicode`)

Evaluation metrics implemented:
- Bag-of-Words Precision
- Bag-of-Words Recall
- Bag-of-Words F1

## Why this architecture

This implementation uses lightweight API clients (`requests`) instead of LangChain. That keeps the evaluation path transparent and easy to adapt to many providers. Any model endpoint that is OpenAI-compatible can be evaluated with the same client.

Supported providers out of the box:
- `openai_compatible` (OpenAI and compatible hosted/self-hosted APIs)
- `anthropic`
- `gemini`
- `deepseek`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
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
```

## Run evaluation

```bash
vlm-eval run --config config.yaml
```

Outputs:
- Progress lines to stdout for each page and each folder (precision/recall/F1 by model)
- CSV at `output_csv` path including page-level, folder-level, and entire-testset rows (`level`/`scope_id`)
- JSON summary printed to stdout (entire-testset precision/recall/F1 by model)

## Notes on PAGE XML parsing

Ground truth is extracted from all `TextEquiv/Unicode` nodes, joined by line breaks.

## Development

```bash
pytest
```
