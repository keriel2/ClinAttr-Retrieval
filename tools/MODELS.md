# Models and settings

Models were loaded from local copies of the public Hugging Face releases named below.

## Generator
- Model: Qwen2.5-7B-Instruct (https://huggingface.co/Qwen/Qwen2.5-7B-Instruct); report: Qwen2.5 Technical Report, arXiv:2412.15115
- Quantization: 4-bit by default (the script's default; `--no-4bit` disables it)
- Temperature: 0.2 (script default)
- Maximum new tokens: 1536 (script default)

## Validator
- Model: Llama-3.1-8B-Instruct (https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct); report: The Llama 3 Herd of Models, arXiv:2407.21783
- The script's default validator is BioMistral. The paper's pairs used Llama, so pass
  `--validator-model llama` (see `REPRODUCE.md`).

## Retrievers
- Dense: BAAI/bge-m3
- Late interaction: jinaai/jina-colbert-v2

## Software
- Python 3.10, CUDA 12.1, torch 2.4.1, transformers 4.49.0; all versions are pinned in `requirements.txt`.
- Random seed 42 (global option, placed before the subcommand).
