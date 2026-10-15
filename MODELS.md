# Models and settings

Fill the bracketed values on the machine that produced the paper's pairs, then commit.
For a model downloaded to a local directory, record the directory name and checksums
(`sha256sum config.json tokenizer_config.json *.safetensors`).

## Generator
- Model: Qwen2.5-7B-Instruct (https://huggingface.co/Qwen/Qwen2.5-7B-Instruct); report: Qwen2.5 Technical Report, arXiv:2412.15115
- Revision or local directory: [commit hash or directory name]
- Quantization: 4-bit by default (the script's default; `--no-4bit` disables it): [confirm for the run]
- Temperature: 0.2 (script default): [confirm]
- Maximum new tokens: 1536 (script default): [confirm]

## Validator
- Model: Llama-3.1-8B-Instruct (https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct); report: The Llama 3 Herd of Models, arXiv:2407.21783
- Revision or local directory: [commit hash or directory name]
- The script's default validator is BioMistral. The paper's pairs used Llama, so pass
  `--validator-model llama` (see `REPRODUCE.md`).

## Retrievers
- Dense: BAAI/bge-m3, revision [hash]
- Late interaction: jinaai/jina-colbert-v2, revision [hash]

## Software
- transformers 4.49.0, Python [version], CUDA [version]; record `pip freeze` in `requirements.txt`.
- Random seed 42 (global option, placed before the subcommand).

## Checksums
[paste output of sha256sum here]
