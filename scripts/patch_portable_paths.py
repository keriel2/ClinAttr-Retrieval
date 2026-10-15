#!/usr/bin/env python3
"""
patch_portable_paths.py   make clinattr_multilingual_retrieval.py portable.

Replaces the hard-coded server paths with environment variables and makes the default
validator match the paper (Llama-3.1-8B-Instruct instead of BioMistral).

    CLINATTR_ROOT   data root (default ./clinattr_data)
    CLINATTR_QWEN, CLINATTR_LLAMA, CLINATTR_BIOMISTRAL   local model directories
                    (default: the Hugging Face model id)

Usage:
    python3 patch_portable_paths.py clinattr_multilingual_retrieval.py     # edits in place, keeps .bak
"""
import ast
import re
import shutil
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
shutil.copyfile(path, path + ".bak")

repl = [
    ('DATA_ROOT = Path("/home/share/henry/dataset")',
     'DATA_ROOT = Path(os.environ.get("CLINATTR_ROOT", "./clinattr_data"))'),
    ('QWEN_PATH = "/home/share/henry/model_download/Qwen2.5-7B-Instruct"',
     'QWEN_PATH = os.environ.get("CLINATTR_QWEN", "Qwen/Qwen2.5-7B-Instruct")'),
    ('LLAMA_PATH = "/home/share/henry/model_download/Llama-3.1-8B-Instruct"',
     'LLAMA_PATH = os.environ.get("CLINATTR_LLAMA", "meta-llama/Llama-3.1-8B-Instruct")'),
    ('BIOMISTRAL_PATH = "/home/share/henry/model_download/BioMistral-7B"',
     'BIOMISTRAL_PATH = os.environ.get("CLINATTR_BIOMISTRAL", "BioMistral/BioMistral-7B")'),
    ('        choices=["qwen", "llama", "biomistral"],\n        default="biomistral",\n    )\n    generate.add_argument("--generator-device"',
     '        choices=["qwen", "llama", "biomistral"],\n        default="llama",\n    )\n    generate.add_argument("--generator-device"'),
]
for old, new in repl:
    if old not in src:
        print("NOT FOUND (already patched or file differs):", old[:60])
        continue
    src = src.replace(old, new)

# docstring examples that mention the server paths
src = src.replace("/home/share/henry/dataset", "$CLINATTR_ROOT")
src = src.replace("/home/share/henry/model_download", "$CLINATTR_MODELS")
ast.parse(src)  # refuse to write a file that does not parse
open(path, "w", encoding="utf-8").write(src)
left = re.findall(r"/home/[a-z]+/[^\s\"']*", src)
print("patched; remaining absolute paths:", left or "none")
