#!/usr/bin/env python3
"""
ClinAttr-ColBERT: attribute-preserving multilingual clinical retrieval
=====================================================================

A single-file research implementation for:

1. Downloading and normalising CUREv1 (English/French/Spanish queries
retrieving English clinical evidence passages).
2. Auditing clinically decisive attributes in queries and passages.
3. Running BM25 and BGE-M3 first-stage retrieval.
4. Reranking dense candidates with Jina-ColBERT-v2 late interaction.
5. Applying a clinical-attribute compatibility layer.
6. Generating controlled counterfactual query-passage pairs with a local
instruction-tuned LLM, validating them with BioMistral, and exporting them
for mandatory human review.
7. Training a lightweight pairwise attribute gate on approved pairs.
8. Evaluating ordinary retrieval and Clinical Attribute Preservation Rate
(CAPR), including per-language and per-attribute results.
9. Optionally using Qwen, Llama, or BioMistral as listwise rerank baselines.

IMPORTANT RESEARCH RULE
-----------------------
Automatically generated counterfactuals are never treated as gold data.
The generation command always writes approved=0. A medically qualified
reviewer must inspect each row and manually set approved=1 before training
or evaluation.

Default paths supplied for this project
---------------------------------------
Data root:
    /home/share/henry/dataset

Local LLMs:
    Qwen:       /home/share/henry/model_download/Qwen2.5-7B-Instruct
    Llama:      /home/share/henry/model_download/Llama-3.1-8B-Instruct
    BioMistral: /home/share/henry/model_download/BioMistral-7B

Retrievers downloaded from Hugging Face unless local paths are supplied:
    Dense: BAAI/bge-m3
    Late interaction: jinaai/jina-colbert-v2

Recommended experiment order
----------------------------
1. Dependency and path check:
    python clinattr_multilingual_retrieval.py doctor

2. Prepare CURE:
    python clinattr_multilingual_retrieval.py prepare

3. Attribute audit:
    python clinattr_multilingual_retrieval.py audit

4. Pilot dense indexes for one domain:
    python clinattr_multilingual_retrieval.py index-dense \
        --domains pulmonology

5. Pilot evaluation:
    python clinattr_multilingual_retrieval.py evaluate-cure \
        --domains pulmonology --languages en fr es --max-queries 20 \
        --methods bm25,dense,late,attr-rule

6. Generate candidate counterfactual pairs:
    python clinattr_multilingual_retrieval.py generate-pairs \
        --domains pulmonology --languages en fr es --num-pairs 120

7. Manually review:
    /home/share/henry/dataset/clinattr_project/counterfactuals/
        counterfactual_candidates.csv

8. Train gate after setting approved=1:
    python clinattr_multilingual_retrieval.py train-gate \
        --pairs-csv /home/share/henry/dataset/clinattr_project/counterfactuals/counterfactual_candidates.csv

9. Evaluate approved pairs:
    python clinattr_multilingual_retrieval.py evaluate-pairs \
        --pairs-csv /home/share/henry/dataset/clinattr_project/counterfactuals/counterfactual_candidates.csv \
        --gate-checkpoint /home/share/henry/dataset/clinattr_project/models/attribute_gate.pt

10. Full CURE evaluation:
    python clinattr_multilingual_retrieval.py index-dense --domains all
    python clinattr_multilingual_retrieval.py evaluate-cure \
        --domains all --languages en fr es \
        --methods bm25,dense,late,attr-rule,attr-gate \
        --gate-checkpoint /home/share/henry/dataset/clinattr_project/models/attribute_gate.pt

The program intentionally separates model development from test evaluation:
counterfactual pairs used for gate training should be split by pair_id into
train/development/test partitions before reporting final CAPR.

Python 3.10+ is recommended.
"""

from __future__ import annotations

import argparse
import ast
import csv
import difflib
import gc
import hashlib
import importlib.util
import json
import logging
import math
import os
import random
import re
import shutil
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Project constants
# ---------------------------------------------------------------------------

DATA_ROOT = Path("/home/share/henry/dataset")
PROJECT_ROOT = DATA_ROOT / "clinattr_project"
HF_CACHE = DATA_ROOT / "hf_cache"

QWEN_PATH = "/home/share/henry/model_download/Qwen2.5-7B-Instruct"
LLAMA_PATH = "/home/share/henry/model_download/Llama-3.1-8B-Instruct"
BIOMISTRAL_PATH = "/home/share/henry/model_download/BioMistral-7B"

DEFAULT_DENSE_MODEL = "BAAI/bge-m3"
DEFAULT_LATE_MODEL = "jinaai/jina-colbert-v2"
CURE_DATASET_ID = "clinia/CUREv1"

DOMAINS = [
    "dentistry_and_oral_health",
    "dermatology",
    "gastroenterology",
    "genetics",
    "neuroscience_and_neurology",
    "orthopedic_surgery",
    "otorhinolaryngology",
    "plastic_surgery",
    "psychiatry_and_psychology",
    "pulmonology",
]
LANGUAGES = ["en", "fr", "es"]

LLM_PATHS = {
    "qwen": QWEN_PATH,
    "llama": LLAMA_PATH,
    "biomistral": BIOMISTRAL_PATH,
}

FEATURE_NAMES = [
    "dense_score",
    "late_score",
    "negation",
    "dosage",
    "frequency",
    "stage",
    "subtype",
    "temporal",
    "severity",
    "attribute_coverage",
]

ATTRIBUTE_FAMILIES = [
    "negation",
    "dosage",
    "frequency",
    "stage",
    "subtype",
    "temporal",
    "severity",
]

REQUIRED_PACKAGES = {
    "torch": "torch",
    "transformers": "transformers",
    "datasets": "datasets",
    "sentence_transformers": "sentence-transformers",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "rank_bm25": "rank-bm25",
    "ragatouille": "ragatouille",
    "bitsandbytes": "bitsandbytes",
    "einops": "einops",
}

# ---------------------------------------------------------------------------
# Logging and generic utilities
# ---------------------------------------------------------------------------


def configure_logging(log_file: Optional[Path] = None, verbose: bool = False) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_project_dirs() -> Dict[str, Path]:
    paths = {
        "root": PROJECT_ROOT,
        "normalized": PROJECT_ROOT / "cure_normalized",
        "queries": PROJECT_ROOT / "cure_normalized" / "queries",
        "corpus": PROJECT_ROOT / "cure_normalized" / "corpus",
        "qrels": PROJECT_ROOT / "cure_normalized" / "qrels",
        "indices": PROJECT_ROOT / "indices",
        "dense_indices": PROJECT_ROOT / "indices" / "dense",
        "counterfactuals": PROJECT_ROOT / "counterfactuals",
        "results": PROJECT_ROOT / "results",
        "models": PROJECT_ROOT / "models",
        "cache": PROJECT_ROOT / "cache",
        "logs": PROJECT_ROOT / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    return paths


def parse_csv_list(value: str, allowed: Optional[Sequence[str]] = None) -> List[str]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    if items == ["all"] and allowed is not None:
        return list(allowed)
    if allowed is not None:
        invalid = [x for x in items if x not in allowed]
        if invalid:
            raise ValueError(f"Invalid values {invalid}; allowed={list(allowed)}")
    return items


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def normalise_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).strip()


def simple_tokens(text: str) -> List[str]:
    return re.findall(r"[\wÀ-ÿµμ]+", normalise_unicode(text).lower(), flags=re.UNICODE)


def zscore(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values)
    if stdev < 1e-9:
        return [0.0 for _ in values]
    return [(float(x) - mean) / stdev for x in values]


def minmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        return [0.0 for _ in values]
    return [(float(x) - lo) / (hi - lo) for x in values]


def batched(items: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Dependency and environment checks
# ---------------------------------------------------------------------------


def command_doctor(args: argparse.Namespace) -> None:
    paths = ensure_project_dirs()
    print("ClinAttr environment report")
    print("=" * 72)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Data root: {DATA_ROOT} | exists={DATA_ROOT.exists()}")
    print(f"Project root: {PROJECT_ROOT}")
    print()

    missing = []
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        found = importlib.util.find_spec(import_name) is not None
        print(f"{pip_name:24s} {'OK' if found else 'MISSING'}")
        if not found:
            missing.append(pip_name)

    print("\nLocal LLM paths")
    for name, path in LLM_PATHS.items():
        print(f"{name:12s} {path} | exists={Path(path).exists()}")

    try:
        import torch

        print("\nCUDA")
        print(f"Available: {torch.cuda.is_available()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            print(
                f"cuda:{idx}: {props.name}; "
                f"{props.total_memory / 1024**3:.1f} GiB"
            )
    except Exception as exc:
        print(f"CUDA check failed: {exc}")

    if missing:
        print("\nInstall missing packages with:")
        print(
            "python -m pip install -U "
            + " ".join(sorted(set(missing)))
        )
        print(
            "\nJina-ColBERT-v2 may additionally require a compatible "
            "flash-attn installation. If flash-attn fails, the code will "
            "still run BM25, BGE-M3, and attribute-rule experiments."
        )
    else:
        print("\nAll core imports are available.")

    print("\nRecommended initial model use")
    print("Qwen: controlled counterfactual generation")
    print("BioMistral: medical validation and primary LLM reranker")
    print("Llama: second-family sensitivity analysis only")


# ---------------------------------------------------------------------------
# CURE dataset preparation
# ---------------------------------------------------------------------------


def _dataset_rows(dataset_obj: Any) -> List[Dict[str, Any]]:
    return [dict(row) for row in dataset_obj]


def command_prepare(args: argparse.Namespace) -> None:
    from datasets import load_dataset

    paths = ensure_project_dirs()
    domains = parse_csv_list(args.domains, DOMAINS)
    languages = parse_csv_list(args.languages, LANGUAGES)

    manifest: Dict[str, Any] = {
        "dataset": CURE_DATASET_ID,
        "domains": domains,
        "languages": languages,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "counts": {},
    }

    for domain in domains:
        logging.info("Preparing domain=%s", domain)

        corpus_path = paths["corpus"] / f"{domain}.jsonl"
        qrels_path = paths["qrels"] / f"{domain}.jsonl"

        if args.force or not corpus_path.exists():
            corpus_ds = load_dataset(
                CURE_DATASET_ID,
                name="corpus",
                split=domain,
                cache_dir=str(HF_CACHE),
            )
            corpus_rows = []
            for row in corpus_ds:
                corpus_rows.append(
                    {
                        "doc_id": str(row["_id"]),
                        "text": normalise_unicode(row.get("text", "")),
                        "title": normalise_unicode(row.get("title", "")),
                        "domain": domain,
                    }
                )
            write_jsonl(corpus_path, corpus_rows)
        else:
            corpus_rows = read_jsonl(corpus_path)

        if args.force or not qrels_path.exists():
            qrels_ds = load_dataset(
                CURE_DATASET_ID,
                name="qrels",
                split=domain,
                cache_dir=str(HF_CACHE),
            )
            qrels_rows = []
            for row in qrels_ds:
                qid = (
                    row.get("query-id")
                    or row.get("query_id")
                    or row.get("query-id")
                )
                did = (
                    row.get("corpus-id")
                    or row.get("corpus_id")
                    or row.get("corpus-id")
                )
                qrels_rows.append(
                    {
                        "query_id": str(qid),
                        "doc_id": str(did),
                        "score": int(row.get("score", 0)),
                        "judge": str(row.get("judge", "")),
                        "explanation": normalise_unicode(row.get("explanation", "")),
                        "domain": domain,
                    }
                )
            write_jsonl(qrels_path, qrels_rows)
        else:
            qrels_rows = read_jsonl(qrels_path)

        manifest["counts"].setdefault(domain, {})
        manifest["counts"][domain]["corpus"] = len(corpus_rows)
        manifest["counts"][domain]["qrels"] = len(qrels_rows)

        for language in languages:
            query_path = paths["queries"] / language / f"{domain}.jsonl"
            if args.force or not query_path.exists():
                query_ds = load_dataset(
                    CURE_DATASET_ID,
                    name=f"queries-{language}",
                    split=domain,
                    cache_dir=str(HF_CACHE),
                )
                query_rows = []
                for row in query_ds:
                    query_rows.append(
                        {
                            "query_id": str(row["_id"]),
                            "text": normalise_unicode(row.get("text", "")),
                            "expertise": str(row.get("expertise", "")),
                            "language": language,
                            "domain": domain,
                        }
                    )
                write_jsonl(query_path, query_rows)
            else:
                query_rows = read_jsonl(query_path)
            manifest["counts"][domain][f"queries_{language}"] = len(query_rows)

    write_json(paths["normalized"] / "manifest.json", manifest)
    logging.info("Prepared CURE at %s", paths["normalized"])


def load_domain_data(
    domain: str,
    language: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    paths = ensure_project_dirs()
    corpus_path = paths["corpus"] / f"{domain}.jsonl"
    query_path = paths["queries"] / language / f"{domain}.jsonl"
    qrels_path = paths["qrels"] / f"{domain}.jsonl"
    missing = [p for p in [corpus_path, query_path, qrels_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "CURE has not been prepared for this domain/language. Missing:\n"
            + "\n".join(str(p) for p in missing)
            + "\nRun the prepare command first."
        )
    return read_jsonl(corpus_path), read_jsonl(query_path), read_jsonl(qrels_path)


# ---------------------------------------------------------------------------
# Clinical attribute extraction
# ---------------------------------------------------------------------------


@dataclass
class AttributeRecord:
    negation: Set[str]
    dosage: Set[str]
    frequency: Set[str]
    stage: Set[str]
    subtype: Set[str]
    temporal: Set[str]
    severity: Set[str]

    def to_jsonable(self) -> Dict[str, List[str]]:
        return {
            name: sorted(getattr(self, name))
            for name in ATTRIBUTE_FAMILIES
        }

    @classmethod
    def empty(cls) -> "AttributeRecord":
        return cls(
            negation=set(),
            dosage=set(),
            frequency=set(),
            stage=set(),
            subtype=set(),
            temporal=set(),
            severity=set(),
        )


class ClinicalAttributeExtractor:
    """
    Deterministic multilingual extractor used for scalable retrieval.

    This is intentionally transparent and auditable. It is not claimed to be a
    complete clinical NLP system. LLM extraction is available separately for
    candidate validation and small top-k analyses.
    """

    NEGATION_EXCLUSIONS = {
        "en": [
            r"\btuberculous\b",
            r"\bimmunolog",
            r"\bcardiac\b",
            r"\bwalk\b",
            r"\baccess\b",
            r"\bhaving\b",
            r"\brepresentative\b",
        ],
        "fr": [
            r"\btuberculeux|tuberculeuse\b",
            r"\bimmunolog",
            r"\bcardiaque\b",
            r"\bmarcher\b",
            r"\bacc[eè]s\b",
            r"\bacc[eè]d",
            r"\breconnaissance\b",
        ],
        "es": [
            r"\btuberculos[oa]s?\b",
            r"\binmunol[oó]gic",
            r"\bcard[ií]ac[oa]\b",
            r"\bandar\b",
            r"\bacced|\bacceso\b",
            r"\bvisi[oó]n\b",
        ],
    }

    NEGATION_PATTERNS = {
        "en": [
            r"\bno\b",
            r"\bnot\b",
            r"\bwithout\b",
            r"\bnegative for\b",
            r"\babsence of\b",
            r"\bdenies?\b",
            r"\bfree of\b",
            r"\bcontraindicat(?:ed|ion)\b",
        ],
        "fr": [
            r"\bpas\b",
            r"\bsans\b",
            r"\baucun(?:e)?\b",
            r"\bnégati(?:f|ve)\b",
            r"\babsence de\b",
            r"\bnie\b",
            r"\bcontre-indiqu",
        ],
        "es": [
            r"\bno\b",
            r"\bsin\b",
            r"\bning[uú]n(?:a)?\b",
            r"\bnegativ[oa]\b",
            r"\bausencia de\b",
            r"\bniega\b",
            r"\bcontraindic",
        ],
    }

    POSITIVE_PATTERNS = {
        "en": [
            r"\bpositive for\b",
            r"\bpresence of\b",
            r"\bconfirmed\b",
            r"\bwith\b",
        ],
        "fr": [
            r"\bpositi(?:f|ve)\b",
            r"\bprésence de\b",
            r"\bconfirm",
            r"\bavec\b",
        ],
        "es": [
            r"\bpositiv[oa]\b",
            r"\bpresencia de\b",
            r"\bconfirm",
            r"\bcon\b",
        ],
    }

    FREQUENCY_MAP = {
        "once_daily": [
            r"\bonce daily\b",
            r"\bonce a day\b",
            r"\b1 time daily\b",
            r"\bqd\b",
            r"\bune fois par jour\b",
            r"\b1 fois par jour\b",
            r"\buna vez al d[ií]a\b",
        ],
        "twice_daily": [
            r"\btwice daily\b",
            r"\btwo times daily\b",
            r"\bbid\b",
            r"\bdeux fois par jour\b",
            r"\b2 fois par jour\b",
            r"\bdos veces al d[ií]a\b",
        ],
        "three_times_daily": [
            r"\bthree times daily\b",
            r"\btid\b",
            r"\btrois fois par jour\b",
            r"\b3 fois par jour\b",
            r"\btres veces al d[ií]a\b",
        ],
        "four_times_daily": [
            r"\bfour times daily\b",
            r"\bqid\b",
            r"\bquatre fois par jour\b",
            r"\bcuatro veces al d[ií]a\b",
        ],
        "weekly": [
            r"\bonce weekly\b",
            r"\bweekly\b",
            r"\bhebdomadaire\b",
            r"\bune fois par semaine\b",
            r"\bsemanal\b",
            r"\buna vez por semana\b",
        ],
        "as_needed": [
            r"\bas needed\b",
            r"\bprn\b",
            r"\bsi besoin\b",
            r"\bseg[uú]n sea necesario\b",
        ],
    }

    TEMPORAL_MAP = {
        "current": [
            r"\bcurrent\b",
            r"\bactive\b",
            r"\bacute\b",
            r"\bongoing\b",
            r"\bactuel(?:le)?\b",
            r"\bactif|active\b",
            r"\ben cours\b",
            r"\bactual\b",
            r"\bactivo|activa\b",
            r"\ben curso\b",
        ],
        "previous": [
            r"\bprevious\b",
            r"\bpast history\b",
            r"\bprior\b",
            r"\bformer\b",
            r"\bant[ée]rieur(?:e)?\b",
            r"\bancien(?:ne)?\b",
            r"\bpr[ée]c[ée]dent(?:e)?\b",
            r"\bprevio|previa\b",
            r"\bantecedente\b",
            r"\banterior\b",
        ],
        "chronic": [
            r"\bchronic\b",
            r"\bchronique\b",
            r"\bcr[oó]nic[oa]\b",
        ],
        "recurrent": [
            r"\brecurrent\b",
            r"\br[ée]current(?:e)?\b",
            r"\brecurrente\b",
        ],
    }

    SEVERITY_MAP = {
        "mild": [r"\bmild\b", r"\bl[ée]ger|l[ée]g[èe]re\b", r"\bleve\b"],
        "moderate": [r"\bmoderate\b", r"\bmod[ée]r[ée]e?\b", r"\bmoderad[oa]\b"],
        "severe": [r"\bsevere\b", r"\bs[ée]v[èe]re\b", r"\bgrave\b", r"\bsever[oa]\b"],
        "critical": [r"\bcritical\b", r"\bcritique\b", r"\bcr[ií]tic[oa]\b"],
    }

    UNIT_ALIASES = {
        "mg": "mg",
        "milligram": "mg",
        "milligrams": "mg",
        "milligramme": "mg",
        "milligrammes": "mg",
        "g": "g",
        "gram": "g",
        "grams": "g",
        "gramme": "g",
        "grammes": "g",
        "mcg": "mcg",
        "µg": "mcg",
        "μg": "mcg",
        "microgram": "mcg",
        "micrograms": "mcg",
        "ml": "mL",
        "milliliter": "mL",
        "milliliters": "mL",
        "millilitre": "mL",
        "millilitres": "mL",
        "l": "L",
        "iu": "IU",
        "ui": "IU",
        "unit": "unit",
        "units": "unit",
        "unidad": "unit",
        "unidades": "unit",
        "unité": "unit",
        "unités": "unit",
        "%": "%",
        "mmol": "mmol",
        "mmol/l": "mmol/L",
        "mg/dl": "mg/dL",
    }

    ROMAN = {
        "i": 1,
        "ii": 2,
        "iii": 3,
        "iv": 4,
        "v": 5,
    }

    def extract(self, text: str, language: str = "en") -> AttributeRecord:
        text_norm = normalise_unicode(text).lower()
        language = language if language in LANGUAGES else "en"
        record = AttributeRecord.empty()

        neg_found = False
        for pattern in self.NEGATION_PATTERNS[language]:
            for match in re.finditer(pattern, text_norm, flags=re.IGNORECASE):
                window_start = max(0, match.start() - 10)
                window_end = min(len(text_norm), match.end() + 40)
                window = text_norm[window_start:window_end]
                excluded = any(
                    re.search(excl, window, flags=re.IGNORECASE)
                    for excl in self.NEGATION_EXCLUSIONS.get(language, [])
                )
                if not excluded:
                    neg_found = True
                    break
            if neg_found:
                break
        if neg_found:
            record.negation.add("negative")

        # Dose, concentration, and quantity patterns.
        unit_pattern = (
            r"(mg/dl|mmol/l|mcg|µg|μg|mg|ml|iu|ui|g|l|%|"
            r"milligrams?|milligrammes?|grams?|grammes?|"
            r"micrograms?|millilit(?:er|re)s?|units?|unidades?|unités?)"
        )
        for match in re.finditer(
            rf"\b(\d+(?:[.,]\d+)?)\s*{unit_pattern}\b",
            text_norm,
            flags=re.IGNORECASE,
        ):
            value = match.group(1).replace(",", ".")
            unit_raw = match.group(2).lower()
            unit = self.UNIT_ALIASES.get(unit_raw, unit_raw)
            record.dosage.add(f"{value} {unit}")

        for canonical, patterns in self.FREQUENCY_MAP.items():
            if any(re.search(p, text_norm, flags=re.IGNORECASE) for p in patterns):
                record.frequency.add(canonical)

        # Staging: "stage IV", "stade 4", "estadio III", TNM-like grade.
        for match in re.finditer(
            r"\b(?:stage|stade|estadio|grade|grado)\s*([ivx]+|\d+)[a-c]?\b",
            text_norm,
            flags=re.IGNORECASE,
        ):
            raw = match.group(1).lower()
            if raw in self.ROMAN:
                stage_num = self.ROMAN[raw]
            else:
                try:
                    stage_num = int(raw)
                except ValueError:
                    continue
            record.stage.add(str(stage_num))

        # Disease/treatment subtype markers.
        subtype_patterns = [
            r"\btype\s*([1-9ivx]+)\b",
            r"\btype de\s*([1-9ivx]+)\b",
            r"\btipo\s*([1-9ivx]+)\b",
            r"\bclass\s*([1-9ivx]+)\b",
            r"\bclasse\s*([1-9ivx]+)\b",
            r"\bclase\s*([1-9ivx]+)\b",
        ]
        for pattern in subtype_patterns:
            for match in re.finditer(pattern, text_norm, flags=re.IGNORECASE):
                raw = match.group(1).lower()
                value = str(self.ROMAN.get(raw, raw))
                record.subtype.add(value)

        for canonical, patterns in self.TEMPORAL_MAP.items():
            if any(re.search(p, text_norm, flags=re.IGNORECASE) for p in patterns):
                record.temporal.add(canonical)

        for canonical, patterns in self.SEVERITY_MAP.items():
            if any(re.search(p, text_norm, flags=re.IGNORECASE) for p in patterns):
                record.severity.add(canonical)

        return record


def compatibility_value(query_values: Set[str], doc_values: Set[str]) -> float:
    """
    +1: at least one exact attribute match.
    -1: both sides explicitly express an attribute but do not overlap.
     0: query has no attribute, or document omits it.
    """
    if not query_values:
        return 0.0
    if not doc_values:
        return 0.0
    if query_values.intersection(doc_values):
        return 1.0
    return -1.0


def attribute_feature_vector(
    query: str,
    document: str,
    language: str,
    dense_score: float,
    late_score: float,
    extractor: ClinicalAttributeExtractor,
) -> List[float]:
    q_attr = extractor.extract(query, language=language)
    # Evidence passages are English in CURE.
    d_attr = extractor.extract(document, language="en")

    family_scores = [
        compatibility_value(
            getattr(q_attr, family),
            getattr(d_attr, family),
        )
        for family in ATTRIBUTE_FAMILIES
    ]
    expressed = sum(bool(getattr(q_attr, family)) for family in ATTRIBUTE_FAMILIES)
    matched_or_conflicted = sum(score != 0.0 for score in family_scores)
    coverage = matched_or_conflicted / max(expressed, 1)
    return [
        float(dense_score),
        float(late_score),
        *family_scores,
        float(coverage),
    ]


def command_audit(args: argparse.Namespace) -> None:
    import pandas as pd

    paths = ensure_project_dirs()
    domains = parse_csv_list(args.domains, DOMAINS)
    languages = parse_csv_list(args.languages, LANGUAGES)
    extractor = ClinicalAttributeExtractor()

    rows: List[Dict[str, Any]] = []
    examples: List[Dict[str, Any]] = []

    for domain in domains:
        for language in languages:
            _, queries, _ = load_domain_data(domain, language)
            for row in queries:
                attrs = extractor.extract(row["text"], language)
                result = {
                    "domain": domain,
                    "language": language,
                    "query_id": row["query_id"],
                    "expertise": row.get("expertise", ""),
                    "text": row["text"],
                }
                total = 0
                for family in ATTRIBUTE_FAMILIES:
                    values = sorted(getattr(attrs, family))
                    result[family] = "|".join(values)
                    result[f"has_{family}"] = int(bool(values))
                    total += int(bool(values))
                result["num_attribute_families"] = total
                rows.append(result)
                if total > 0 and len(examples) < args.max_examples:
                    examples.append(result)

    frame = pd.DataFrame(rows)
    out_csv = paths["results"] / "attribute_audit_queries.csv"
    frame.to_csv(out_csv, index=False)

    summary = (
        frame.groupby(["language", "domain"])[
            [f"has_{family}" for family in ATTRIBUTE_FAMILIES]
        ]
        .agg(["sum", "mean"])
        .reset_index()
    )
    summary_path = paths["results"] / "attribute_audit_summary.csv"
    summary.to_csv(summary_path, index=False)
    write_json(paths["results"] / "attribute_examples.json", examples)

    logging.info("Audit rows: %d", len(frame))
    logging.info("Saved: %s", out_csv)
    logging.info("Saved: %s", summary_path)


# ---------------------------------------------------------------------------
# Local LLM support
# ---------------------------------------------------------------------------


class LocalLLM:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        load_in_4bit: bool = True,
        max_memory_gib: Optional[int] = None,
    ) -> None:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        self.model_path = model_path
        self.device_name = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if device.startswith("cuda:"):
            gpu_id = int(device.split(":", 1)[1])
            kwargs["device_map"] = {"": gpu_id}
            kwargs["torch_dtype"] = torch.bfloat16
            if max_memory_gib is not None:
                kwargs["max_memory"] = {gpu_id: f"{max_memory_gib}GiB"}
            if load_in_4bit:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
        else:
            kwargs["device_map"] = {"": "cpu"}
            kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.model.eval()

    @property
    def input_device(self):
        try:
            return self.model.get_input_embeddings().weight.device
        except Exception:
            return next(self.model.parameters()).device

    def render(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n"

    def generate(
        self,
        system: str,
        user: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> str:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        prompt = self.render(system, user)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            add_special_tokens=False,
        )
        encoded = {k: v.to(self.input_device) for k, v in encoded.items()}
        input_len = encoded["input_ids"].shape[1]
        logging.debug("prompt_tokens=%d max_new_tokens=%d", input_len, max_new_tokens)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-5),
            "top_p": 0.9,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        with torch.inference_mode():
            output = self.model.generate(
                **encoded, **generation_kwargs, return_dict_in_generate=True
            )
        sequence = output.sequences[0]
        new_tokens = sequence[input_len:]
        hit_token_limit = len(new_tokens) >= max_new_tokens
        ended_with_eos = (
            len(new_tokens) > 0
            and self.tokenizer.eos_token_id is not None
            and int(new_tokens[-1]) == int(self.tokenizer.eos_token_id)
        )
        logging.info(
            "generation stopped: generated_tokens=%d max_new_tokens=%d "
            "hit_token_limit=%s ended_with_eos=%s",
            len(new_tokens), max_new_tokens, hit_token_limit, ended_with_eos,
        )
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _repair_truncated_json(raw_from_brace: str) -> Optional[str]:
    """
    Best-effort repair for a JSON object cut off mid-generation (token
    limit reached, or the model emitted its own end-of-sequence token
    before closing the object). Finds the last complete "key": value pair
    -- string, bool, null, or number -- in document order and closes the
    object there, dropping only the trailing incomplete field. Since our
    prompts put the field we actually need to act on (e.g. "decision")
    before free-text fields like "reason"/"explanation", this reliably
    rescues the decision even when the trailing text got cut off.
    Returns None if no complete key/value pair is found at all.
    """
    if not raw_from_brace.lstrip().startswith("{"):
        return None
    pair_pattern = re.compile(
        r'"[^"]+"\s*:\s*(?:"(?:[^"\\]|\\.)*"|true|false|null|-?\d+(?:\.\d+)?)'
    )
    matches = list(pair_pattern.finditer(raw_from_brace))
    if not matches:
        return None
    body = raw_from_brace[: matches[-1].end()]
    return body + "}"


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    candidates.append(text)
    if start >= 0:
        repaired = _repair_truncated_json(text[start:])
        if repaired is not None:
            candidates.append(repaired)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            obj = ast.literal_eval(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    raise ValueError(f"Could not parse JSON object from response:\n{text[:1000]}")


# ---------------------------------------------------------------------------
# Counterfactual pair generation and validation
# ---------------------------------------------------------------------------


GENERATOR_SYSTEM = """
You generate controlled multilingual clinical retrieval stress tests by
editing ONE short span of an evidence passage.

Given:
- one clinical query in English, French, or Spanish;
- one relevant English evidence passage;

identify a SHORT, VERBATIM span of the passage that expresses one of the
seven attributes below, and propose a replacement for that span that
changes the clinically decisive value. Allowed attributes:
negation, dosage, administration frequency, disease subtype, disease stage,
temporal status, or severity.

The "attribute_type" field in your JSON output MUST be exactly one of these
seven lowercase strings, with no other value permitted:
negation, dosage, frequency, subtype, stage, temporal, severity.
Do not invent new categories.

If the passage does NOT clearly and directly support one of these seven
attributes, do not force a weak or fabricated match. Instead return
exactly this JSON and nothing else:
{"skip": true, "reason": "brief reason, under 15 words"}

Requirements for "source_span":
1. Copy it EXACTLY, character for character, from the passage you were
   given. Do not paraphrase, summarise, or retype it from memory.
2. Keep it SHORT -- a few words to one short clause is enough. Do not
   copy a full sentence or paragraph.
3. It must occur, ideally, only once in the passage.

Requirements for "replacement_span":
4. It replaces "source_span" directly in the passage, so it must fit
   grammatically in the exact same place.
5. Change only the one clinically decisive value -- do not change
   anything else about the sentence.
6. Keep it in English, matching the passage's language.

Requirements for "counterfactual_query":
7. Preserve the query's original language.
8. Adjust it so it is compatible with the edited passage and NOT
   compatible with the original passage.
9. Do not state the new value outright as a qualifier (e.g. appending
   "...if X" or "...si Y") -- that gives the answer away directly rather
   than testing retrieval.
10. Do not invent unsafe treatment recommendations.

Return JSON only:
{
  "attribute_type": "negation|dosage|frequency|subtype|stage|temporal|severity",
  "source_span": "verbatim text copied exactly from the passage",
  "replacement_span": "the replacement text",
  "original_value": "short label for the original value",
  "counterfactual_value": "short label for the new value",
  "counterfactual_query": "...",
  "explanation": "brief explanation of the one controlled change, under 20 words"
}
""".strip()


VALIDATOR_SYSTEM = """
You are a conservative medical dataset validator.

You will be shown an original query, a counterfactual query, an original
passage, and an edited passage where exactly one short span was replaced
by construction (this has already been verified programmatically -- you
do not need to re-check it). Your job is medical and logical judgment,
not text comparison.

Assess:
1. Is the edited passage medically plausible? Does it invent an unsafe or
   unsupported treatment recommendation?
2. Is the original query compatible with the original passage, and NOT
   compatible with the edited passage? Is the counterfactual query
   compatible with the edited passage, and NOT compatible with the
   original passage?
3. Does the counterfactual query state the new value outright as a
   qualifier, giving the answer away directly rather than testing
   retrieval?
4. Does the counterfactual query concern the SAME clinical topic,
   condition, and anatomical site as the original query, differing ONLY
   in the tested attribute? If the counterfactual query drifts to a
   different topic, condition, or body part, set same_topic to false.

You MUST include all five fields below, in this exact order, before
stopping. Do not stop generating after only some of the fields.
{
  "medically_plausible": true|false,
  "pair_logic_valid": true|false,
  "query_leaks_answer": true|false,
  "unsafe_or_unsupported": true|false,
  "same_topic": true|false,
  "reason": "brief reason, under 10 words"
}
""".strip()


def build_qrels_map(qrels: Sequence[Mapping[str, Any]]) -> Dict[str, List[Tuple[str, int]]]:
    mapping: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for row in qrels:
        score = int(row.get("score", 0))
        if score > 0:
            mapping[str(row["query_id"])].append((str(row["doc_id"]), score))
    for qid in mapping:
        mapping[qid].sort(key=lambda x: -x[1])
    return mapping


def _coerce_bool(value):
    """
    Best-effort coercion of a validator JSON field to a Python bool.
    Handles real JSON booleans plus common stand-ins a model might emit
    instead ("true"/"false", "yes"/"no", 1/0). Returns None if the value
    can't be confidently interpreted -- callers should treat None as
    failing the check (fail closed on ambiguity, not open).
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "1"):
            return True
        if v in ("false", "no", "0"):
            return False
    return None


def _language_looks_preserved(query: str, expected_language: str) -> bool:
    """
    Cheap, dependency-free MARKER check -- not a verified determination --
    of whether a query is plausibly still in its expected language. This
    is diagnostic metadata only, never a hard gate: it can false-warn on
    valid French/Spanish queries lacking accented characters or listed
    question words, and always returns True for English, so it cannot
    detect English-to-French/Spanish drift. Kept because the earlier
    validator problem was BioMistral hallucinating false claims of a
    language violation that hadn't happened -- a rough independent marker
    is safer than trusting the model's own self-report for this.
    """
    q = query.lower()
    if expected_language == "fr":
        markers = ["qu'est-ce", "quel", "quelle", "o\u00f9", "\u00e9", "\u00e8", "\u00e0", "\u00e7", "\u00ea"]
    elif expected_language == "es":
        markers = ["qu\u00e9", "cu\u00e1l", "c\u00f3mo", "\u00bf", "\u00f1", "\u00e1", "\u00ed", "\u00f3", "\u00fa"]
    else:
        return True
    return any(marker in q for marker in markers)


def _spans_grounded(
    validation: Mapping[str, Any],
    positive_passage: str,
    negative_passage: str,
) -> Tuple[bool, str]:
    """
    Code-level check that the validator's quoted spans are real, distinct
    substrings of the actual passages -- not hallucinated. This closes the
    main hallucination path we observed: a model asserting "the spans
    differ" or "language preserved" without the claim being grounded in
    the actual text. Case- and whitespace-insensitive. Returns
    (grounded, reason_if_not_grounded).
    """
    pos_quote = str(validation.get("positive_span_quote", "")).strip()
    neg_quote = str(validation.get("negative_span_quote", "")).strip()
    if not pos_quote or not neg_quote:
        return False, "missing quoted span"

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip().lower()

    if _norm(pos_quote) not in _norm(positive_passage):
        return False, "positive_span_quote not found verbatim in positive passage"
    if _norm(neg_quote) not in _norm(negative_passage):
        return False, "negative_span_quote not found verbatim in negative passage"
    if _norm(pos_quote) == _norm(neg_quote):
        return False, "quoted spans are identical"
    return True, ""


def _locate_span(passage: str, span: str) -> Optional[Tuple[int, int]]:
    """
    Locates `span` inside `passage`, tolerating whitespace differences the
    model may introduce when re-typing a short quoted span (double
    spaces, a dropped newline). Returns the (start, end) offsets of the
    unique match in `passage`'s own real text, or None if the span cannot
    be located exactly once. Never returns an approximate/fuzzy boundary --
    if a clean, unique match can't be found, this fails closed rather
    than guessing, since guessing here would risk editing the wrong text.
    """
    span = span.strip()
    if not span:
        return None
    exact_matches = [m.span() for m in re.finditer(re.escape(span), passage)]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        return None  # ambiguous -- refuse to guess which occurrence
    tokens = span.split()
    if not tokens:
        return None
    pattern = r"\s+".join(re.escape(t) for t in tokens)
    flexible_matches = [
        m.span() for m in re.finditer(pattern, passage, flags=re.IGNORECASE)
    ]
    if len(flexible_matches) == 1:
        return flexible_matches[0]
    return None


def _apply_span_edit(
    original_passage: str, source_span: str, replacement_span: str
) -> Optional[str]:
    """
    Locates source_span in original_passage and returns the edited
    passage with it replaced by replacement_span, editing the passage's
    own real text at the located position (not the model's possibly
    slightly different re-typing of the span). Returns None if the span
    cannot be uniquely located.
    """
    location = _locate_span(original_passage, source_span)
    if location is None:
        return None
    start, end = location
    return original_passage[:start] + replacement_span + original_passage[end:]


def command_generate_pairs(args: argparse.Namespace) -> None:
    import pandas as pd

    paths = ensure_project_dirs()
    domains = parse_csv_list(args.domains, DOMAINS)
    languages = parse_csv_list(args.languages, LANGUAGES)
    rng = random.Random(args.seed)

    source_examples: List[Dict[str, Any]] = []
    for domain in domains:
        corpus, _, qrels = load_domain_data(domain, "en")
        corpus_by_id = {row["doc_id"]: row for row in corpus}
        qrels_map = build_qrels_map(qrels)
        for language in languages:
            _, queries, _ = load_domain_data(domain, language)
            for query in queries:
                relevant = qrels_map.get(query["query_id"], [])
                if not relevant:
                    continue
                top_doc_id, relevance = relevant[0]
                doc = corpus_by_id.get(top_doc_id)
                if doc is None:
                    continue
                source_examples.append(
                    {
                        "domain": domain,
                        "language": language,
                        "query_id": query["query_id"],
                        "doc_id": top_doc_id,
                        "relevance": relevance,
                        "query": query["text"],
                        "passage": f"{doc.get('title', '')}\n{doc.get('text', '')}".strip(),
                    }
                )

    rng.shuffle(source_examples)
    source_examples = source_examples[: args.num_pairs]
    logging.info("Source examples selected: %d", len(source_examples))

    generator_name = args.generator_model
    validator_name = args.validator_model
    generator = LocalLLM(
        LLM_PATHS[generator_name],
        device=args.generator_device,
        load_in_4bit=not args.no_4bit,
    )

    interim_path = paths["counterfactuals"] / f"generation_interim{args.out_suffix}.jsonl"
    if not args.resume and interim_path.exists():
        interim_path.unlink()
    completed_ids = set()
    generated_rows: List[Dict[str, Any]] = []
    if args.resume and interim_path.exists():
        generated_rows = read_jsonl(interim_path)
        completed_ids = {row["pair_id"] for row in generated_rows}

    for index, source in enumerate(source_examples):
        pair_id = stable_hash(
            f"{source['language']}|{source['query_id']}|{source['doc_id']}"
        )[:16]
        if pair_id in completed_ids:
            continue
        user = (
            f"Query language: {source['language']}\n"
            f"Clinical query:\n{source['query']}\n\n"
            f"Relevant English evidence passage:\n{source['passage']}"
        )
        try:
            raw = generator.generate(
                GENERATOR_SYSTEM,
                user,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=args.seed + index,
            )
            proposal = extract_json_object(raw)
            if proposal.get("skip"):
                row = {
                    "pair_id": pair_id,
                    **source,
                    "proposal": {},
                    "generator_raw": raw,
                    "generator_model": generator_name,
                    "generation_error": "",
                    "generation_skipped": True,
                    "skip_reason": str(proposal.get("reason", "")),
                }
                append_jsonl(interim_path, row)
                generated_rows.append(row)
                if (index + 1) % args.log_every == 0:
                    logging.info("Generated %d/%d", index + 1, len(source_examples))
                continue

            proposed_attr = str(proposal.get("attribute_type", "")).strip().lower()
            if proposed_attr not in ATTRIBUTE_FAMILIES:
                raise ValueError(
                    f"attribute_type {proposed_attr!r} is not one of "
                    f"{ATTRIBUTE_FAMILIES}; discarding candidate."
                )

            source_span = str(proposal.get("source_span", ""))
            replacement_span = str(proposal.get("replacement_span", ""))
            if not source_span.strip() or not replacement_span.strip():
                raise ValueError(
                    "source_span/replacement_span missing or empty; discarding candidate."
                )
            if source_span.strip() == replacement_span.strip():
                raise ValueError(
                    "source_span and replacement_span are identical; no-op edit."
                )

            counterfactual_passage = _apply_span_edit(
                source["passage"], source_span, replacement_span
            )
            if counterfactual_passage is None:
                raise ValueError(
                    "source_span could not be uniquely located in the original "
                    "passage; discarding candidate."
                )

            # The model is never asked to reproduce the original query or
            # passage, or to construct the full counterfactual passage --
            # the original is injected directly from the real CURE source,
            # and the counterfactual passage is built deterministically in
            # Python from the located span. This guarantees exactly one
            # contiguous, grounded edit with zero risk of paraphrasing,
            # truncation, or unrelated changes.
            proposal["original_query"] = source["query"]
            proposal["original_passage"] = source["passage"]
            proposal["counterfactual_passage"] = counterfactual_passage

            row = {
                "pair_id": pair_id,
                **source,
                "proposal": proposal,
                "generator_raw": raw,
                "generator_model": generator_name,
                "generation_error": "",
            }
        except Exception as exc:
            logging.exception("Generation failed pair_id=%s", pair_id)
            row = {
                "pair_id": pair_id,
                **source,
                "proposal": {},
                "generator_raw": "",
                "generator_model": generator_name,
                "generation_error": repr(exc),
            }
        append_jsonl(interim_path, row)
        generated_rows.append(row)
        if (index + 1) % args.log_every == 0:
            logging.info("Generated %d/%d", index + 1, len(source_examples))

    del generator
    cleanup_cuda()

    validator = LocalLLM(
        LLM_PATHS[validator_name],
        device=args.validator_device,
        load_in_4bit=not args.no_4bit,
    )

    output_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(generated_rows):
        proposal = row.get("proposal") or {}
        orig_passage = str(proposal.get("original_passage", ""))
        cf_passage = str(proposal.get("counterfactual_passage", ""))
        orig_query = str(proposal.get("original_query", ""))
        cf_query = str(proposal.get("counterfactual_query", ""))

        if not proposal:
            reason = (
                f"Generator skipped: {row.get('skip_reason', '')}"
                if row.get("generation_skipped")
                else "Generation failed or returned no proposal."
            )
            validation = {
                "decision": "reject_candidate",
                "reason": reason,
            }
            validator_raw = ""
        elif orig_passage.strip() == cf_passage.strip():
            validation = {
                "decision": "reject_candidate",
                "reason": "original and counterfactual passages are identical.",
            }
            validator_raw = ""
        elif orig_query.strip() == cf_query.strip():
            validation = {
                "decision": "reject_candidate",
                "reason": "original and counterfactual queries are identical.",
            }
            validator_raw = ""
        else:
            try:
                validator_raw = validator.generate(
                    VALIDATOR_SYSTEM,
                    json.dumps(
                        {
                            "query_language": row["language"],
                            "proposal": proposal,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    max_new_tokens=400,
                    temperature=0.0,
                    seed=args.seed + 100000 + index,
                )
                logging.info(
                    "validator_raw pair_id=%s len=%d text=%r",
                    row["pair_id"], len(validator_raw), validator_raw,
                )
                validation = extract_json_object(validator_raw)
                language_marker_check = _language_looks_preserved(cf_query, row["language"])
                validation["language_marker_check"] = language_marker_check
                if not language_marker_check:
                    logging.warning(
                        "Language marker check failed pair_id=%s language=%s query=%r "
                        "-- this is now a hard gate, candidate will be rejected",
                        row["pair_id"], row["language"], cf_query,
                    )

                required_fields = [
                    "medically_plausible",
                    "pair_logic_valid",
                    "query_leaks_answer",
                    "unsafe_or_unsupported",
                    "same_topic",
                ]
                missing_fields = [f for f in required_fields if f not in validation]

                if missing_fields:
                    logging.info(
                        "Validator response incomplete pair_id=%s missing=%s -- retrying with sampling",
                        row["pair_id"], missing_fields,
                    )
                    try:
                        retry_raw = validator.generate(
                            VALIDATOR_SYSTEM,
                            json.dumps(
                                {"query_language": row["language"], "proposal": proposal},
                                ensure_ascii=False,
                                indent=2,
                            ),
                            max_new_tokens=400,
                            temperature=0.4,
                            seed=args.seed + 300000 + index,
                        )
                        retry_validation = extract_json_object(retry_raw)
                        retry_missing = [f for f in required_fields if f not in retry_validation]
                        if not retry_missing:
                            validation = retry_validation
                            validation["language_marker_check"] = language_marker_check
                            missing_fields = []
                    except Exception:
                        pass

                if missing_fields:
                    validation["decision"] = "reject_candidate"
                    validation["validation_status"] = "incomplete_output"
                    validation["reason"] = "missing validator fields: " + ", ".join(missing_fields)
                else:
                    checks = {
                        "medically_plausible": _coerce_bool(validation.get("medically_plausible")) is True,
                        "pair_logic_valid": _coerce_bool(validation.get("pair_logic_valid")) is True,
                        "query_leaks_answer_ok": _coerce_bool(validation.get("query_leaks_answer")) is False,
                        "unsafe_or_unsupported_ok": _coerce_bool(validation.get("unsafe_or_unsupported")) is False,
                        "same_topic": _coerce_bool(validation.get("same_topic")) is True,
                        "language_ok": language_marker_check,
                    }
                    accepted = all(checks.values())
                    validation["validation_status"] = "complete"
                    validation["decision"] = "accept_candidate" if accepted else "reject_candidate"
                    if not accepted:
                        failed = [k for k, v in checks.items() if not v]
                        validation["reason"] = "deterministic gate failed: " + ", ".join(failed)
            except Exception as exc:
                logging.exception("Validation failed pair_id=%s", row["pair_id"])
                validation = {
                    "decision": "reject_candidate",
                    "reason": repr(exc),
                }
                validator_raw = ""

        output_rows.append(
            {
                "pair_id": row["pair_id"],
                "domain": row["domain"],
                "language": row["language"],
                "query_id": row["query_id"],
                "doc_id": row["doc_id"],
                "attribute_type": proposal.get("attribute_type", ""),
                "original_value": proposal.get("original_value", ""),
                "counterfactual_value": proposal.get("counterfactual_value", ""),
                "source_span": proposal.get("source_span", ""),
                "replacement_span": proposal.get("replacement_span", ""),
                "original_query": proposal.get("original_query", row["query"]),
                "counterfactual_query": proposal.get("counterfactual_query", ""),
                "positive_passage": proposal.get("original_passage", row["passage"]),
                "negative_passage": proposal.get("counterfactual_passage", ""),
                "proposal_explanation": proposal.get("explanation", ""),
                "validator_decision": validation.get("decision", ""),
                "medically_plausible": validation.get("medically_plausible", ""),
                "pair_logic_valid": validation.get("pair_logic_valid", ""),
                "query_leaks_answer": validation.get("query_leaks_answer", ""),
                "unsafe_or_unsupported": validation.get("unsafe_or_unsupported", ""),
                "validator_reason": validation.get("reason", ""),
                "validator_validation_status": validation.get("validation_status", ""),
                "validator_language_marker_check": validation.get("language_marker_check", ""),
                "generator_model": generator_name,
                "validator_model": validator_name,
                # Mandatory manual decision. Never auto-approve.
                "approved": 0,
                "reviewer": "",
                "review_notes": "",
                "split": "",
            }
        )
        if (index + 1) % args.log_every == 0:
            logging.info("Validated %d/%d", index + 1, len(generated_rows))

    del validator
    cleanup_cuda()

    output_path = paths["counterfactuals"] / f"counterfactual_candidates{args.out_suffix}.csv"
    pd.DataFrame(output_rows).to_csv(output_path, index=False)
    logging.info("Saved %d candidates to %s", len(output_rows), output_path)
    logging.warning(
        "All rows have approved=0. Manual clinical review is mandatory."
    )


# ---------------------------------------------------------------------------
# Dense retrieval
# ---------------------------------------------------------------------------


class DenseEncoder:
    def __init__(
        self,
        model_name_or_path: str = DEFAULT_DENSE_MODEL,
        device: str = "cuda:0",
        batch_size: int = 32,
        max_seq_length: int = 512,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name_or_path = model_name_or_path
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(
            model_name_or_path,
            device=device,
            cache_folder=str(HF_CACHE / "models"),
            trust_remote_code=True,
        )
        self.model.max_seq_length = max_seq_length

    def encode(self, texts: Sequence[str], show_progress: bool = True):
        import numpy as np

        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype="float32")


def corpus_text(row: Mapping[str, Any]) -> str:
    title = normalise_unicode(row.get("title", ""))
    text = normalise_unicode(row.get("text", ""))
    return f"{title}\n{text}".strip() if title else text


def dense_index_dir(domain: str, model_name: str) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    return ensure_project_dirs()["dense_indices"] / safe_model / domain


def command_index_dense(args: argparse.Namespace) -> None:
    import numpy as np

    domains = parse_csv_list(args.domains, DOMAINS)
    encoder = DenseEncoder(
        model_name_or_path=args.dense_model,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )

    for domain in domains:
        corpus, _, _ = load_domain_data(domain, "en")
        index_dir = dense_index_dir(domain, args.dense_model)
        index_dir.mkdir(parents=True, exist_ok=True)
        emb_path = index_dir / "embeddings.npy"
        meta_path = index_dir / "metadata.jsonl"
        manifest_path = index_dir / "manifest.json"

        if emb_path.exists() and meta_path.exists() and not args.force:
            logging.info("Dense index exists; skipping %s", domain)
            continue

        texts = [corpus_text(row) for row in corpus]
        embeddings = encoder.encode(texts)
        # Float16 storage; converted to float32 at search time.
        np.save(emb_path, embeddings.astype("float16"))
        write_jsonl(
            meta_path,
            (
                {
                    "doc_id": row["doc_id"],
                    "title": row.get("title", ""),
                    "text": row.get("text", ""),
                    "domain": domain,
                }
                for row in corpus
            ),
        )
        write_json(
            manifest_path,
            {
                "domain": domain,
                "model": args.dense_model,
                "num_documents": len(corpus),
                "embedding_dim": int(embeddings.shape[1]),
                "normalised": True,
            },
        )
        logging.info(
            "Saved dense index domain=%s docs=%d dim=%d",
            domain,
            len(corpus),
            embeddings.shape[1],
        )


def dense_search(
    encoder: DenseEncoder,
    query: str,
    domain: str,
    model_name: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    import numpy as np

    index_dir = dense_index_dir(domain, model_name)
    emb_path = index_dir / "embeddings.npy"
    meta_path = index_dir / "metadata.jsonl"
    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Dense index missing for {domain}. Run index-dense first."
        )
    corpus_embeddings = np.load(emb_path, mmap_mode="r")
    query_embedding = encoder.encode([query], show_progress=False)[0]
    scores = np.asarray(corpus_embeddings, dtype="float32") @ query_embedding
    top_k = min(top_k, scores.shape[0])
    idx = np.argpartition(-scores, top_k - 1)[:top_k]
    idx = idx[np.argsort(-scores[idx])]
    metadata = read_jsonl(meta_path)
    results = []
    for rank, corpus_idx in enumerate(idx, start=1):
        row = metadata[int(corpus_idx)]
        results.append(
            {
                "doc_id": row["doc_id"],
                "text": corpus_text(row),
                "title": row.get("title", ""),
                "dense_score": float(scores[int(corpus_idx)]),
                "rank": rank,
            }
        )
    return results


# ---------------------------------------------------------------------------
# BM25 and late-interaction reranking
# ---------------------------------------------------------------------------


class BM25Retriever:
    def __init__(self, corpus: Sequence[Mapping[str, Any]]) -> None:
        from rank_bm25 import BM25Okapi

        self.corpus = list(corpus)
        tokenized = [simple_tokens(corpus_text(row)) for row in self.corpus]
        self.model = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        import numpy as np

        scores = self.model.get_scores(simple_tokens(query))
        top_k = min(top_k, len(scores))
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        results = []
        for rank, corpus_idx in enumerate(idx, start=1):
            row = self.corpus[int(corpus_idx)]
            results.append(
                {
                    "doc_id": row["doc_id"],
                    "text": corpus_text(row),
                    "title": row.get("title", ""),
                    "bm25_score": float(scores[int(corpus_idx)]),
                    "rank": rank,
                }
            )
        return results


class LateInteractionReranker:
    def __init__(
        self,
        model_name_or_path: str = DEFAULT_LATE_MODEL,
        device: str = "cuda:0",
    ) -> None:
        # RAGatouille/ColBERT respects the active CUDA device.
        if device.startswith("cuda:"):
            import torch

            torch.cuda.set_device(int(device.split(":", 1)[1]))
        from ragatouille import RAGPretrainedModel

        self.model_name_or_path = model_name_or_path
        self.rag = RAGPretrainedModel.from_pretrained(model_name_or_path)

    def rerank(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        top_k: Optional[int] = None,
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        documents = [str(row["text"]) for row in candidates]
        top_k = top_k or len(documents)
        results = self.rag.rerank(
            query=query,
            documents=documents,
            k=min(top_k, len(documents)),
            zero_index_ranks=False,
            bsize=batch_size,
        )

        # RAGatouille returns content but not the original candidate position.
        positions: Dict[str, List[int]] = defaultdict(list)
        for idx, text in enumerate(documents):
            positions[text].append(idx)

        output = []
        used: Set[int] = set()
        for rank, result in enumerate(results, start=1):
            content = str(result.get("content", ""))
            candidate_idx = None
            for idx in positions.get(content, []):
                if idx not in used:
                    candidate_idx = idx
                    break
            if candidate_idx is None:
                # Conservative fallback for whitespace normalisation differences.
                normalised = normalise_unicode(content)
                for idx, text in enumerate(documents):
                    if idx not in used and normalise_unicode(text) == normalised:
                        candidate_idx = idx
                        break
            if candidate_idx is None:
                logging.warning("Could not map a ColBERT result back to candidate.")
                continue
            used.add(candidate_idx)
            row = dict(candidates[candidate_idx])
            row["late_score"] = float(result.get("score", 0.0))
            row["rank"] = rank
            output.append(row)
        return output


# ---------------------------------------------------------------------------
# Lightweight attribute gate
# ---------------------------------------------------------------------------


class AttributeGate:
    """
    Small non-negative linear gate trained with pairwise margin loss.

    Non-negative weights are implemented through softplus(raw_weights), making
    the learned contribution interpretable. Since incompatibility features are
    negative, a positive family weight penalises mismatches and rewards matches.

    feature_mode controls an ablation: "full" (default) uses all ten
    features; "fusion_only" zeroes the seven attribute-family features and
    the coverage feature before scoring, so the model can only learn from
    dense and late scores. This is architecturally identical to the full
    gate -- same parameter count, same training code -- but mathematically
    cannot use attribute information, regardless of what weight it assigns
    those (masked) inputs. Used to test whether the full gate's CAPR
    improvement is attributable to the attribute mechanism specifically,
    rather than to learned dense-late fusion alone.
    """

    def __init__(self, num_features: int = len(FEATURE_NAMES), feature_mode: str = "full") -> None:
        import torch

        self.torch = torch
        self.raw_weights = torch.nn.Parameter(torch.zeros(num_features))
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.feature_mode = feature_mode

    def parameters(self):
        return [self.raw_weights, self.bias]

    def _feature_mask(self, device=None):
        # Feature order: 0=dense_score, 1=late_score, 2-8=seven attribute
        # families, 9=attribute_coverage (see FEATURE_NAMES).
        mask = self.torch.ones(len(FEATURE_NAMES))
        if self.feature_mode == "fusion_only":
            mask[2:] = 0.0
        if device is not None:
            mask = mask.to(device)
        return mask

    def score(self, features):
        weights = self.torch.nn.functional.softplus(self.raw_weights)
        mask = self._feature_mask(device=features.device)
        return (features * mask) @ weights + self.bias

    def state_dict(self) -> Dict[str, Any]:
        return {
            "raw_weights": self.raw_weights.detach().cpu(),
            "bias": self.bias.detach().cpu(),
            "feature_names": FEATURE_NAMES,
            "feature_mode": self.feature_mode,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        with self.torch.no_grad():
            self.raw_weights.copy_(state["raw_weights"])
            self.bias.copy_(state["bias"])
        self.feature_mode = state.get("feature_mode", "full")

    def learned_weights(self) -> List[float]:
        return (
            self.torch.nn.functional.softplus(self.raw_weights)
            .detach()
            .cpu()
            .tolist()
        )


def score_pair_features(
    dense_encoder: DenseEncoder,
    late_reranker: Optional[LateInteractionReranker],
    query: str,
    positive_passage: str,
    negative_passage: str,
    language: str,
    extractor: ClinicalAttributeExtractor,
) -> Tuple[List[float], List[float], Dict[str, float]]:
    import numpy as np

    embeddings = dense_encoder.encode(
        [query, positive_passage, negative_passage],
        show_progress=False,
    )
    dense_pos = float(np.dot(embeddings[0], embeddings[1]))
    dense_neg = float(np.dot(embeddings[0], embeddings[2]))

    late_pos = 0.0
    late_neg = 0.0
    if late_reranker is not None:
        reranked = late_reranker.rerank(
            query,
            [
                {"doc_id": "positive", "text": positive_passage, "dense_score": dense_pos},
                {"doc_id": "negative", "text": negative_passage, "dense_score": dense_neg},
            ],
            top_k=2,
            batch_size=2,
        )
        late_scores = {row["doc_id"]: row.get("late_score", 0.0) for row in reranked}
        late_pos = float(late_scores.get("positive", 0.0))
        late_neg = float(late_scores.get("negative", 0.0))

    # Pairwise normalisation is deliberately symmetric.
    dense_pair = zscore([dense_pos, dense_neg])
    late_pair = zscore([late_pos, late_neg])

    pos_features = attribute_feature_vector(
        query,
        positive_passage,
        language,
        dense_pair[0],
        late_pair[0],
        extractor,
    )
    neg_features = attribute_feature_vector(
        query,
        negative_passage,
        language,
        dense_pair[1],
        late_pair[1],
        extractor,
    )
    diagnostics = {
        "dense_pos": dense_pos,
        "dense_neg": dense_neg,
        "late_pos": late_pos,
        "late_neg": late_neg,
    }
    return pos_features, neg_features, diagnostics


def compute_norm_stats(feature_rows):
    """Mean and standard deviation of raw dense and late scores over training pairs."""
    import numpy as np

    dense_values = []
    late_values = []
    for row in feature_rows:
        diag = row["diagnostics"]
        dense_values.extend([diag["dense_pos"], diag["dense_neg"]])
        late_values.extend([diag["late_pos"], diag["late_neg"]])
    return {
        "dense_mean": float(np.mean(dense_values)),
        "dense_std": float(max(np.std(dense_values), 1e-6)),
        "late_mean": float(np.mean(late_values)),
        "late_std": float(max(np.std(late_values), 1e-6)),
    }


def apply_global_norm(pos_features, neg_features, diagnostics, stats):
    """Replace the pairwise-normalized dense/late entries (indices 0 and 1) with
    globally standardized raw scores. Attribute entries are left untouched."""
    pos = list(pos_features)
    neg = list(neg_features)
    pos[0] = (diagnostics["dense_pos"] - stats["dense_mean"]) / stats["dense_std"]
    neg[0] = (diagnostics["dense_neg"] - stats["dense_mean"]) / stats["dense_std"]
    pos[1] = (diagnostics["late_pos"] - stats["late_mean"]) / stats["late_std"]
    neg[1] = (diagnostics["late_neg"] - stats["late_mean"]) / stats["late_std"]
    return pos, neg


def load_approved_pairs(path: Path, split: Optional[str] = None):
    import pandas as pd

    frame = pd.read_csv(path)
    if "approved" not in frame.columns:
        raise ValueError("Pairs CSV must contain an approved column.")
    approved = frame[
        frame["approved"].astype(str).str.lower().isin(["1", "true", "yes"])
    ].copy()
    if split is not None:
        if "split" not in approved.columns:
            raise ValueError("Requested a split but the CSV has no split column.")
        approved = approved[approved["split"].astype(str) == split].copy()
    if approved.empty:
        raise ValueError(
            "No approved pairs found. Human reviewers must set approved=1."
        )
    required = {
        "pair_id",
        "language",
        "attribute_type",
        "original_query",
        "counterfactual_query",
        "positive_passage",
        "negative_passage",
    }
    missing = required.difference(approved.columns)
    if missing:
        raise ValueError(f"Pairs CSV missing columns: {sorted(missing)}")
    return approved


def command_train_gate(args: argparse.Namespace) -> None:
    import numpy as np
    import torch

    paths = ensure_project_dirs()
    frame = load_approved_pairs(args.pairs_csv, split=args.train_split)
    if args.max_pairs:
        frame = frame.head(args.max_pairs)

    set_seed(args.seed)
    extractor = ClinicalAttributeExtractor()
    dense_encoder = DenseEncoder(
        model_name_or_path=args.dense_model,
        device=args.dense_device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )
    late_reranker = None
    if not args.no_late:
        late_reranker = LateInteractionReranker(
            model_name_or_path=args.late_model,
            device=args.late_device,
        )

    cache_path = paths["cache"] / "gate_training_features.jsonl"
    cache_by_id = {}
    if cache_path.exists() and args.resume:
        cache_by_id = {row["pair_id"]: row for row in read_jsonl(cache_path)}
    elif cache_path.exists():
        cache_path.unlink()

    feature_rows = []
    for i, row in frame.iterrows():
        pair_id = str(row["pair_id"])
        if pair_id in cache_by_id:
            record = cache_by_id[pair_id]
        else:
            pos_features, neg_features, diagnostics = score_pair_features(
                dense_encoder,
                late_reranker,
                str(row["original_query"]),
                str(row["positive_passage"]),
                str(row["negative_passage"]),
                str(row["language"]),
                extractor,
            )
            record = {
                "pair_id": pair_id,
                "attribute_type": str(row["attribute_type"]),
                "language": str(row["language"]),
                "positive_features": pos_features,
                "negative_features": neg_features,
                "diagnostics": diagnostics,
            }
            append_jsonl(cache_path, record)
        feature_rows.append(record)
        if (len(feature_rows)) % args.log_every == 0:
            logging.info("Feature pairs: %d/%d", len(feature_rows), len(frame))

    norm_stats = None
    if getattr(args, "norm", "pairwise") == "global":
        norm_stats = compute_norm_stats(feature_rows)
        logging.info("Global normalisation statistics: %s", norm_stats)
        rebuilt_rows = []
        for record in feature_rows:
            new_pos, new_neg = apply_global_norm(
                record["positive_features"],
                record["negative_features"],
                record["diagnostics"],
                norm_stats,
            )
            rebuilt_rows.append(
                {**record, "positive_features": new_pos, "negative_features": new_neg}
            )
        feature_rows = rebuilt_rows

    positive = torch.tensor(
        [row["positive_features"] for row in feature_rows],
        dtype=torch.float32,
        device=args.gate_device,
    )
    negative = torch.tensor(
        [row["negative_features"] for row in feature_rows],
        dtype=torch.float32,
        device=args.gate_device,
    )

    # Train/development split by pair_id to avoid leakage.
    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(feature_rows))
    rng.shuffle(indices)
    dev_size = max(1, int(len(indices) * args.dev_fraction))
    dev_idx = indices[:dev_size]
    train_idx = indices[dev_size:]
    if len(train_idx) == 0:
        train_idx = dev_idx

    gate = AttributeGate(num_features=len(FEATURE_NAMES), feature_mode=args.feature_mode)
    logging.info("DEBUG gate.feature_mode after construction: %s", gate.feature_mode)
    gate.raw_weights = torch.nn.Parameter(gate.raw_weights.to(args.gate_device))
    gate.bias = torch.nn.Parameter(gate.bias.to(args.gate_device))
    optimizer = torch.optim.AdamW(
        gate.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_dev = -float("inf")
    best_state = None
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        gate.raw_weights.train() if hasattr(gate.raw_weights, "train") else None
        optimizer.zero_grad()
        pos_scores = gate.score(positive[train_idx])
        neg_scores = gate.score(negative[train_idx])
        margin_loss = torch.nn.functional.softplus(
            args.margin - (pos_scores - neg_scores)
        ).mean()
        l1 = args.l1_weight * torch.nn.functional.softplus(
            gate.raw_weights
        ).mean()
        loss = margin_loss + l1
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            dev_pos = gate.score(positive[dev_idx])
            dev_neg = gate.score(negative[dev_idx])
            dev_acc = float((dev_pos > dev_neg).float().mean().item())
            dev_margin = float((dev_pos - dev_neg).mean().item())

        if epoch == 1 or epoch % args.log_every == 0:
            logging.info(
                "epoch=%d loss=%.5f dev_acc=%.4f dev_margin=%.4f",
                epoch,
                float(loss.item()),
                dev_acc,
                dev_margin,
            )

        selection_value = dev_acc + 1e-3 * dev_margin
        if selection_value > best_dev:
            best_dev = selection_value
            best_state = {
                "raw_weights": gate.raw_weights.detach().cpu().clone(),
                "bias": gate.bias.detach().cpu().clone(),
            }
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logging.info("Early stopping at epoch=%d", epoch)
                break

    if best_state is not None:
        gate.load_state_dict(best_state)

    checkpoint_path = args.output_checkpoint or (
        paths["models"] / "attribute_gate.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **gate.state_dict(),
            "dense_model": args.dense_model,
            "late_model": None if args.no_late else args.late_model,
            "train_pairs": len(train_idx),
            "dev_pairs": len(dev_idx),
            "seed": args.seed,
            "margin": args.margin,
            "norm_stats": norm_stats,
        },
        checkpoint_path,
    )
    weights = dict(zip(FEATURE_NAMES, gate.learned_weights()))
    write_json(checkpoint_path.with_suffix(".weights.json"), weights)
    logging.info("Saved gate checkpoint: %s", checkpoint_path)
    logging.info("Learned weights: %s", weights)


def load_gate(checkpoint_path: Path, device: str = "cpu") -> AttributeGate:
    import torch

    state = torch.load(checkpoint_path, map_location=device)
    gate = AttributeGate(num_features=len(FEATURE_NAMES))
    gate.raw_weights = torch.nn.Parameter(gate.raw_weights.to(device))
    gate.bias = torch.nn.Parameter(gate.bias.to(device))
    gate.load_state_dict(state)
    gate.norm_stats = state.get("norm_stats")
    return gate


def rule_attribute_score(features: Sequence[float], attr_weight: float = 0.35) -> float:
    dense_score = float(features[0])
    late_score = float(features[1])
    family_scores = features[2:9]
    coverage = float(features[9])
    return (
        0.5 * dense_score
        + 0.5 * late_score
        + attr_weight * sum(float(x) for x in family_scores)
        + 0.10 * coverage
    )


def fusion_only_score(features: Sequence[float]) -> float:
    """
    Ablation: dense-late score fusion with NO attribute term at all.
    """
    dense_score = float(features[0])
    late_score = float(features[1])
    return 0.5 * dense_score + 0.5 * late_score


def late_attr_score(features: Sequence[float], attr_weight: float = 0.35) -> float:
    """
    Ablation: late-interaction score plus the attribute term, with NO
    dense term at all. Used to check whether attr-rule's improvement
    depends on the dense-late fusion specifically, or holds even when
    attributes are added to late-interaction alone.
    """
    late_score = float(features[1])
    family_scores = features[2:9]
    return late_score + attr_weight * sum(float(x) for x in family_scores)


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------


def qrels_for_language(
    queries: Sequence[Mapping[str, Any]],
    qrels: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, int]]:
    query_ids = {str(row["query_id"]) for row in queries}
    output: Dict[str, Dict[str, int]] = defaultdict(dict)
    for row in qrels:
        qid = str(row["query_id"])
        if qid in query_ids and int(row.get("score", 0)) > 0:
            output[qid][str(row["doc_id"])] = int(row["score"])
    return output


def dcg_at_k(relevances: Sequence[int], k: int) -> float:
    return sum(
        (2 ** int(rel) - 1) / math.log2(rank + 1)
        for rank, rel in enumerate(relevances[:k], start=1)
    )


def evaluate_ranked_list(
    ranked_doc_ids: Sequence[str],
    relevant: Mapping[str, int],
    k_values: Sequence[int] = (5, 10),
) -> Dict[str, float]:
    rel_sequence = [int(relevant.get(doc_id, 0)) for doc_id in ranked_doc_ids]
    output: Dict[str, float] = {}

    for k in k_values:
        retrieved_relevant = sum(rel > 0 for rel in rel_sequence[:k])
        output[f"recall@{k}"] = retrieved_relevant / max(len(relevant), 1)

    k = 10
    dcg = dcg_at_k(rel_sequence, k)
    ideal = sorted(relevant.values(), reverse=True)
    idcg = dcg_at_k(ideal, k)
    output["ndcg@10"] = dcg / idcg if idcg > 0 else 0.0

    reciprocal_rank = 0.0
    precisions = []
    relevant_seen = 0
    for rank, rel in enumerate(rel_sequence[:10], start=1):
        if rel > 0:
            if reciprocal_rank == 0.0:
                reciprocal_rank = 1.0 / rank
            relevant_seen += 1
            precisions.append(relevant_seen / rank)
    output["mrr@10"] = reciprocal_rank
    output["map@10"] = (
        sum(precisions) / min(len(relevant), 10)
        if relevant
        else 0.0
    )
    return output


def aggregate_metrics(per_query: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not per_query:
        return {}
    metric_names = sorted(
        {
            key
            for row in per_query
            for key, value in row.items()
            if isinstance(value, (int, float))
        }
    )
    return {
        metric: statistics.fmean(float(row.get(metric, 0.0)) for row in per_query)
        for metric in metric_names
    }


def rank_with_attributes(
    query: str,
    language: str,
    candidates: Sequence[Mapping[str, Any]],
    extractor: ClinicalAttributeExtractor,
    gate: Optional[AttributeGate],
    attr_weight: float,
    score_mode: str = "full",
) -> List[Dict[str, Any]]:
    import torch

    if not candidates:
        return []
    dense_scores = [float(row.get("dense_score", 0.0)) for row in candidates]
    late_scores = [float(row.get("late_score", 0.0)) for row in candidates]
    dense_norm = zscore(dense_scores)
    late_norm = zscore(late_scores)

    feature_matrix = []
    output = []
    for idx, row in enumerate(candidates):
        features = attribute_feature_vector(
            query,
            str(row["text"]),
            language,
            dense_norm[idx],
            late_norm[idx],
            extractor,
        )
        feature_matrix.append(features)
        result = dict(row)
        result["attribute_features"] = dict(zip(FEATURE_NAMES, features))
        output.append(result)

    if gate is None:
        if score_mode == "fusion_only":
            scores = [fusion_only_score(features) for features in feature_matrix]
        elif score_mode == "late_attr":
            scores = [
                late_attr_score(features, attr_weight=attr_weight)
                for features in feature_matrix
            ]
        else:
            scores = [
                rule_attribute_score(features, attr_weight=attr_weight)
                for features in feature_matrix
            ]
    else:
        tensor = torch.tensor(feature_matrix, dtype=torch.float32)
        device = gate.raw_weights.device
        with torch.no_grad():
            scores = gate.score(tensor.to(device)).cpu().tolist()

    for row, score in zip(output, scores):
        row["attribute_score"] = float(score)
    output.sort(key=lambda x: -x["attribute_score"])
    for rank, row in enumerate(output, start=1):
        row["rank"] = rank
    return output


def save_run(
    output_path: Path,
    run_rows: Sequence[Mapping[str, Any]],
) -> None:
    write_jsonl(output_path, run_rows)


def command_evaluate_cure(args: argparse.Namespace) -> None:
    import pandas as pd

    paths = ensure_project_dirs()
    domains = parse_csv_list(args.domains, DOMAINS)
    languages = parse_csv_list(args.languages, LANGUAGES)
    methods = parse_csv_list(
        args.methods,
        ["bm25", "dense", "late", "attr-rule", "attr-gate", "fusion-only", "late-attr"],
    )

    needs_dense = any(
        method in methods for method in ["dense", "late", "attr-rule", "attr-gate"]
    )
    needs_late = any(
        method in methods for method in ["late", "attr-rule", "attr-gate"]
    )

    dense_encoder = None
    if needs_dense:
        dense_encoder = DenseEncoder(
            model_name_or_path=args.dense_model,
            device=args.dense_device,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
        )

    late_reranker = None
    if needs_late and not args.skip_late:
        late_reranker = LateInteractionReranker(
            model_name_or_path=args.late_model,
            device=args.late_device,
        )

    gate = None
    if "attr-gate" in methods:
        if args.gate_checkpoint is None:
            raise ValueError("--gate-checkpoint is required for attr-gate.")
        gate = load_gate(args.gate_checkpoint, device=args.gate_device)

    extractor = ClinicalAttributeExtractor()
    metric_rows: List[Dict[str, Any]] = []
    run_root = paths["results"] / "runs"
    run_root.mkdir(parents=True, exist_ok=True)

    for domain in domains:
        corpus, _, qrels = load_domain_data(domain, "en")
        bm25 = BM25Retriever(corpus) if "bm25" in methods else None

        for language in languages:
            _, queries, _ = load_domain_data(domain, language)
            if args.max_queries is not None:
                queries = queries[: args.max_queries]
            relevant_map = qrels_for_language(queries, qrels)

            per_method_query_metrics: Dict[str, List[Dict[str, float]]] = defaultdict(list)
            run_rows_by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

            for query_index, query_row in enumerate(queries):
                query_id = query_row["query_id"]
                query_text = query_row["text"]
                relevant = relevant_map.get(query_id, {})
                if not relevant:
                    continue

                if bm25 is not None:
                    bm25_results = bm25.search(query_text, args.final_k)
                    metrics = evaluate_ranked_list(
                        [row["doc_id"] for row in bm25_results],
                        relevant,
                    )
                    per_method_query_metrics["bm25"].append(metrics)
                    run_rows_by_method["bm25"].append(
                        {
                            "query_id": query_id,
                            "query": query_text,
                            "results": bm25_results,
                        }
                    )

                dense_results: List[Dict[str, Any]] = []
                if needs_dense and dense_encoder is not None:
                    dense_results = dense_search(
                        dense_encoder,
                        query_text,
                        domain,
                        args.dense_model,
                        args.candidate_k,
                    )
                    if "dense" in methods:
                        metrics = evaluate_ranked_list(
                            [row["doc_id"] for row in dense_results[: args.final_k]],
                            relevant,
                        )
                        per_method_query_metrics["dense"].append(metrics)
                        run_rows_by_method["dense"].append(
                            {
                                "query_id": query_id,
                                "query": query_text,
                                "results": dense_results[: args.final_k],
                            }
                        )

                late_results = dense_results
                if needs_late and dense_results:
                    if late_reranker is not None:
                        late_results = late_reranker.rerank(
                            query_text,
                            dense_results[: args.late_candidate_k],
                            top_k=args.late_candidate_k,
                            batch_size=args.late_batch_size,
                        )
                    else:
                        # Dense order fallback for debugging without ColBERT.
                        late_results = [
                            {**row, "late_score": row.get("dense_score", 0.0)}
                            for row in dense_results[: args.late_candidate_k]
                        ]

                    if "late" in methods:
                        metrics = evaluate_ranked_list(
                            [row["doc_id"] for row in late_results[: args.final_k]],
                            relevant,
                        )
                        per_method_query_metrics["late"].append(metrics)
                        run_rows_by_method["late"].append(
                            {
                                "query_id": query_id,
                                "query": query_text,
                                "results": late_results[: args.final_k],
                            }
                        )

                if "attr-rule" in methods and late_results:
                    attr_results = rank_with_attributes(
                        query_text,
                        language,
                        late_results,
                        extractor,
                        gate=None,
                        attr_weight=args.attr_weight,
                    )
                    metrics = evaluate_ranked_list(
                        [row["doc_id"] for row in attr_results[: args.final_k]],
                        relevant,
                    )
                    per_method_query_metrics["attr-rule"].append(metrics)
                    run_rows_by_method["attr-rule"].append(
                        {
                            "query_id": query_id,
                            "query": query_text,
                            "results": attr_results[: args.final_k],
                        }
                    )

                if "fusion-only" in methods and late_results:
                    fusion_results = rank_with_attributes(
                        query_text,
                        language,
                        late_results,
                        extractor,
                        gate=None,
                        attr_weight=args.attr_weight,
                        score_mode="fusion_only",
                    )
                    metrics = evaluate_ranked_list(
                        [row["doc_id"] for row in fusion_results[: args.final_k]],
                        relevant,
                    )
                    per_method_query_metrics["fusion-only"].append(metrics)
                    run_rows_by_method["fusion-only"].append(
                        {
                            "query_id": query_id,
                            "query": query_text,
                            "results": fusion_results[: args.final_k],
                        }
                    )

                if "late-attr" in methods and late_results:
                    late_attr_results = rank_with_attributes(
                        query_text,
                        language,
                        late_results,
                        extractor,
                        gate=None,
                        attr_weight=args.attr_weight,
                        score_mode="late_attr",
                    )
                    metrics = evaluate_ranked_list(
                        [row["doc_id"] for row in late_attr_results[: args.final_k]],
                        relevant,
                    )
                    per_method_query_metrics["late-attr"].append(metrics)
                    run_rows_by_method["late-attr"].append(
                        {
                            "query_id": query_id,
                            "query": query_text,
                            "results": late_attr_results[: args.final_k],
                        }
                    )

                if "attr-gate" in methods and late_results:
                    gate_results = rank_with_attributes(
                        query_text,
                        language,
                        late_results,
                        extractor,
                        gate=gate,
                        attr_weight=args.attr_weight,
                    )
                    metrics = evaluate_ranked_list(
                        [row["doc_id"] for row in gate_results[: args.final_k]],
                        relevant,
                    )
                    per_method_query_metrics["attr-gate"].append(metrics)
                    run_rows_by_method["attr-gate"].append(
                        {
                            "query_id": query_id,
                            "query": query_text,
                            "results": gate_results[: args.final_k],
                        }
                    )

                if (query_index + 1) % args.log_every == 0:
                    logging.info(
                        "domain=%s lang=%s queries=%d/%d",
                        domain,
                        language,
                        query_index + 1,
                        len(queries),
                    )

            for method, query_metrics in per_method_query_metrics.items():
                summary = aggregate_metrics(query_metrics)
                metric_rows.append(
                    {
                        "domain": domain,
                        "language": language,
                        "method": method,
                        "num_queries": len(query_metrics),
                        **summary,
                    }
                )
                save_run(
                    run_root / f"{domain}_{language}_{method}.jsonl",
                    run_rows_by_method[method],
                )

    frame = pd.DataFrame(metric_rows)
    out_path = paths["results"] / "cure_metrics.csv"
    frame.to_csv(out_path, index=False)

    overall = (
        frame.groupby(["language", "method"])
        .apply(
            lambda group: pd.Series(
                {
                    "num_queries": int(group["num_queries"].sum()),
                    **{
                        metric: (
                            group[metric] * group["num_queries"]
                        ).sum()
                        / max(group["num_queries"].sum(), 1)
                        for metric in ["ndcg@10", "mrr@10", "map@10", "recall@5", "recall@10"]
                        if metric in group.columns
                    },
                }
            )
        )
        .reset_index()
    )
    overall_path = paths["results"] / "cure_metrics_overall.csv"
    overall.to_csv(overall_path, index=False)
    logging.info("Saved: %s", out_path)
    logging.info("Saved: %s", overall_path)


# ---------------------------------------------------------------------------
# Counterfactual CAPR evaluation
# ---------------------------------------------------------------------------


def bootstrap_mean_ci(
    values: Sequence[float],
    iterations: int = 5000,
    seed: int = 42,
) -> Tuple[float, float]:
    import numpy as np

    arr = np.asarray(values, dtype="float64")
    if len(arr) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype="float64")
    for i in range(iterations):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means[i] = sample.mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def command_evaluate_pairs(args: argparse.Namespace) -> None:
    import pandas as pd
    import torch

    paths = ensure_project_dirs()
    frame = load_approved_pairs(args.pairs_csv, split=args.test_split)
    if args.max_pairs:
        frame = frame.head(args.max_pairs)

    extractor = ClinicalAttributeExtractor()
    dense_encoder = DenseEncoder(
        model_name_or_path=args.dense_model,
        device=args.dense_device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )
    late_reranker = None
    if not args.no_late:
        late_reranker = LateInteractionReranker(
            model_name_or_path=args.late_model,
            device=args.late_device,
        )
    gate = None
    if args.gate_checkpoint is not None:
        gate = load_gate(args.gate_checkpoint, device=args.gate_device)

    output_rows: List[Dict[str, Any]] = []
    for index, row in frame.iterrows():
        # Original-query direction (A_o): does the retriever prefer the real
        # passage over its counterfactual, when asked the original query?
        pos_features, neg_features, diagnostics = score_pair_features(
            dense_encoder,
            late_reranker,
            str(row["original_query"]),
            str(row["positive_passage"]),
            str(row["negative_passage"]),
            str(row["language"]),
            extractor,
        )

        norm_stats = getattr(gate, "norm_stats", None) if gate is not None else None
        if norm_stats:
            pos_features, neg_features = apply_global_norm(
                pos_features, neg_features, diagnostics, norm_stats
            )

        dense_pos = diagnostics["dense_pos"]
        dense_neg = diagnostics["dense_neg"]
        late_pos = diagnostics["late_pos"]
        late_neg = diagnostics["late_neg"]
        rule_pos = rule_attribute_score(pos_features, attr_weight=args.attr_weight)
        rule_neg = rule_attribute_score(neg_features, attr_weight=args.attr_weight)

        gate_pos = gate_neg = float("nan")
        if gate is not None:
            tensor = torch.tensor(
                [pos_features, neg_features],
                dtype=torch.float32,
                device=gate.raw_weights.device,
            )
            with torch.no_grad():
                gate_scores = gate.score(tensor).cpu().tolist()
            gate_pos, gate_neg = map(float, gate_scores)

        # Counterfactual-query direction (A_c): does the retriever reverse
        # its preference to favour the counterfactual passage, when asked
        # the counterfactual query? This is the direction the original
        # pipeline never computed. A method that always prefers the real
        # CURE passage regardless of which query is asked will score well
        # on A_o and poorly on A_c -- that gap is exactly what one-directional
        # CAPR was hiding.
        c_pos_features, c_neg_features, c_diagnostics = score_pair_features(
            dense_encoder,
            late_reranker,
            str(row["counterfactual_query"]),
            str(row["negative_passage"]),  # d_c is "positive" in this direction
            str(row["positive_passage"]),  # d_o is "negative" in this direction
            str(row["language"]),
            extractor,
        )

        if norm_stats:
            c_pos_features, c_neg_features = apply_global_norm(
                c_pos_features, c_neg_features, c_diagnostics, norm_stats
            )

        c_dense_pos = c_diagnostics["dense_pos"]
        c_dense_neg = c_diagnostics["dense_neg"]
        c_late_pos = c_diagnostics["late_pos"]
        c_late_neg = c_diagnostics["late_neg"]
        c_rule_pos = rule_attribute_score(c_pos_features, attr_weight=args.attr_weight)
        c_rule_neg = rule_attribute_score(c_neg_features, attr_weight=args.attr_weight)

        c_gate_pos = c_gate_neg = float("nan")
        if gate is not None:
            c_tensor = torch.tensor(
                [c_pos_features, c_neg_features],
                dtype=torch.float32,
                device=gate.raw_weights.device,
            )
            with torch.no_grad():
                c_gate_scores = gate.score(c_tensor).cpu().tolist()
            c_gate_pos, c_gate_neg = map(float, c_gate_scores)

        dense_o_correct = int(dense_pos > dense_neg)
        dense_c_correct = int(c_dense_pos > c_dense_neg)
        late_o_correct = int(late_pos > late_neg)
        late_c_correct = int(c_late_pos > c_late_neg)
        rule_o_correct = int(rule_pos > rule_neg)
        rule_c_correct = int(c_rule_pos > c_rule_neg)
        gate_o_correct = int(gate_pos > gate_neg) if gate is not None else ""
        gate_c_correct = int(c_gate_pos > c_gate_neg) if gate is not None else ""

        output_rows.append(
            {
                "pair_id": row["pair_id"],
                "language": row["language"],
                "domain": row.get("domain", ""),
                "attribute_type": row["attribute_type"],
                "dense_positive": dense_pos,
                "dense_negative": dense_neg,
                "dense_margin": dense_pos - dense_neg,
                "dense_correct": dense_o_correct,
                "dense_c_positive": c_dense_pos,
                "dense_c_negative": c_dense_neg,
                "dense_c_margin": c_dense_pos - c_dense_neg,
                "dense_c_correct": dense_c_correct,
                "dense_pair_consistent": int(bool(dense_o_correct) and bool(dense_c_correct)),
                "late_positive": late_pos,
                "late_negative": late_neg,
                "late_margin": late_pos - late_neg,
                "late_correct": late_o_correct,
                "late_c_positive": c_late_pos,
                "late_c_negative": c_late_neg,
                "late_c_margin": c_late_pos - c_late_neg,
                "late_c_correct": late_c_correct,
                "late_pair_consistent": int(bool(late_o_correct) and bool(late_c_correct)),
                "attr_rule_positive": rule_pos,
                "attr_rule_negative": rule_neg,
                "attr_rule_margin": rule_pos - rule_neg,
                "attr_rule_correct": rule_o_correct,
                "attr_rule_c_positive": c_rule_pos,
                "attr_rule_c_negative": c_rule_neg,
                "attr_rule_c_margin": c_rule_pos - c_rule_neg,
                "attr_rule_c_correct": rule_c_correct,
                "attr_rule_pair_consistent": int(bool(rule_o_correct) and bool(rule_c_correct)),
                "attr_gate_positive": gate_pos,
                "attr_gate_negative": gate_neg,
                "attr_gate_margin": gate_pos - gate_neg if gate is not None else float("nan"),
                "attr_gate_correct": gate_o_correct,
                "attr_gate_c_positive": c_gate_pos,
                "attr_gate_c_negative": c_gate_neg,
                "attr_gate_c_margin": c_gate_pos - c_gate_neg if gate is not None else float("nan"),
                "attr_gate_c_correct": gate_c_correct,
                "attr_gate_pair_consistent": (
                    int(bool(gate_o_correct) and bool(gate_c_correct))
                    if gate is not None
                    else ""
                ),
                "positive_features": json.dumps(
                    dict(zip(FEATURE_NAMES, pos_features))
                ),
                "negative_features": json.dumps(
                    dict(zip(FEATURE_NAMES, neg_features))
                ),
            }
        )
        if len(output_rows) % args.log_every == 0:
            logging.info("Evaluated pairs: %d/%d", len(output_rows), len(frame))

    result_frame = pd.DataFrame(output_rows)
    detailed_path = paths["results"] / "counterfactual_pair_scores.csv"
    result_frame.to_csv(detailed_path, index=False)

    method_prefixes = ["dense", "late", "attr_rule"]
    if gate is not None:
        method_prefixes.append("attr_gate")

    summaries = []
    group_columns = [
        [],
        ["language"],
        ["attribute_type"],
        ["language", "attribute_type"],
    ]
    for group_cols in group_columns:
        if group_cols:
            grouped = result_frame.groupby(group_cols, dropna=False)
        else:
            grouped = [("overall", result_frame)]
        for group_key, group in grouped:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            label = (
                {"scope": "overall"}
                if not group_cols
                else {"scope": "+".join(group_cols), **dict(zip(group_cols, group_key))}
            )
            for method in method_prefixes:
                correct_values = group[f"{method}_correct"].astype(float).tolist()
                margin_values = group[f"{method}_margin"].astype(float).tolist()
                capr = statistics.fmean(correct_values)
                lo, hi = bootstrap_mean_ci(
                    correct_values,
                    iterations=args.bootstrap_iterations,
                    seed=args.seed,
                )

                # Symmetric metrics. capr above is A_o (kept for continuity
                # with prior results); the following are new.
                c_correct_values = group[f"{method}_c_correct"].astype(float).tolist()
                capr_c = statistics.fmean(c_correct_values)
                c_lo, c_hi = bootstrap_mean_ci(
                    c_correct_values,
                    iterations=args.bootstrap_iterations,
                    seed=args.seed,
                )

                pair_consistent_values = group[f"{method}_pair_consistent"].astype(float).tolist()
                pair_consistency = statistics.fmean(pair_consistent_values)
                pc_lo, pc_hi = bootstrap_mean_ci(
                    pair_consistent_values,
                    iterations=args.bootstrap_iterations,
                    seed=args.seed,
                )

                capr_balanced = (capr + capr_c) / 2.0

                summaries.append(
                    {
                        **label,
                        "method": method,
                        "n": len(group),
                        "capr": capr,
                        "capr_ci95_low": lo,
                        "capr_ci95_high": hi,
                        "capr_counterfactual_direction": capr_c,
                        "capr_counterfactual_direction_ci95_low": c_lo,
                        "capr_counterfactual_direction_ci95_high": c_hi,
                        "capr_balanced": capr_balanced,
                        "pair_consistency": pair_consistency,
                        "pair_consistency_ci95_low": pc_lo,
                        "pair_consistency_ci95_high": pc_hi,
                        "mean_margin": statistics.fmean(margin_values),
                    }
                )

    summary_frame = pd.DataFrame(summaries)
    summary_path = paths["results"] / "counterfactual_capr_summary.csv"
    summary_frame.to_csv(summary_path, index=False)
    logging.info("Saved: %s", detailed_path)
    logging.info("Saved: %s", summary_path)


# ---------------------------------------------------------------------------
# LLM listwise reranking baseline
# ---------------------------------------------------------------------------


LLM_RERANK_SYSTEM = """
You are a medical evidence retrieval reranker.

Rank candidate English passages for a clinical query. Prioritise:
1. direct relevance to the clinical information need;
2. exact preservation of decisive attributes such as negation, dose, unit,
   frequency, disease subtype, stage, temporal status, and severity;
3. evidence quality and specificity.

Do not answer the medical question. Return JSON only:
{
  "ranking": ["D3", "D1", "D2"],
  "notes": {
    "D3": "brief relevance and attribute justification"
  }
}
""".strip()


def llm_rerank_candidates(
    llm: LocalLLM,
    query: str,
    language: str,
    candidates: Sequence[Mapping[str, Any]],
    max_chars_per_doc: int,
    seed: int,
) -> List[Dict[str, Any]]:
    document_blocks = []
    id_to_row = {}
    for index, row in enumerate(candidates, start=1):
        local_id = f"D{index}"
        id_to_row[local_id] = dict(row)
        document_blocks.append(
            f"{local_id}:\n{str(row['text'])[:max_chars_per_doc]}"
        )
    user = (
        f"Query language: {language}\n"
        f"Clinical query:\n{query}\n\n"
        f"Candidate passages:\n\n" + "\n\n".join(document_blocks)
    )
    raw = llm.generate(
        LLM_RERANK_SYSTEM,
        user,
        max_new_tokens=512,
        temperature=0.0,
        seed=seed,
    )
    parsed = extract_json_object(raw)
    ranking = parsed.get("ranking", [])
    if not isinstance(ranking, list):
        raise ValueError("LLM ranking is not a list.")

    output = []
    seen = set()
    for local_id in ranking:
        local_id = str(local_id)
        if local_id in id_to_row and local_id not in seen:
            row = id_to_row[local_id]
            row["llm_note"] = (parsed.get("notes") or {}).get(local_id, "")
            output.append(row)
            seen.add(local_id)
    for local_id, row in id_to_row.items():
        if local_id not in seen:
            output.append(row)
    for rank, row in enumerate(output, start=1):
        row["rank"] = rank
    return output


def command_llm_rerank(args: argparse.Namespace) -> None:
    import pandas as pd

    paths = ensure_project_dirs()
    domains = parse_csv_list(args.domains, DOMAINS)
    languages = parse_csv_list(args.languages, LANGUAGES)
    model_names = parse_csv_list(
        args.llm_models,
        ["qwen", "llama", "biomistral"],
    )

    # Reuse saved dense/late run candidates. If no late run exists, use dense.
    metric_rows = []
    for model_index, model_name in enumerate(model_names):
        llm = LocalLLM(
            LLM_PATHS[model_name],
            device=args.device,
            load_in_4bit=not args.no_4bit,
        )
        for domain in domains:
            _, _, qrels = load_domain_data(domain, "en")
            for language in languages:
                _, queries, _ = load_domain_data(domain, language)
                if args.max_queries:
                    queries = queries[: args.max_queries]
                relevant_map = qrels_for_language(queries, qrels)
                query_map = {row["query_id"]: row for row in queries}

                candidate_file = (
                    paths["results"]
                    / "runs"
                    / f"{domain}_{language}_{args.candidate_method}.jsonl"
                )
                if not candidate_file.exists():
                    raise FileNotFoundError(
                        f"Candidate run missing: {candidate_file}. "
                        "Run evaluate-cure first."
                    )
                candidate_runs = read_jsonl(candidate_file)
                per_query = []
                output_runs = []
                for idx, run in enumerate(candidate_runs):
                    qid = run["query_id"]
                    if qid not in query_map:
                        continue
                    candidates = run["results"][: args.top_k]
                    reranked = llm_rerank_candidates(
                        llm,
                        run["query"],
                        language,
                        candidates,
                        max_chars_per_doc=args.max_chars_per_doc,
                        seed=args.seed + model_index * 100000 + idx,
                    )
                    metrics = evaluate_ranked_list(
                        [row["doc_id"] for row in reranked],
                        relevant_map.get(qid, {}),
                    )
                    per_query.append(metrics)
                    output_runs.append(
                        {
                            "query_id": qid,
                            "query": run["query"],
                            "results": reranked,
                        }
                    )
                    if (idx + 1) % args.log_every == 0:
                        logging.info(
                            "model=%s domain=%s lang=%s %d/%d",
                            model_name,
                            domain,
                            language,
                            idx + 1,
                            len(candidate_runs),
                        )

                summary = aggregate_metrics(per_query)
                metric_rows.append(
                    {
                        "model": model_name,
                        "domain": domain,
                        "language": language,
                        "candidate_method": args.candidate_method,
                        "num_queries": len(per_query),
                        **summary,
                    }
                )
                save_run(
                    paths["results"]
                    / "runs"
                    / f"{domain}_{language}_llm-{model_name}.jsonl",
                    output_runs,
                )

        del llm
        cleanup_cuda()

    out_path = paths["results"] / "llm_rerank_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(out_path, index=False)
    logging.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Pilot orchestration
# ---------------------------------------------------------------------------


def command_pilot(args: argparse.Namespace) -> None:
    """
    Runs a controlled one-domain pilot by invoking this file as subprocesses.
    This keeps GPU-heavy components isolated and ensures memory is released.
    """
    script = Path(__file__).resolve()
    common = [sys.executable, str(script)]

    def run(parts: List[str]) -> None:
        import subprocess

        logging.info("RUN: %s", " ".join(parts))
        subprocess.run(parts, check=True)

    run(
        common
        + [
            "prepare",
            "--domains",
            args.domain,
            "--languages",
            "en,fr,es",
        ]
    )
    run(
        common
        + [
            "audit",
            "--domains",
            args.domain,
            "--languages",
            "en,fr,es",
        ]
    )
    run(
        common
        + [
            "index-dense",
            "--domains",
            args.domain,
            "--dense-model",
            args.dense_model,
            "--device",
            args.dense_device,
        ]
    )
    run(
        common
        + [
            "evaluate-cure",
            "--domains",
            args.domain,
            "--languages",
            "en,fr,es",
            "--max-queries",
            str(args.max_queries),
            "--methods",
            "bm25,dense,late,attr-rule",
            "--dense-model",
            args.dense_model,
            "--late-model",
            args.late_model,
            "--dense-device",
            args.dense_device,
            "--late-device",
            args.late_device,
        ]
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_common_retrieval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--late-model", default=DEFAULT_LATE_MODEL)
    parser.add_argument("--dense-device", default="cuda:0")
    parser.add_argument("--late-device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--late-batch-size", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=512)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Attribute-preserving multilingual clinical retrieval.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor")
    doctor.set_defaults(func=command_doctor)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--domains", default="all")
    prepare.add_argument("--languages", default="en,fr,es")
    prepare.add_argument("--force", action="store_true")
    prepare.set_defaults(func=command_prepare)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--domains", default="all")
    audit.add_argument("--languages", default="en,fr,es")
    audit.add_argument("--max-examples", type=int, default=100)
    audit.set_defaults(func=command_audit)

    index_dense = subparsers.add_parser("index-dense")
    index_dense.add_argument("--domains", default="all")
    index_dense.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    index_dense.add_argument("--device", default="cuda:0")
    index_dense.add_argument("--batch-size", type=int, default=32)
    index_dense.add_argument("--max-seq-length", type=int, default=512)
    index_dense.add_argument("--force", action="store_true")
    index_dense.set_defaults(func=command_index_dense)

    generate = subparsers.add_parser("generate-pairs")
    generate.add_argument("--domains", default="all")
    generate.add_argument("--languages", default="en,fr,es")
    generate.add_argument("--num-pairs", type=int, default=300)
    generate.add_argument(
        "--generator-model",
        choices=["qwen", "llama", "biomistral"],
        default="qwen",
    )
    generate.add_argument(
        "--validator-model",
        choices=["qwen", "llama", "biomistral"],
        default="biomistral",
    )
    generate.add_argument("--generator-device", default="cuda:0")
    generate.add_argument("--validator-device", default="cuda:0")
    generate.add_argument("--max-new-tokens", type=int, default=1536)
    generate.add_argument("--temperature", type=float, default=0.2)
    generate.add_argument("--no-4bit", action="store_true")
    generate.add_argument("--resume", action="store_true")
    generate.add_argument("--log-every", type=int, default=5)
    generate.add_argument("--out-suffix", default="",
                          help="appended to output file names, e.g. _llamagen")
    generate.set_defaults(func=command_generate_pairs)

    train = subparsers.add_parser("train-gate")
    train.add_argument("--pairs-csv", type=Path, required=True)
    train.add_argument("--train-split", default=None)
    train.add_argument("--max-pairs", type=int, default=None)
    add_common_retrieval_args(train)
    train.add_argument("--no-late", action="store_true")
    train.add_argument("--gate-device", default="cuda:2")
    train.add_argument("--learning-rate", type=float, default=2e-2)
    train.add_argument("--weight-decay", type=float, default=1e-3)
    train.add_argument("--l1-weight", type=float, default=1e-3)
    train.add_argument("--margin", type=float, default=0.5)
    train.add_argument("--epochs", type=int, default=500)
    train.add_argument("--patience", type=int, default=50)
    train.add_argument("--dev-fraction", type=float, default=0.2)
    train.add_argument(
        "--feature-mode",
        choices=["full", "fusion_only"],
        default="full",
        help="'full' uses attribute features; 'fusion_only' masks them out (dense/late only) for the ablation.",
    )
    train.add_argument(
        "--norm",
        choices=["pairwise", "global"],
        default="pairwise",
        help="'pairwise' z-normalizes dense/late scores over the two candidates "
        "(original behaviour); 'global' standardizes them with training-set "
        "statistics so score magnitudes are preserved.",
    )
    train.add_argument("--output-checkpoint", type=Path, default=None)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--log-every", type=int, default=10)
    train.set_defaults(func=command_train_gate)

    evaluate = subparsers.add_parser("evaluate-cure")
    evaluate.add_argument("--domains", default="all")
    evaluate.add_argument("--languages", default="en,fr,es")
    evaluate.add_argument(
        "--methods",
        default="bm25,dense,late,attr-rule",
    )
    add_common_retrieval_args(evaluate)
    evaluate.add_argument("--gate-checkpoint", type=Path, default=None)
    evaluate.add_argument("--gate-device", default="cpu")
    evaluate.add_argument("--max-queries", type=int, default=None)
    evaluate.add_argument("--candidate-k", type=int, default=100)
    evaluate.add_argument("--late-candidate-k", type=int, default=50)
    evaluate.add_argument("--final-k", type=int, default=10)
    evaluate.add_argument("--attr-weight", type=float, default=0.35)
    evaluate.add_argument("--skip-late", action="store_true")
    evaluate.add_argument("--log-every", type=int, default=10)
    evaluate.set_defaults(func=command_evaluate_cure)

    eval_pairs = subparsers.add_parser("evaluate-pairs")
    eval_pairs.add_argument("--pairs-csv", type=Path, required=True)
    eval_pairs.add_argument("--test-split", default=None)
    eval_pairs.add_argument("--gate-checkpoint", type=Path, default=None)
    add_common_retrieval_args(eval_pairs)
    eval_pairs.add_argument("--gate-device", default="cpu")
    eval_pairs.add_argument("--no-late", action="store_true")
    eval_pairs.add_argument("--max-pairs", type=int, default=None)
    eval_pairs.add_argument("--attr-weight", type=float, default=0.35)
    eval_pairs.add_argument("--bootstrap-iterations", type=int, default=5000)
    eval_pairs.add_argument("--log-every", type=int, default=10)
    eval_pairs.set_defaults(func=command_evaluate_pairs)

    llm_rerank = subparsers.add_parser("llm-rerank")
    llm_rerank.add_argument("--domains", default="all")
    llm_rerank.add_argument("--languages", default="en,fr,es")
    llm_rerank.add_argument(
        "--llm-models",
        default="biomistral",
        help="Comma-separated qwen,llama,biomistral.",
    )
    llm_rerank.add_argument(
        "--candidate-method",
        choices=["dense", "late", "attr-rule", "attr-gate"],
        default="late",
    )
    llm_rerank.add_argument("--device", default="cuda:0")
    llm_rerank.add_argument("--top-k", type=int, default=10)
    llm_rerank.add_argument("--max-queries", type=int, default=50)
    llm_rerank.add_argument("--max-chars-per-doc", type=int, default=1800)
    llm_rerank.add_argument("--no-4bit", action="store_true")
    llm_rerank.add_argument("--log-every", type=int, default=5)
    llm_rerank.set_defaults(func=command_llm_rerank)

    pilot = subparsers.add_parser("pilot")
    pilot.add_argument(
        "--domain",
        choices=DOMAINS,
        default="pulmonology",
    )
    pilot.add_argument("--max-queries", type=int, default=20)
    pilot.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    pilot.add_argument("--late-model", default=DEFAULT_LATE_MODEL)
    pilot.add_argument("--dense-device", default="cuda:0")
    pilot.add_argument("--late-device", default="cuda:1")
    pilot.set_defaults(func=command_pilot)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    paths = ensure_project_dirs()
    configure_logging(
        paths["logs"] / f"{args.command}.log",
        verbose=args.verbose,
    )
    set_seed(args.seed)
    logging.info("Command: %s", args.command)
    logging.info("Project root: %s", PROJECT_ROOT)
    args.func(args)


if __name__ == "__main__":
    main()