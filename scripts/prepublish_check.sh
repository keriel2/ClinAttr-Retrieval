#!/usr/bin/env bash
# prepublish_check.sh   run from the repository root before making the repository public.
# Prints findings; exits non-zero if anything needs attention.
set -u
bad=0
note() { echo "[CHECK] $*"; bad=1; }

echo "1. Secrets and tokens"
grep -rInE --exclude-dir=.git --exclude=prepublish_check.sh \
  '(hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|BEGIN (RSA|OPENSSH|PRIVATE) KEY|api[_-]?key *[=:]|password *[=:])' . \
  && note "possible secret above"

echo "2. Absolute paths and private server names"
grep -rInE --exclude-dir=.git --exclude=prepublish_check.sh --exclude=patch_portable_paths.py \
  '(/home/[a-z]+/|/root/|/Users/[A-Za-z]+/|C:\\Users\\)' . \
  && note "absolute paths above (run scripts/patch_portable_paths.py, then re-check docstrings)"

echo "3. Email addresses"
grep -rInE --exclude-dir=.git --exclude=prepublish_check.sh \
  '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' . \
  && note "email addresses above (keep only ones you want public)"

echo "4. Large files (over 50 MB) and model weights"
find . -path ./.git -prune -o -type f \( -size +50M -o -name '*.pt' -o -name '*.safetensors' -o -name '*.bin' \) -print | grep . \
  && note "large or weight files above (use releases or Zenodo, not git)"

echo "5. Files that must not be public"
find . -path ./.git -prune -o -type f \( -name 'annotation_key.csv' -o -name 'annotation_*_done.xlsx' -o -name '.env' \) -print | grep . \
  && note "private files above"

echo "6. Required files"
for f in README.md LICENSE DATA_LICENSE.md CITATION.cff DATA_SCHEMA.md MODELS.md REPRODUCE.md; do
  [ -f "$f" ] || note "missing $f"
done

echo "7. Unfilled placeholders"
grep -rIn --exclude-dir=.git --exclude=prepublish_check.sh -E 'TODO|\[commit hash|\[confirm|\[paste' . \
  && note "placeholders above"

[ $bad -eq 0 ] && echo "All checks passed." || echo "Fix the items marked [CHECK] before publishing."
exit $bad
