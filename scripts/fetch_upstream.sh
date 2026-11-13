#!/usr/bin/env bash
# Mirrors read-only, public files from sksoumik/judging-the-judges into data/upstream/.
# No API key needed (public GitHub raw content, free). Re-running is safe (idempotent overwrite).
set -euo pipefail

# Run this script from the project root (judging-the-judges-study/), e.g.:
#   bash scripts/fetch_upstream.sh
REPO_RAW="https://raw.githubusercontent.com/sksoumik/judging-the-judges/main"
API_TREE="https://api.github.com/repos/sksoumik/judging-the-judges/git/trees/main?recursive=1"
DEST="data/upstream"

if [ ! -d "src" ] || [ ! -d "scripts" ]; then
  echo "run this from the judging-the-judges-study project root" >&2
  exit 1
fi

mkdir -p "$DEST"

# Files we actually need for E0 (aggregated-only scope) + reconciliation tables + configs.
# results/raw/ and data/human_study/ are excluded: confirmed .gitignore'd upstream, not fetchable.
FILES=(
  "README.md"
  "configs/experiments.yaml"
  "configs/models.yaml"
  "configs/strategies.yaml"
  "analysis/figures/table1_main_results.csv"
  "analysis/figures/table2_bias_summary.csv"
  "analysis/figures/table1_kappa.csv"
  "analysis/figures/table3_length_split.csv"
  "analysis/figures/table_llmbar_results.csv"
  "analysis/figures/table_llmbar_with_ci.csv"
  "data/custom/controlled_pairs.jsonl"
  "data/raw/mt_bench/mt_bench.jsonl"
  "data/raw/llmbar/llmbar.jsonl"
)

for f in "${FILES[@]}"; do
  mkdir -p "$DEST/$(dirname "$f")"
  echo "fetching $f"
  curl -sf "$REPO_RAW/$f" -o "$DEST/$f"
done

mkdir -p "$DEST/results/aggregated"
echo "listing results/aggregated via git trees API"
curl -sf "$API_TREE" | (command -v python3 >/dev/null && python3 || python) -c "
import json, sys
tree = json.load(sys.stdin)['tree']
for item in tree:
    if item['path'].startswith('results/aggregated/') and item['type'] == 'blob':
        print(item['path'])
" | tr -d '\r' > "$DEST/.aggregated_file_list.txt"

while read -r path; do
  echo "fetching $path"
  curl -sf "$REPO_RAW/$path" -o "$DEST/$path"
done < "$DEST/.aggregated_file_list.txt"

echo "done: $(wc -l < "$DEST/.aggregated_file_list.txt") aggregated metrics files mirrored"
