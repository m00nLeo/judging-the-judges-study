"""Cross-platform mirror of public files from sksoumik/judging-the-judges into data/upstream/.

Works identically on Windows (PowerShell/cmd), macOS, Linux, and Google Colab
without requiring bash, curl, or tr.
"""

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_RAW = "https://raw.githubusercontent.com/sksoumik/judging-the-judges/main"
API_TREE = "https://api.github.com/repos/sksoumik/judging-the-judges/git/trees/main?recursive=1"

STATIC_FILES = [
    "README.md",
    "configs/experiments.yaml",
    "configs/models.yaml",
    "configs/strategies.yaml",
    "analysis/figures/table1_main_results.csv",
    "analysis/figures/table2_bias_summary.csv",
    "analysis/figures/table1_kappa.csv",
    "analysis/figures/table3_length_split.csv",
    "analysis/figures/table_llmbar_results.csv",
    "analysis/figures/table_llmbar_with_ci.csv",
    "data/custom/controlled_pairs.jsonl",
    "data/raw/mt_bench/mt_bench.jsonl",
    "data/raw/llmbar/llmbar.jsonl",
]


def _download_one(rel_path: str, dest_root: Path) -> str:
    out_path = dest_root / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{REPO_RAW}/{rel_path}"
    req = urllib.request.Request(url, headers={"User-Agent": "judging-the-judges-study/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        out_path.write_bytes(resp.read())
    return rel_path


def fetch_upstream(project_root: Path | None = None, max_workers: int = 8) -> Path:
    root = project_root or Path(__file__).resolve().parents[1]
    dest = root / "data" / "upstream"
    dest.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(API_TREE, headers={"User-Agent": "judging-the-judges-study/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        tree = json.loads(resp.read().decode("utf-8"))["tree"]

    agg_paths = [
        item["path"]
        for item in tree
        if item["path"].startswith("results/aggregated/") and item["type"] == "blob"
    ]
    (dest / ".aggregated_file_list.txt").write_text("\n".join(agg_paths) + "\n", encoding="utf-8")

    all_paths = STATIC_FILES + agg_paths
    print(f"Downloading {len(STATIC_FILES)} static files + {len(agg_paths)} aggregated metrics files...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(lambda p: _download_one(p, dest), all_paths))

    print(f"done: {len(agg_paths)} aggregated metrics files mirrored into {dest}")
    return dest


if __name__ == "__main__":
    fetch_upstream()

