from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from profiles.utils.git_utils import clone_repo
from profiles.utils.repository_id import get_repository_id

from profiles.ai_usage_profile_eop.droid_detect.droiddetect_large import compute_droiddetect_metrics, load_model

from pathlib import Path
import csv

CODEBASES_TO_ANALYZE_PATH = Path(__file__).parent.parent.parent / "input_files" / "github_links.csv"
CLONES_DIR = Path(__file__).parent.parent.parent / "system_files" / "codebase_clones"

def compute_ai_usage_profile_eop(rows):
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_codebase, r["Year"], r["Link"], droiddetect_tokenizer, droiddetect_model) for r in rows]
        for f in as_completed(futures):
            print(f.result())
        
def process_codebase(year, link, droiddetect_tokenizer, droiddetect_model):
    repository_id = get_repository_id(year, link)
    print(f"Processing codebase: {repository_id}.")

    repo_path = clone_repo(link, CLONES_DIR, int(year))

    print("Computing droiddetect metrics...")
    compute_droiddetect_metrics(
        repo_path, repository_id, droiddetect_tokenizer, droiddetect_model
    )

    return f"Finished processing codebase: {repository_id}"

def combine_ai_usage_profiles_eop(rows):
    current_dir = Path.cwd()
    droiddetect_eop_dir = current_dir / "system_files" / "droiddetect_metrics_eop"

    output_csv = current_dir / "output_files" / "ai_usage_profile_eop.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    combined_rows = []
    seen_headers = ["repository_id", "project_year", "filepath"]

    for row in rows:
        year = row["Year"]
        link = row["Link"]
        repository_id = get_repository_id(year, link)

        droiddetect_csv = droiddetect_eop_dir / f"{repository_id}_droiddetect_metrics.csv"

        with open(droiddetect_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            droiddetect_headers = reader.fieldnames or []
            droiddetect_rows = {r["filepath"]: r for r in reader}

        for h in droiddetect_headers:
            if h not in seen_headers and h != "filepath":
                seen_headers.append(h)

        # Only emit rows where filepath exists in both
        common_filepaths = set(droiddetect_rows.keys())
        if not common_filepaths:
            print(f"[WARNING] No common filepaths for {repository_id}, skipping.")
            continue

        for filepath in sorted(common_filepaths):
            combined_rows.append({
                "repository_id": repository_id,
                "project_year": year,
                "filepath": filepath,
                **{k: v for k, v in droiddetect_rows[filepath].items() if k != "filepath"},
            })

    combined_rows.sort(key=lambda x: (
        x.get("project_year", ""),
        x.get("repository_id", ""),
        x.get("filepath", ""),
    ))

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=seen_headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined_rows)

    print(f"[INFO] Combined AI usage profile written to {output_csv} ({len(combined_rows)} rows)")
    
if __name__ == "__main__":
    droiddetect_tokenizer, droiddetect_model = load_model()
    
    with open(CODEBASES_TO_ANALYZE_PATH, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    print(f"Loaded {len(rows)} codebases to analyze.")
    compute_ai_usage_profile_eop(rows)
    combine_ai_usage_profiles_eop(rows)
