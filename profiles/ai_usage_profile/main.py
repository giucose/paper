from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from profiles.utils.git_utils import extract_each_commit
from profiles.utils.repository_id import get_repository_id

from profiles.ai_usage_profile.developer_metrics.compute_developer_metrics import compute_developer_activity_metrics_data
from profiles.ai_usage_profile.droid_detect.droiddetect_large import compute_droiddetect_metrics, load_model

from pathlib import Path
import csv

CODEBASES_TO_ANALYZE_PATH = Path(__file__).parent.parent.parent / "input_files" / "github_links.csv"
CLONES_DIR = Path(__file__).parent.parent.parent / "system_files" / "codebase_clones"

def compute_ai_usage_profile(rows):
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_codebase, r["Year"], r["Link"], droiddetect_tokenizer, droiddetect_model) for r in rows]
        for f in as_completed(futures):
            print(f.result())
        
    # get each codebase's path and commits to analyze from github_links.csv
    
    # for each commit, get following:
    # - developer metrics
    # - droiddetect results
    # - whodunit results

    # combine into AI usage profile

def process_codebase(year, link, droiddetect_tokenizer, droiddetect_model):

    commits, repo_path = extract_each_commit(link, int(year), CLONES_DIR)
    repository_id = get_repository_id(year, link)
    current_dir = Path.cwd()

    individual_repository_data_dir = (
        current_dir
        / "profiles"
        / "ai_usage_profile"
        / "individual_repository_data"
        / str(year)
        / repository_id
    )

    individual_ai_usage_profile = Path(
        individual_repository_data_dir / "ai_usage_profile.csv"
    )

    # accumulator
    print(f"Processing codebase: {repository_id}, found {len(commits)} commits to analyze.")
    print("Computing developer metrics...")

    values_developer_metrics, headers_developer_metrics = compute_developer_activity_metrics_data(
        commits, repository_id, repo_path, individual_repository_data_dir
    )

    print("Computing droiddetect metrics...")
    values_droiddetect, headers_droiddetect = compute_droiddetect_metrics(
        commits, repo_path, repository_id, year, droiddetect_tokenizer, droiddetect_model, individual_repository_data_dir
    )
    
    # All values are dicts keyed by commit, so we can combine them by iterating over commits and pulling from each dict
    # All headers are lists, so we can combine them into one list of headers for the final CSV
    
    headers_combined = (
        ["repository_id", "project_year", "snapshot_commit"] +
        [h for h in headers_developer_metrics if h != "commit"] +
        [h for h in headers_droiddetect if h != "commit"]
    )

    ai_usage_profile = []

    for commit in commits:
        row = {}

        row["repository_id"] = repository_id
        row["snapshot_commit"] = commit
        row["project_year"] = year

        # Merge without duplicating "commit"
        for source in (
            values_developer_metrics.get(commit, {}),
            values_droiddetect.get(commit, {}),
        ):
            for k, v in source.items():
                if k != "commit":
                    row[k] = v

        if all(header in row and row[header] is not None for header in headers_combined):
            ai_usage_profile.append(row)

    if os.path.exists(individual_ai_usage_profile):
        os.remove(individual_ai_usage_profile)

    individual_repository_data_dir.parent.mkdir(parents=True, exist_ok=True)

    with open(individual_ai_usage_profile, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers_combined)
        writer.writeheader()
        writer.writerows(ai_usage_profile)
        
    return f"Finished processing codebase: {repository_id}, processed {len(ai_usage_profile)} commits with complete data."

def combine_ai_usage_profiles():
    current_dir = Path.cwd()
    individual_repository_data_dir = current_dir / "profiles" / "ai_usage_profile" / "individual_repository_data"
    combined_ai_usage_profile_path = current_dir / "output_files" / "ai_usage_profile.csv"

    combined_ai_usage_profile = []
    headers_combined = None

    for year_dir in individual_repository_data_dir.iterdir():
        if year_dir.is_dir():
            for repo_dir in year_dir.iterdir():
                if repo_dir.is_dir():
                    ai_usage_profile_path = repo_dir / "ai_usage_profile.csv"
                    if ai_usage_profile_path.exists():
                        with open(ai_usage_profile_path, "r", encoding="utf-8") as f:
                            reader = csv.DictReader(f)
                            rows = list(reader)
                            if headers_combined is None:
                                headers_combined = reader.fieldnames
                            combined_ai_usage_profile.extend(rows)

    priority = ["repository_id", "snapshot_commit", "project_year", "commit_date_str"]

    # keep only existing + preserve remaining order
    ordered_headers = [h for h in priority if h in headers_combined] + [
        h for h in headers_combined if h not in priority
    ]

    with open(combined_ai_usage_profile_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_headers)
        writer.writeheader()

        combined_ai_usage_profile.sort(
            key=lambda x: (
                x.get("project_year"),
                x.get("repository_id"),
                x.get("commit_date_str")
            )
        )

        writer.writerows(combined_ai_usage_profile)
        
if __name__ == "__main__":
    droiddetect_tokenizer, droiddetect_model = load_model()
    
    with open(CODEBASES_TO_ANALYZE_PATH, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    print(f"Loaded {len(rows)} codebases to analyze.")
    compute_ai_usage_profile(rows)
    
    combine_ai_usage_profiles()

