from pathlib import Path
import subprocess
from dateutil import parser
import csv

from profiles.software_quality_profile.extract_all_programming_languages import get_relevant_extensions

def compute_developer_activity_metrics_data_per_commit(
    repo_path: Path,
    commit: str,
    prev_commit: str
):
    """
    Computes MSR-style developer metrics for a repo at each commit snapshot.
    Each commit serves as a reference point; metrics are computed only for
    the changes since the previous snapshot to see project evolution.
    """

    loc_added = 0
    loc_deleted = 0
    num_files_changed = 0
    loc_per_file = []

    relevant_extensions = get_relevant_extensions()

    try:
        # -------------------------
        # LOC added / deleted + files changed
        # -------------------------
        loc_result = subprocess.run(
            ["git", "show", "--numstat", "--format=", commit],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )

        for line in loc_result.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) != 3:
                continue

            added, deleted, filename = parts
            ext = Path(filename).suffix.lower()

            if ext not in relevant_extensions:
                continue

            try:
                a = int(added)
                d = int(deleted)
            except ValueError:
                continue

            loc_added += a
            loc_deleted += d
            num_files_changed += 1
            loc_per_file.append(a + d)

        code_churn = loc_added + loc_deleted
        avg_loc_per_file = (
            sum(loc_per_file) / num_files_changed if num_files_changed > 0 else 0
        )

        # -------------------------
        # Snapshot commit date
        # -------------------------
        date_result = subprocess.run(
            ["git", "show", "-s", "--format=%cI", commit],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        commit_date_str = date_result.stdout.strip()

        # -------------------------
        # Commit author
        # -------------------------
        author_result = subprocess.run(
            ["git", "show", "-s", "--format=%an", commit],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        author_id = author_result.stdout.strip()

        # -------------------------
        # Author prior commits (before snapshot)
        # -------------------------
        try:
            prior_commits_result = subprocess.run(
                ["git", "rev-list", "--count", f"{commit}^", "--author", author_id],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            author_prior_commits = int(prior_commits_result.stdout.strip())
        except subprocess.CalledProcessError:
            author_prior_commits = 0
            
        # -------------------------
        # Days since author last commit
        # -------------------------
        last_author_commit_result = subprocess.run(
            [
                "git", "log", "-1",
                "--author", author_id,
                "--format=%cI",
                f"{commit}^"
            ],
            cwd=repo_path,
            capture_output=True,
            text=True
        )

        if last_author_commit_result.stdout.strip():
            last_date = parser.isoparse(
                last_author_commit_result.stdout.strip()
            )
            current_date = parser.isoparse(commit_date_str)
            days_since_author_last_commit = (
                current_date - last_date
            ).days
        else:
            days_since_author_last_commit = None

        # -------------------------
        # Active developers since previous snapshot
        # -------------------------
        if prev_commit:
            devs_result = subprocess.run(
                [
                    "git", "log",
                    f"{prev_commit}..{commit}",
                    "--format=%an"
                ],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            active_devs = set(devs_result.stdout.splitlines())
            n_active_devs_since_last_snapshot = len(active_devs)
        else:
            n_active_devs_since_last_snapshot = 1

        # -------------------------
        # Final row
        # -------------------------
        row = [
            commit,
            commit_date_str,
            author_id,
            author_prior_commits,
            days_since_author_last_commit,
            loc_added,
            loc_deleted,
            code_churn,
            avg_loc_per_file,
            n_active_devs_since_last_snapshot,
            num_files_changed,
        ]

        return row
    
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to compute developer metrics for commit {commit}: {e}")
        return None

def compute_developer_activity_metrics_data(commits, repository_id, repo_path, storage_dir):
    path = storage_dir / "developer_metrics.csv"
    if path.exists():
        print(f"[INFO] Developer metrics already computed for {repository_id}, loading from {path}")
        # Load and return existing data
        rows = {}
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                commit = row["commit"]
                rows[commit] = row
            headers_developer_metrics = reader.fieldnames
            
        return rows, headers_developer_metrics
    
    rows = {}

    # Sort commits chronologically (optional if already sorted)
    commits = list(commits)

    for i in range(len(commits)):
        commit = commits[i]
        prev_commit = commits[i - 1] if i > 0 else None
        
        if( not prev_commit ):
            continue  
        
        print(f"[DEBUG] Calculating developer metrics for {commit} ({ i + 1 }/{ len(commits) }) ({repository_id})")
                    
        row = compute_developer_activity_metrics_data_per_commit(
            repo_path,
            commit,
            prev_commit
        )
        
        if row is not None:
            rows[commit] = row
    
    headers_developer_metrics = [
        "commit",
        "commit_date_str",
        "author_id",
        "author_prior_commits",
        "days_since_author_last_commit",
        "loc_added",
        "loc_deleted",
        "code_churn",
        "avg_loc_per_file",
        "n_active_devs_since_last_snapshot",
        "num_files_changed",
    ]
     
    print(f"[INFO] Developer metrics finished for repo: {repository_id}")
    
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers_developer_metrics)
        for commit in commits:
            if commit in rows:
                writer.writerow(rows[commit])
                
    return rows, headers_developer_metrics
