import csv
from pathlib import Path
import shutil
from typing import List
from dateutil import parser
import subprocess
import hashlib
from collections import defaultdict

from profiles.utils.file_extensions import get_relevant_extensions

def compute_ai_usage_profile(
    repo_id: str,
    repo_path: Path,
    commits: List[str],
):
    rows = []

    # Sort commits chronologically (optional if already sorted)
    commits = list(commits)

    for i in range(len(commits)):
        commit = commits[i]
        prev_commit = commits[i - 1] if i > 0 else None
        
        if( not prev_commit ):
            continue  
        
        print(f"[DEBUG] Calculating developer metrics for {commit} ({ i + 1 }/{ len(commits) }) ({repo_id})")
                    
        row = compute_developer_activity_metrics_data(
            repo_id,
            repo_path,
            commit,
            prev_commit
        )
        
        rows.append(row)

    print(f"[INFO] Developer metrics finished for repo: {repo_id}")
    
    return rows
    
def get_changed_files(repo_path: Path, commit: str, relevant_extensions: set[str]):
    result = subprocess.run(
        ["git", "show", "--name-only", "--format=", commit],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True
    )
    files = []
    for line in result.stdout.splitlines():
        ext = Path(line).suffix.lower()
        if ext in relevant_extensions:
            files.append(line)
    return files

def compute_developer_activity_metrics_data(
    repo_id: str,
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
    files_changed = 0
    loc_per_file = []

    relevant_extensions = get_relevant_extensions()

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
        files_changed += 1
        loc_per_file.append(a + d)

    code_churn = loc_added + loc_deleted
    avg_loc_per_file = (
        sum(loc_per_file) / files_changed if files_changed > 0 else 0
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
    prior_commits_result = subprocess.run(
        ["git", "rev-list", "--count", f"{commit}^", "--author", author_id],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True
    )
    author_prior_commits = int(prior_commits_result.stdout.strip())

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
        repo_id,
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
        files_changed,
    ]
    
    return row

def compute_diff_and_store(commits: List[str], repo_path: Path, repo_id: str, year: int, reuse_diffs: bool = True) -> Path:
    # Only process source/text files
    relevant_extensions= get_relevant_extensions()

    base_dir = Path("temp_diffs") / str(year) / repo_id

    # Remove old directory if it exists
    if base_dir.exists() and base_dir.is_dir():
        shutil.rmtree(base_dir)

    # Recreate fresh directory
    base_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(1, len(commits)):
        prev_commit = commits[i - 1]
        curr_commit = commits[i]

        if((base_dir / curr_commit / "diff").exists() and (base_dir / curr_commit / "flat").exists() and reuse_diffs):
            print(f"[INFO] Reusing existing diffs for {curr_commit} ({repo_id})")
            continue

        try:
            files_result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=ACM", prev_commit, curr_commit],
                cwd=repo_path,
                capture_output=True,  # bytes
                check=True,
                text=True,
            )
            changed_files = files_result.stdout.splitlines()
            
        except subprocess.CalledProcessError as e:
            print(f"[WARNING] Git diff --name-only failed for {prev_commit} -> {curr_commit}: {e}")
            continue
        
        if(len(changed_files) == 0):
            print(f"[INFO] No changed files in {prev_commit} -> {curr_commit}, skipping.")
            continue
        
        commit_dir = base_dir / curr_commit / "diff"
        commit_dir.mkdir(parents=True, exist_ok=True)

        for file_path in changed_files:
            extenstion = "." + file_path.split(".")[-1].lower()
            
            if extenstion not in relevant_extensions:
                continue  # skip binaries / non-source files
            
            show_result = subprocess.run(
                ["git", "show", f"{curr_commit}:{file_path}"],
                cwd=repo_path,
                capture_output=True,  # bytes
                check=True,
            )
            raw = show_result.stdout
            try:
                content = raw.decode("utf-8")  # decode only text/source files
            except UnicodeDecodeError:
                print(f"[WARNING] Could not decode {file_path}, skipping binary/undecodable content.")
                continue
            
            target_path = commit_dir / file_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding="utf-8")
            
        flatten_commit_diff(commit_dir, base_dir / curr_commit / "flat")

        is_empty = not any(commit_dir.rglob("*"))
        if( is_empty ):
            print(f"[INFO] No relevant source/text file changes in {prev_commit} -> {curr_commit}, skipping.")
            shutil.rmtree(base_dir / curr_commit)
            continue
        else:
            print(f"[INFO] Created and stored diffs for {curr_commit} ({repo_id})")

    return base_dir

def flatten_commit_diff(commit_dir: Path, flat_base_dir: Path):
    flat_base_dir.mkdir(parents=True, exist_ok=True)

    file_paths_to_flatten = commit_dir.rglob("*")
    
    for file_path in file_paths_to_flatten:
        if file_path.is_file():
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # skip binaries or undecodable files
                continue
            
            # Compute SHA1 hash of the content
            file_hash = hashlib.sha1(content.encode("utf-8")).hexdigest()
            file_name = file_path.name
            file_id = file_hash + "_" + file_name
            
            flat_file_path = flat_base_dir / file_id

            # Write flattened file
            flat_file_path.write_text(content, encoding="utf-8")