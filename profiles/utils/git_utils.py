import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple
from datetime import datetime, timedelta


def sanitize_github_url(url: str) -> str:
    """Remove query parameters or fragments from a GitHub URL and ensure .git suffix."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    sanitized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if not sanitized.endswith(".git"):
        sanitized += ".git"
    return sanitized

def get_default_branch(repo_path_or_url: str) -> str:
    """Detect default branch, works for local path or remote URL."""
    import os

    if Path(repo_path_or_url).exists():  # local repo
        result = subprocess.run(
            ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
            cwd=repo_path_or_url,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().split("/")[-1]

    # remote URL
    try:
        result = subprocess.run(
            ["git", "ls-remote", "--symref", repo_path_or_url, "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        for line in result.stdout.splitlines():
            if line.startswith("ref:"):
                return line.split()[1].replace("refs/heads/", "")
    except subprocess.CalledProcessError:
        pass

    for branch in ("main", "master", "develop"):
        try:
            subprocess.run(
                ["git", "ls-remote", "--exit-code", repo_path_or_url, branch],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            return branch
        except subprocess.CalledProcessError:
            continue

    # fallback clone
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(["git", "clone", "--quiet", repo_path_or_url, tmpdir], check=True)
        result = subprocess.run(
            ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().split("/")[-1]

from datetime import datetime

def clone_repo(github_url: str, clone_dir: Path, year: int) -> Path:
    """
    Ensure full repository clone and checkout state at 31 Dec of given year.
    """
    clone_dir.mkdir(parents=True, exist_ok=True)
    year_dir = clone_dir / str(year)
    year_dir.mkdir(exist_ok=True)
    
    repo_name = github_url.rstrip("/").split("/")[-1].replace(".git", "")
    repo_path = clone_dir / str(year) / repo_name

    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"

    if repo_path.exists():
        subprocess.run(
            ["git", "fetch", "--all", "--tags", "--prune", "--unshallow"],
            cwd=repo_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        subprocess.run(
            [
                "git",
                "clone",
                "--no-single-branch",
                "--filter=blob:none",
                github_url,
                repo_path,
            ],
            check=True,
            env=env
        )

    # Find latest commit before 31 Dec of the given year
    cutoff = f"{year}-12-31 23:59:59"
    result = subprocess.run(
        ["git", "rev-list", "-1", f"--before={cutoff}", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )

    commit_hash = result.stdout.strip()

    if not commit_hash:
        return 
    
    subprocess.run(
        ["git", "checkout", commit_hash],
        cwd=repo_path,
        check=True,
    )
    
    return repo_path

def get_commits_in_range(repo_path: Path, branch: str, start_date: datetime, end_date: datetime) -> List[Tuple[str, datetime]]:
    """Return list of (commit_hash, datetime) for commits in the date range."""
    git_log_cmd = [
        "git", "log",
        "--reverse",
        "--pretty=format:%H|%ct",
        f"--since={start_date.isoformat()}",
        f"--until={end_date.isoformat()}",
        branch
    ]
    result = subprocess.run(
        git_log_cmd,
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True
    )

    commits = []
    for line in result.stdout.splitlines():
        commit_hash, unix_ts = line.split("|")
        commit_time = datetime.utcfromtimestamp(int(unix_ts))
        commits.append((commit_hash, commit_time))
    return commits

def checkout_main_branch(repo_path: Path, default_branch: str):
    """Checkout the main branch to avoid detached HEAD state."""
    subprocess.run(["git", "checkout", default_branch], cwd=repo_path, check=True)

def extract_each_commit(
    github_url: str,
    year: int,
    clones_dir: Path,
) -> Tuple[List[str], Path]:
    """Extract all commits in chronological order (Aug–Dec of a given year)."""

    github_url = sanitize_github_url(github_url)
    repo_path = clone_repo(github_url, clones_dir, year=year)

    start_date = datetime(year, 8, 1)
    end_date = datetime(year, 12, 31, 23, 59, 59)

    default_branch = get_default_branch(repo_path)
    commits = get_commits_in_range(
        repo_path,
        default_branch,
        start_date,
        end_date
    )
    
    # Ensure chronological order (oldest → newest)
    commits.sort(key=lambda x: x[1])

    commit_hashes = [commit_hash for commit_hash, _ in commits]

    return commit_hashes, repo_path