import csv
from pathlib import Path
from typing import List
from profiles.utils.git_utils import clone_repo
from profiles.ai_component_profile.file_extensions import get_relevant_extensions

def extract_file_extensions(github_urls: List[str], output_file: Path, years: List[int]):
    """
    Clones GitHub repositories into ./clones and extracts all unique file extensions.
    Saves them into a text file, one per line.
    """
    extensions = set()
    clone_base_dir = Path(__file__).parent.parent / "system_files" / "codebase_clones"  # <-- Cloning here
    relevant_extensions = get_relevant_extensions()
    
    for url, year in zip(github_urls, years):
        repo_path = clone_repo(url, clone_base_dir, year=year)
        # Walk through all files in the repo
        for file_path in repo_path.rglob("*"):
            if file_path.is_file() and file_path.suffix and file_path.suffix.lower() in relevant_extensions:
                extensions.add(file_path.suffix.lower())  # normalize to lowercase

    # Write unique extensions to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for ext in sorted(extensions):
            f.write(f"{ext}\n")

    print(f"[INFO] Found {len(extensions)} unique file extensions. Saved to {output_file}")

if(__name__ == "__main__"):
    csv_path = Path(__file__).parent.parent.parent / "input_files" / "github_links.csv"
    links = []

    # Read CSV file
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]

    links = [row["Link"].strip() for row in rows if row.get("Link")]
    years = [row["Year"].strip() for row in rows if row.get("Year")]

    output_file = Path(__file__).parent.parent.parent / "output_files" / "file_extensions.txt"
    extract_file_extensions(links, output_file, years)
