import subprocess
from pathlib import Path
import csv
import os
import shutil
from typing import Optional
import requests

from profiles.utils.git_utils import extract_each_commit
from profiles.utils.repository_id import get_repository_id

CODEBASES_TO_ANALYZE_PATH = Path(__file__).parent.parent.parent / "input_files" / "github_links.csv"
CLONES_DIR = Path(__file__).parent.parent.parent / "system_files" / "codebase_clones"

# Create and insert your SonarQube token here
SONARQUBE_TOKEN='squ_...'

# Enter the URL of your SonarQube server if different from default
SONARQUBE_SERVER_URL = "http://localhost:9000"

OUTPUT_CSV = Path(__file__).parent.parent.parent.parent / "output_files" / "software_quality_profile.csv"

ALL_METRICS = ",".join([
    "bugs","reliability_rating","reliability_remediation_effort",
    "vulnerabilities","security_rating","security_remediation_effort",
    "security_hotspots","security_review_rating",
    "code_smells","sqale_index","sqale_debt_ratio","sqale_rating",
    "coverage","lines_to_cover","uncovered_lines","tests",
    "duplicated_lines_density","duplicated_lines","duplicated_blocks","duplicated_files",
    "ncloc","lines","statements","functions","classes","files",
    "comment_lines","comment_lines_density",
    "complexity","cognitive_complexity",
    "violations","open_issues"
])

ADDITIONAL_ISSUE_METRICS = ",".join([
    "n_blocker","n_critical","n_major","n_minor","n_info",
    "n_security","n_reliability","n_maintainability",
    "n_consistent","n_intentional","n_adaptable","n_responsible",
    "total_issues_per_kloc","critical_plus_blocker_ratio",
    "major_ratio","minor_ratio","info_ratio","security_ratio","reliability_ratio","maintainability_ratio",
    "languages"
])

def get_processed_repos() -> set[str]: 
    """
    Return all project keys from SonarQube using proper paging.
    """
    BASE_URL = f"{SONARQUBE_SERVER_URL}/api/projects/search_my_scannable_projects"

    results: set[str] = set()

    response = requests.get(
        BASE_URL,
        auth=(SONARQUBE_TOKEN, ""),
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    # collect keys
    for c in data.get("projects", []):
        key = c.get("key")
        if key:
            results.add(key)

    print(f"[INFO] Retrieved {len(results)} processed repositories from SonarQube")
    return results

def compile_java_project(repo_path: Path) -> Optional[Path]:
    """
    Compile Java project if it contains .java files.
    Returns path to compiled classes or None if no Java files found.
    """
   
    # Plain Java
    java_files = list(repo_path.rglob("*.java"))
    if not java_files:
        return None

    bin_dir = repo_path / "bin"
    bin_dir.mkdir(exist_ok=True)

    javac_base = ["javac", "-d", str(bin_dir), "-Xmaxerrs", "500", "-Xmaxwarns", "500"]
    classpath_jars = list((repo_path / "lib").glob("*.jar"))
    if classpath_jars:
        javac_base += ["-cp", os.pathsep.join(str(j) for j in classpath_jars)]

    chunk_size = 50
    for i in range(0, len(java_files), chunk_size):
        chunk = java_files[i:i+chunk_size]
        result = subprocess.run(
            javac_base + [str(f) for f in chunk],
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            print(f"[ERROR] javac failed on chunk {i//chunk_size + 1}")
            print(result.stderr)
            
    return bin_dir

def run_sonar_scanner(repo_path: Path, project_key: str, java_binaries: Optional[Path] = None):
    print(f"[INFO] Running SonarScanner for {project_key}")
    
    cmd = [
        "sonar-scanner",
        f"-Dsonar.projectKey={project_key}",
        "-Dsonar.sources=.",
        f"-Dsonar.host.url={SONARQUBE_SERVER_URL}",
        f"-Dsonar.login={SONARQUBE_TOKEN}",
    ]

    if java_binaries and java_binaries.exists():
        cmd.append(f"-Dsonar.java.binaries={java_binaries}")

    result = subprocess.run(
        cmd,
        cwd=repo_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    if result.returncode != 0:
        print(f"[ERROR] SonarScanner failed: {project_key}")
        print(result.stdout)
        return

def export_all_metrics():
    if OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()

    components = get_processed_repos()
    num_components = len(components)

    # Build deterministic fieldnames
    base_fields = ["repository_id", "snapshot_commit", "date"]
    derived_fields = [
        "debt_per_kloc",
        "complexity_per_kloc",
        "quality_index",
    ]

    # Remove "new" metrics since you skip them anyway
    metric_fields = sorted(
        m for m in ALL_METRICS.split(",")
        if not m.startswith("new")
    )
    
    issues_fields = ADDITIONAL_ISSUE_METRICS.split(",")

    fieldnames = base_fields + metric_fields + issues_fields +derived_fields

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, component in enumerate(components):
            print(f"[INFO] ({i+1}/{num_components}) Exporting metrics for {component}")
            repository_id, date_str, snapshot_commit = component.rsplit(".", 2)
            date_str = date_str.replace("TZp", "+")

            row = {
                "repository_id": repository_id,
                "snapshot_commit": snapshot_commit,
                "date": date_str
            }

            measures_response = requests.get(
                f"{SONARQUBE_SERVER_URL}/api/measures/component",
                params={"component": component, "metricKeys": ALL_METRICS},
                auth=(SONARQUBE_TOKEN, "")
            )
            
            measures_response.raise_for_status()
            measures = measures_response.json()["component"]["measures"]

            for m in measures:
                metric = m["metric"]
                value = m.get("value")

                if metric.startswith("new"):
                    continue

                if value is not None:
                    try:
                        value = float(value) if "." in value else int(value)
                    except ValueError:
                        pass

                row[metric] = value

            ncloc = row.get("ncloc", 0)

            if ncloc == 0:
                row["debt_per_kloc"] = None
                row["complexity_per_kloc"] = None
            else:
                row["debt_per_kloc"] = (
                    row.get("sqale_index", 0) / (ncloc / 1000)
                )

                row["complexity_per_kloc"] = (
                    row.get("complexity", 0) / (ncloc / 1000)
                )

            row["quality_index"] = (
                (
                    row.get("reliability_rating", 5)
                    + row.get("security_rating", 5)
                    + row.get("sqale_rating", 5)
                    + (6 - row.get("coverage", 0) / 20)
                ) / 4
            )
            
            get_issues_metrics(row, component)  
            
            if row.get("lines") is not None and row.get("ncloc") is not None:
                writer.writerow(row)
                f.flush()  # ensures data is written immediately
    
def get_issues_metrics(row, component: str) -> dict[str, Optional[float]]:
    issues_response = requests.get(
        f"{SONARQUBE_SERVER_URL}/api/issues/search?components={component}",
        auth=(SONARQUBE_TOKEN, ""),
        params={"facets": "impactSoftwareQualities,severities,types,impactSeverities,codeVariants,cleanCodeAttributeCategories,languages"}
    )
    facets = issues_response.json()["facets"]
    
    def facet_count(property_name, value):
        for facet in facets:
            if facet.get("property") == property_name:
                return sum(
                    v.get("count", 0)
                    for v in facet.get("values", [])
                    if v.get("val") == value
                )
        return 0

    n_critical = facet_count("severities", "CRITICAL")
    n_major = facet_count("severities", "MAJOR")
    n_minor = facet_count("severities", "MINOR")
    n_info = facet_count("severities", "INFO")
    n_blocker = facet_count("severities", "BLOCKER")

    n_security = facet_count("impactSoftwareQualities", "SECURITY")
    n_reliability = facet_count("impactSoftwareQualities", "RELIABILITY")
    n_maintainability = facet_count("impactSoftwareQualities", "MAINTAINABILITY")

    n_consistent = facet_count("cleanCodeAttributeCategories", "CONSISTENT")
    n_intentional = facet_count("cleanCodeAttributeCategories", "INTENTIONAL")
    n_adaptable = facet_count("cleanCodeAttributeCategories", "ADAPTABLE")
    n_responsible = facet_count("cleanCodeAttributeCategories", "RESPONSIBLE")
    
    for facet in facets:
        if facet.get("property") == "languages":
            languages = facet["values"]
    
    total_issues_count = issues_response.json().get("total", 0)

    ncloc = row.get("ncloc", 0)

    row.update({
        "n_blocker": n_blocker,
        "n_critical": n_critical,
        "n_major": n_major,
        "n_minor": n_minor,
        "n_info": n_info,
        "n_security": n_security,
        "n_reliability": n_reliability,
        "n_maintainability": n_maintainability,
        "n_consistent": n_consistent,
        "n_intentional": n_intentional,
        "n_adaptable": n_adaptable,
        "n_responsible": n_responsible,
        "total_issues_per_kloc": total_issues_count / (ncloc / 1000) if ncloc > 0 else 0,
        "critical_plus_blocker_ratio": (n_critical + n_blocker) / total_issues_count if total_issues_count else 0,
        "major_ratio": n_major / total_issues_count if total_issues_count else 0,
        "minor_ratio": n_minor / total_issues_count if total_issues_count else 0,
        "info_ratio": n_info / total_issues_count if total_issues_count else 0,
        "security_ratio": n_security / total_issues_count if total_issues_count else 0,
        "reliability_ratio": n_reliability / total_issues_count if total_issues_count else 0,
        "maintainability_ratio": n_maintainability / total_issues_count if total_issues_count else 0,
        "languages": languages
    })

    return row
    
def sort_csv_by_repository_id_and_date(input_csv: Path, output_csv: Path):
    with input_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    rows.sort(key=lambda r: (r["repository_id"], r["date"]))

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
def main():
    with open(CODEBASES_TO_ANALYZE_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"[INFO] Cloning repositories and running SonarScanner... ({len(rows)} repos)")
    processed_repos = get_processed_repos()
    
    for row in rows:
        link = row["Link"].strip()
        year = row["Year"].strip()
        commits, repo_path = extract_each_commit(link, int(year), CLONES_DIR)

        print(f"[INFO] Processing {link} ({len(commits)} commits to analyze)")
        repo_id = get_repository_id(year, link)
        
        for commit_hash in commits:
            date_result = subprocess.run(
                ["git", "show", "-s", "--format=%cI", commit_hash],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            commit_date_str = date_result.stdout.strip().replace("+", "TZp")
            project_key = f"{repo_id}.{commit_date_str}.{commit_hash}"
            
            if project_key in processed_repos:
                print(f"[INFO] Project {project_key} already exists in SonarQube, skipping analysis.")
                continue
            
            # Checkout commit in detached HEAD
            # 1. Reset working tree (discard local edits)
            subprocess.run(["git", "reset", "--hard"], cwd=repo_path, check=True)

            # 2. Remove untracked files/folders
            subprocess.run(["git", "clean", "-fd"], cwd=repo_path, check=True)

            # 3. Checkout commit
            subprocess.run(["git", "checkout", "--quiet", commit_hash], cwd=repo_path, check=True)
            
            print(f"[INFO] Analyzing commit {commit_hash} of {repo_id}")

            # Compile Java if any
            java_binaries = compile_java_project(repo_path)
            run_sonar_scanner(repo_path, project_key=project_key, java_binaries=java_binaries)

            # Cleanup compiled classes if desired
            if java_binaries and java_binaries.exists() and java_binaries.name in ("bin", "target", "build"):
                shutil.rmtree(java_binaries)
            
    print("[INFO] All repositories processed")

def delete_all_projects_in_sonarqube():
    """
    Deletes all projects in SonarQube instance.
    USE WITH CAUTION.
    """
    BASE_URL = f"{SONARQUBE_SERVER_URL}/api/projects/delete"
    processed_repos = get_processed_repos()

    for project_key in processed_repos:
        print(f"[INFO] Deleting project {project_key} from SonarQube")
        response = requests.post(
            BASE_URL,
            auth=(SONARQUBE_TOKEN, ""),
            data={"project": project_key}
        )
        if response.status_code == 204:
            print(f"[INFO] Successfully deleted {project_key}")
        else:
            print(f"[ERROR] Failed to delete {project_key}: {response.status_code} - {response.text}")
           
def delete_prefix_projects_in_sonarqube(prefix: str):
    """
    Deletes projects in SonarQube instance that start with the given prefix.
    USE WITH CAUTION.
    """
    BASE_URL = f"{SONARQUBE_SERVER_URL}/api/projects/delete"
    processed_repos = get_processed_repos()

    for project_key in processed_repos:
        if project_key.startswith(prefix):
            print(f"[INFO] Deleting project {project_key} from SonarQube")
            response = requests.post(
                BASE_URL,
                auth=(SONARQUBE_TOKEN, ""),
                data={"project": project_key}
            )
            if response.status_code == 204:
                print(f"[INFO] Successfully deleted {project_key}")
            else:
                print(f"[ERROR] Failed to delete {project_key}: {response.status_code} - {response.text}")
                 
if __name__ == "__main__":
    # delete_all_projects_in_sonarqube()
    # delete_prefix_projects_in_sonarqube("2020_Kundestyrt-Gruppe-10_Purefolio-backend")
    main()
    export_all_metrics()
    sort_csv_by_repository_id_and_date(OUTPUT_CSV, OUTPUT_CSV)
    
# ID in SONARQUBE is in format: {repo_id}_{commit_index}:{commit_hash}