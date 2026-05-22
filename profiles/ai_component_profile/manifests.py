from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import re
import tomli as tomllib
from collections import defaultdict
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from profiles.utils.git_utils import clone_repo, sanitize_github_url

DEPENDENCIES = defaultdict(lambda: defaultdict(set))
OUTPUT_DIR = Path(__file__).parent.parent.parent / "system_files" / "dependency_files"
CODEBASES_CSV = Path(__file__).parent.parent.parent / "input_files" / "github_links.csv"
CLONES_DIR = Path(__file__).parent.parent.parent / "system_files" / "codebase_clones"

# ---------- HELPERS ----------

def walk_repo(root):
    for dirpath, _, filenames in os.walk(root):
        if ".git" in dirpath:
            continue
        for f in filenames:
            yield os.path.join(dirpath, f)

# ---------- PYTHON ----------

def parse_requirements(path, repo_name):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                DEPENDENCIES["python"][repo_name].add(line.split("==")[0])

def parse_pyproject(path, repo_name):
    with open(path, "rb") as f:
        data = tomllib.load(f)

    for section in (
        data.get("project", {}).get("dependencies", []),
        data.get("tool", {}).get("poetry", {}).get("dependencies", {}).keys()
    ):
        for dep in section:
            DEPENDENCIES["python"][repo_name].add(dep)

# ---------- JAVA ----------

def parse_pom_xml(path, repo_name):
    tree = ET.parse(path)
    root = tree.getroot()

    # Handle XML namespaces
    ns = {"m": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}

    for dep in root.findall(".//m:dependency", ns) + root.findall(".//dependency"):
        gid = dep.find("m:groupId", ns) or dep.find("groupId")
        aid = dep.find("m:artifactId", ns) or dep.find("artifactId")
        if gid is not None and aid is not None:
            DEPENDENCIES["java"][repo_name].add(f"{gid.text}:{aid.text}")

GRADLE_DEP_RE = re.compile(
    r"(implementation|api|compileOnly|testImplementation)\s*\(?['\"]([^:'\"]+):([^:'\"]+)"
)

def parse_gradle_file(path, repo_name):
    with open(path, errors="ignore") as f:
        for match in GRADLE_DEP_RE.findall(f.read()):
            group, artifact = match[1], match[2]
            DEPENDENCIES["java"][repo_name].add(f"{group}:{artifact}")

# ---------- C# ----------

def parse_csproj(path, repo_name):
    tree = ET.parse(path)
    root = tree.getroot()

    ns = {"m": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}

    for pr in root.findall(".//m:PackageReference", ns) + root.findall(".//PackageReference"):
        name = pr.attrib.get("Include") or pr.attrib.get("Update")
        if name:
            DEPENDENCIES["csharp"][repo_name].add(name)

def parse_packages_config(path, repo_name):
    tree = ET.parse(path)
    root = tree.getroot()

    for pkg in root.findall(".//package"):
        name = pkg.attrib.get("id")
        if name:
            DEPENDENCIES["csharp"][repo_name].add(name)

def parse_directory_packages_props(path, repo_name):
    tree = ET.parse(path)
    root = tree.getroot()

    ns = {"m": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}

    for pv in root.findall(".//m:PackageVersion", ns) + root.findall(".//PackageVersion"):
        name = pv.attrib.get("Include")
        if name:
            DEPENDENCIES["csharp"][repo_name].add(name)
      
# ---------- JAVASCRIPT ----------

def parse_package_json(path, repo_name):
    with open(path) as f:
        data = json.load(f)

    for section in ("dependencies", "devDependencies"):
        for dep in data.get(section, {}):
            DEPENDENCIES["javascript"][repo_name].add(dep)

# ---------- RUST ----------

def parse_cargo_toml(path, repo_name):
    with open(path, "rb") as f:
        data = tomllib.load(f)

    for dep in data.get("dependencies", {}):
        DEPENDENCIES["rust"][repo_name].add(dep)

# ---------- STATIC IMPORT FALLBACK ----------

PY_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([a-zA-Z0-9_]+)", re.M)
JS_IMPORT_RE = re.compile(r"(?:require\(|from\s+)[\"']([^\"']+)")
JAVA_IMPORT_RE = re.compile(r"^\s*import\s+([a-zA-Z0-9_.]+)", re.M)
CSHARP_USING_RE = re.compile(r"^\s*using\s+([A-Za-z0-9_.]+);", re.M)

def parse_python_imports(path, repo_name):
    with open(path, errors="ignore") as f:
        for match in PY_IMPORT_RE.findall(f.read()):
            DEPENDENCIES["python"][repo_name].add(match)

def parse_js_imports(path, repo_name):
    with open(path, errors="ignore") as f:
        for match in JS_IMPORT_RE.findall(f.read()):
            if path.endswith((".js", ".jsx")):
                DEPENDENCIES["javascript"][repo_name].add(match.split("/")[0])
            elif path.endswith((".ts", ".tsx")):
                DEPENDENCIES["typescript"][repo_name].add(match.split("/")[0])

def parse_java_imports(path, repo_name):
    with open(path, errors="ignore") as f:
        for imp in JAVA_IMPORT_RE.findall(f.read()):
            DEPENDENCIES["java"][repo_name].add(imp.split(".")[0])

def parse_csharp_usings(path, repo_name):
    with open(path, errors="ignore") as f:
        for u in CSHARP_USING_RE.findall(f.read()):
            if not u.startswith(("System", "Microsoft")):
                DEPENDENCIES["csharp"][repo_name].add(u.split(".")[0])
                
# ---------- MAIN ----------

def extract_dependencies(repo_path, repo_name):
    found = False
    for path in walk_repo(repo_path):
        filename = os.path.basename(path)

        try:
            if filename == "requirements.txt":
                parse_requirements(path, repo_name)
                found = True
            elif filename == "pyproject.toml":
                parse_pyproject(path, repo_name)
                found = True

            elif filename == "package.json":
                parse_package_json(path, repo_name)
                found = True
                
            elif filename == "Cargo.toml":
                parse_cargo_toml(path, repo_name)
                found = True

            elif path.endswith(".py"):
                parse_python_imports(path, repo_name)
                found = True

            elif filename == "pom.xml":
                parse_pom_xml(path, repo_name)
                found = True
            
            elif filename in ("build.gradle", "build.gradle.kts"):
                parse_gradle_file(path, repo_name)
                found = True
            
            elif path.endswith(".java"):
                parse_java_imports(path, repo_name)
                found = True

            elif path.endswith((".js", ".ts", ".jsx", ".tsx")):
                parse_js_imports(path, repo_name)
                found = True
                
            elif filename.endswith(".csproj"):
                parse_csproj(path, repo_name)
                found = True
            
            elif filename == "packages.config":
                parse_packages_config(path, repo_name)
                found = True
                
            elif filename == "Directory.Packages.props":
                parse_directory_packages_props(path, repo_name)
                found = True
                
            elif path.endswith(".cs"):
                parse_csharp_usings(path, repo_name)
                found = True
                
        except Exception:
            pass  # be resilient
        
    if not found:
        DEPENDENCIES["unknown"][repo_name].add("NOT FOUND")
                 
    return {k: {repo: sorted(deps) for repo, deps in v.items()} for k, v in DEPENDENCIES.items()}

def main():
    with open(CODEBASES_CSV, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    paths_and_year = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        print("[INFO] Cloning repositories and extracting dependencies... (" + str(len(rows)) + " repos)")
        
        for i in range(len(rows)):
            row = rows[i]
            link = row["Link"].strip()
            year = row["Year"].strip()
            github_url = sanitize_github_url(link)
            repo_path = clone_repo(github_url, clone_dir=CLONES_DIR, year=int(year))
            repo_year = row["Year"].strip()

            paths_and_year.append((repo_path, repo_year))
            repo_id =  repo_year  + "_" + link.replace("https://github.com/", "").replace("/", "_")
            executor.submit(
                extract_dependencies,
                repo_path,
                repo_id
            )
            
            print(f"[INFO] ({i+1}/{len(rows)}) Extracted dependencies for {repo_id}")
    
    path = OUTPUT_DIR / "dependencies.json"
    
    if(not path.parent.exists()):
        path.parent.mkdir(parents=True, exist_ok=True)
        
    with open(OUTPUT_DIR / "dependencies.json", "w") as f:
        json.dump({k: {repo: list(deps) for repo, deps in v.items()} for k, v in DEPENDENCIES.items()}, f, indent=2)

    num_programming_languages = len(DEPENDENCIES)
    programming_languages = [key for key in DEPENDENCIES.keys()]
    num_codebases = sum(len(v) for v in DEPENDENCIES.values())
    ranked = sorted(
        [(lang, len(DEPENDENCIES[lang])) for lang in programming_languages],
        key=lambda x: x[1],
        reverse=True
    )
    ranked_dependencies = {lang: count for lang, count in ranked}
    
    print(f"[INFO] Extraction complete. Found {num_programming_languages} programming languages across {num_codebases} codebases.")
    print(f"[INFO] Programming languages ranked by number of codebases:")
    for lang, count in ranked:
        print(f"  - {lang}: {count} codebases")
    
    print(ranked_dependencies)
    
if __name__ == "__main__":
    main()
