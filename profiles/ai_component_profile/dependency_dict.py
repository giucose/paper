
from pathlib import Path

DEPENDENCIES_JSON = Path(__file__).parent / "codebases/dependencies.json"
OUTPUT_FILE = Path(__file__).parent / "codebases/all_dependencies.txt"

with open(DEPENDENCIES_JSON, "r") as f:
    import json
    DEPENDENCY_DICT = json.load(f)

dependencies = set()

for key in DEPENDENCY_DICT:
    for codebase in DEPENDENCY_DICT[key]:
        for dep in DEPENDENCY_DICT[key][codebase]:
            dependencies.add(dep)
            
with open(OUTPUT_FILE, "w") as f:
    for dep in sorted(dependencies):
        f.write(f"{dep}\n")