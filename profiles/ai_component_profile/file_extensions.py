from typing import List

relevant_extensions = [
    ".c", ".cpp", ".cc", ".h", ".hpp", ".hh",
    ".cs", ".csproj"  ,
    ".java", ".kt", ".kts",
    ".py", ".ipynb",
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".go",
    ".rs",
    ".swift", ".m", ".mm",
    ".php",
    ".pl", ".sh", ".bash",
    ".lua",
    ".dart",
]

def get_relevant_extensions() -> List[str]:
    """Return the list of relevant programming language file extensions."""
    return relevant_extensions[:]
