from gitingest import ingest

# -------- CONFIG --------
REPO_PATH = "."

EXCLUDE_PATTERNS = [
    "__pycache__/*",
    ".venv/*",
    "repo.txt",
    "*.log",
]

OUTPUT_FILE = "repo.txt"
# ------------------------


def main():
    summary, tree, content = ingest(
        REPO_PATH,
        exclude_patterns=EXCLUDE_PATTERNS,
    )

    final_output = f"{summary}\n\n{tree}\n\n{content}"

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(final_output)

    print(f"Generated {OUTPUT_FILE}")


if __name__ == "__main__":
    main()