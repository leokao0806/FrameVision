"""Fail when a document contains a 0.xxx number that no results JSON backs.

Usage (from the repo root): python report/check_numbers.py [file ...]
Default file: report/final_report.built.md. Exit code 1 lists every unbacked number with its line.
"""

import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = [
    "results/subtask2_final.json",
    "results/c1_image_text.json",
    "results/c2_vlm_text.json",
    "results/region_coverage.json",
    "baselines/results/subtask1_metrics.json",  # Subtask 1 section
    "baselines/results/leakage_analysis.json",  # C6
]
# Numbers that are settings or quotes, not results. Each needs a reason.
ALLOW = {
    "0.05", "0.25", "0.5", "0.75", "0.95",  # E9 alpha grid and threshold range (Section 6.5)
    "0.4421",  # withdrawn README number, quoted once in Section 7
}
NUMBER = re.compile(r"(?<![\d.])0\.\d+")
COMMENT = re.compile(r"<!--.*?-->", re.S)


def _key(value):
    return f"{round(abs(float(value)), 3):.3f}"


def _values(node, out):
    if isinstance(node, dict):
        for value in node.values():
            _values(value, out)
    elif isinstance(node, list):
        for value in node:
            _values(value, out)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        out.add(_key(node))
    return out


def find_unbacked_numbers(md_text, sources, allow):
    """0.xxx numbers (sign ignored, compared at 3 decimals) found in no source and not in `allow`, in text order."""
    backed = set()
    for source in sources:
        _values(source, backed)
    text = COMMENT.sub("", md_text)
    return [n for n in NUMBER.findall(text) if n not in allow and _key(n) not in backed]


def load_sources(repo=REPO):
    sources = []
    for rel in SOURCES:
        path = os.path.join(repo, rel)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                sources.append(json.load(f))
        else:
            print(f"[check_numbers] source not found, skipped: {rel}", file=sys.stderr)
    return sources


def main(paths):
    sources = load_sources()
    failed = False
    for path in paths or [os.path.join(REPO, "report", "final_report.built.md")]:
        with open(path, encoding="utf-8") as f:
            lines = COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), f.read()).splitlines()
        for number, line in ((n, i) for i, text in enumerate(lines, 1)
                             for n in find_unbacked_numbers(text, sources, ALLOW)):
            print(f"{path}:{line}: unbacked number {number}")
            failed = True
    if failed:
        sys.exit(1)
    print("check_numbers: every 0.xxx number is backed by a results JSON")


if __name__ == "__main__":
    main(sys.argv[1:])
