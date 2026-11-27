"""
Check that two copies of the SemEval-2027 Task 9 release are the same data.
CS 5134/6034 NLP Fall 2026 - Team FrameVision

Standard library only, so it runs in any Python 3.8+ environment.

Usage:
  python data_manifest.py write   --root DIR [--labels FILE] --out MANIFEST.json
  python data_manifest.py compare REFERENCE OTHER [--ref-labels FILE] [--other-labels FILE]
                                  [--ref-name NAME] [--other-name NAME] [--save-other OUT.json]

REFERENCE and OTHER are each a manifest .json or a release root directory, the folder that
holds initial-training-data/ and training-second-batch-oct/. A directory's labels default to
<root>/labels.txt. Exit code: 0 when every file the Subtask 2 pipelines read matches,
1 when one does not, 2 on bad arguments.
"""

import argparse
import hashlib
import json
import os
import sys

FORMAT = 1
# Files under these folders are read by the Subtask 2 training pipeline (src/dataset.py).
CRITICAL_PREFIXES = ("initial-training-data/subtask2/", "training-second-batch-oct/EN/")
TEXT_EXTENSIONS = {".txt", ".html", ".htm", ".tsv", ".csv", ".json", ".md"}
JUNK_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
MAX_LISTED = 20


def _is_junk(rel):
    parts = rel.split("/")
    return "__MACOSX" in parts or parts[-1] in JUNK_NAMES or parts[-1].startswith("._")


def _normalized_text(data):
    """Bytes with the BOM, CRLF / CR line endings and trailing whitespace removed."""
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n").rstrip()


def _hash_file(path, is_text):
    if is_text:
        with open(path, "rb") as f:
            data = f.read()
        return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
                "norm": hashlib.sha256(_normalized_text(data)).hexdigest()}
    digest, size = hashlib.sha256(), 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "size": size}


def read_labels(path):
    with open(path, "rb") as f:
        text = _normalized_text(f.read()).decode("utf-8")
    return [line.strip() for line in text.split("\n") if line.strip()]


def build_manifest(root, labels_path=None):
    """Hashes every file under root, except OS junk and the root-level labels.txt."""
    files = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            rel = os.path.relpath(os.path.join(dirpath, name), root).replace(os.sep, "/")
            if _is_junk(rel) or rel == "labels.txt":
                continue
            ext = os.path.splitext(name)[1].lower()
            files[rel] = _hash_file(os.path.join(dirpath, name), ext in TEXT_EXTENSIONS)
    if labels_path is None and os.path.isfile(os.path.join(root, "labels.txt")):
        labels_path = os.path.join(root, "labels.txt")
    return {"format": FORMAT, "root_name": os.path.basename(os.path.normpath(root)),
            "labels": read_labels(labels_path) if labels_path else None,
            "file_count": len(files), "files": files}


def load_side(arg, labels_path=None):
    if os.path.isdir(arg):
        return build_manifest(arg, labels_path)
    with open(arg, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("format") != FORMAT or "files" not in manifest:
        raise ValueError(f"{arg} is not a data_manifest.py manifest")
    if labels_path:
        manifest["labels"] = read_labels(labels_path)
    return manifest


def _compare_labels(ref, other):
    if ref is None or other is None:
        return {"status": "missing", "only_ref": [], "only_other": []}
    if ref == other:
        return {"status": "same", "only_ref": [], "only_other": []}
    only_ref = [x for x in ref if x not in other]
    only_other = [x for x in other if x not in ref]
    return {"status": "different" if only_ref or only_other else "order",
            "only_ref": only_ref, "only_other": only_other}


def _is_critical(rel):
    return rel.startswith(CRITICAL_PREFIXES)


def compare_manifests(ref, other):
    rf, of = ref["files"], other["files"]
    identical, text_only, content = [], [], []
    for rel in sorted(set(rf) & set(of)):
        a, b = rf[rel], of[rel]
        if a["sha256"] == b["sha256"]:
            identical.append(rel)
        elif a.get("norm") and a.get("norm") == b.get("norm"):
            text_only.append(rel)
        else:
            content.append(rel)
    missing = sorted(set(rf) - set(of))
    extra = sorted(set(of) - set(rf))
    extra_by_hash = {}
    for rel in extra:
        extra_by_hash.setdefault(of[rel]["sha256"], []).append(rel)
    moved = []
    for rel in missing:
        candidates = extra_by_hash.get(rf[rel]["sha256"])
        if candidates:
            moved.append([rel, candidates.pop(0)])
    moved_from = {pair[0] for pair in moved}
    moved_to = {pair[1] for pair in moved}
    missing = [rel for rel in missing if rel not in moved_from]
    extra = [rel for rel in extra if rel not in moved_to]

    labels = _compare_labels(ref.get("labels"), other.get("labels"))
    critical_failures = sorted(rel for rel in missing + content + sorted(moved_from) if _is_critical(rel))
    if labels["status"] == "different":
        critical_failures.append("labels.txt (label names differ)")

    warnings = []
    if text_only:
        warnings.append(f"{len(text_only)} text file(s) differ only in line endings, BOM or trailing whitespace")
    noncritical = [rel for rel in missing + content if not _is_critical(rel)]
    if noncritical:
        warnings.append(f"{len(noncritical)} file(s) outside the Subtask 2 pipeline folders are missing or differ")
    if any(_is_critical(rel) for rel in extra):
        warnings.append("OTHER has extra files inside the pipeline folders (unused unless an annotation lists them)")
    if labels["status"] == "order":
        warnings.append("labels.txt has the same names in another order (class indices differ between copies)")
    if labels["status"] == "missing":
        warnings.append("labels.txt is missing on one side, so label names were not compared")

    return {"passed": not critical_failures, "critical_failures": critical_failures,
            "identical": identical, "text_only": text_only, "content": content,
            "missing": missing, "extra": extra, "moved": moved, "labels": labels,
            "warnings": warnings, "ref_count": len(rf), "other_count": len(of)}


def _area(rel):
    parts = rel.split("/")
    return "/".join(parts[:2]) if len(parts) > 2 else parts[0]


def _listing(title, items):
    if not items:
        return []
    lines = [f"{title} ({len(items)}):"]
    lines += [f"  {item}" for item in items[:MAX_LISTED]]
    if len(items) > MAX_LISTED:
        lines.append(f"  ... and {len(items) - MAX_LISTED} more")
    return lines


def format_report(result, ref_name="REFERENCE", other_name="OTHER"):
    lines = [f"Data check: REFERENCE = {ref_name}, OTHER = {other_name}",
             f"Files: REFERENCE {result['ref_count']}, OTHER {result['other_count']}"]
    verdict = "PASS" if result["passed"] else "FAIL"
    lines.append(f"Verdict: {verdict} - files read by the Subtask 2 pipelines "
                 + ("all match" if result["passed"] else "do not all match"))
    lines.append("")
    areas = {}
    for key in ("identical", "text_only", "content", "missing", "extra"):
        for rel in result[key]:
            areas.setdefault(_area(rel), {}).setdefault(key, 0)
            areas[_area(rel)][key] += 1
    lines.append(f"{'area':42} {'same':>6} {'eol':>5} {'diff':>5} {'miss':>5} {'extra':>6}")
    for area in sorted(areas):
        c = areas[area]
        lines.append(f"{area:42} {c.get('identical', 0):6} {c.get('text_only', 0):5} "
                     f"{c.get('content', 0):5} {c.get('missing', 0):5} {c.get('extra', 0):6}")
    lines.append("(eol = same text, different line endings / BOM)")
    lines.append("")
    lines += _listing("FAIL: pipeline files that are missing, moved or different", result["critical_failures"])
    lines += _listing("Content differs", result["content"])
    lines += _listing("Only in REFERENCE", result["missing"])
    lines += _listing("Only in OTHER", result["extra"])
    lines += _listing("Same bytes at a different path (REFERENCE -> OTHER)",
                      [f"{a} -> {b}" for a, b in result["moved"]])
    lines += _listing("Line endings / BOM only", result["text_only"])
    labels = result["labels"]
    lines.append(f"labels.txt: {labels['status']}")
    if labels["only_ref"] or labels["only_other"]:
        lines.append(f"  only in REFERENCE: {labels['only_ref']}")
        lines.append(f"  only in OTHER: {labels['only_other']}")
    lines += _listing("Warnings", result["warnings"])
    return "\n".join(lines)


def _check_root(path):
    """Returns an error message when a directory does not look like the release root."""
    base = os.path.normpath(path)
    if not os.path.isdir(base) or any(
            os.path.isdir(os.path.join(base, p.split("/")[0])) for p in CRITICAL_PREFIXES):
        return None
    for dirpath, dirnames, _ in os.walk(base):
        if "training-second-batch-oct" in dirnames or "initial-training-data" in dirnames:
            return f"{path} is not the release root; use {dirpath} instead"
        if dirpath.count(os.sep) - base.count(os.sep) >= 2:
            dirnames[:] = []
    return f"{path} has no initial-training-data/ or training-second-batch-oct/ folder"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("write", help="hash a release root into a manifest")
    w.add_argument("--root", required=True)
    w.add_argument("--labels")
    w.add_argument("--out", required=True)
    c = sub.add_parser("compare", help="compare two manifests or release roots")
    c.add_argument("reference")
    c.add_argument("other")
    c.add_argument("--ref-labels")
    c.add_argument("--other-labels")
    c.add_argument("--ref-name", default="REFERENCE")
    c.add_argument("--other-name", default="OTHER")
    c.add_argument("--save-other", help="also write OTHER's manifest to this path")
    args = parser.parse_args(argv)

    roots = [args.root] if args.command == "write" else [args.reference, args.other]
    for path in roots:
        if not os.path.exists(path):
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2
        problem = _check_root(path)
        if problem:
            print(f"error: {problem}", file=sys.stderr)
            return 2

    if args.command == "write":
        manifest = build_manifest(args.root, args.labels)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1, sort_keys=True)
        print(f"Wrote {manifest['file_count']} files to {args.out}")
        return 0

    ref = load_side(args.reference, args.ref_labels)
    other = load_side(args.other, args.other_labels)
    if args.save_other:
        with open(args.save_other, "w", encoding="utf-8") as f:
            json.dump(other, f, indent=1, sort_keys=True)
    result = compare_manifests(ref, other)
    print(format_report(result, args.ref_name, args.other_name))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
