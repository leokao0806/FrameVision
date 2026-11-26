"""The results lock (docs/preregistration.md).

Before results/LOCKED exists, C1, C2 and C4 must not read test labels or test predictions.

Usage:
    python -m analysis.lock --create     # once, when every experiment is finished
"""
import argparse
import datetime
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK_PATH = os.path.join(REPO, "results", "LOCKED")


def is_locked():
    return os.path.isfile(LOCK_PATH)


def require_lock():
    if not is_locked():
        raise SystemExit(
            f"{LOCK_PATH} does not exist: test labels and test predictions are not read before the results are "
            "locked (docs/preregistration.md). Create it with `python -m analysis.lock --create` once every "
            "experiment is finished."
        )


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def create_lock():
    if is_locked():
        raise SystemExit(f"{LOCK_PATH} already exists; the results were already locked.")
    os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)
    dirty = bool(_git("status", "--porcelain"))
    with open(LOCK_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"commit {_git('rev-parse', 'HEAD')}\n")
        f.write(f"locked_at {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}\n")
        f.write(f"uncommitted_changes {str(dirty).lower()}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create the results lock.")
    parser.add_argument("--create", action="store_true", help="write results/LOCKED with the current commit")
    args = parser.parse_args(argv)
    if args.create:
        create_lock()
        print(f"Locked: {LOCK_PATH}")
    else:
        print(f"{LOCK_PATH}: {'locked' if is_locked() else 'not locked'}")


if __name__ == "__main__":
    main()
