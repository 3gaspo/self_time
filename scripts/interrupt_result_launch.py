#!/usr/bin/env python3
"""Mark unfinished TIME tasks from one failed cluster launch interrupted."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from timebench.pipeline import interrupt_launch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--launch-id", required=True)
    args = parser.parse_args()
    changed = interrupt_launch(args.root, args.launch_id)
    print(
        "TIME interrupted-launch recovery "
        f"launch_id={args.launch_id} tasks={len(changed)} root={args.root}"
    )
    for run_dir in changed:
        print(f"  interrupted {run_dir}")


if __name__ == "__main__":
    main()
