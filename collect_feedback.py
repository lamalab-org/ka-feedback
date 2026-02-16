"""
Aggregate chemist feedback from local feedback/ directory.

Output structure:
{
  "user_name": {
    "run_type_1": {
      "networks": [network1, network2, ...],
      "whats_good": ["...", "..."],
      "whats_bad": ["...", "..."],
      "suggested_actions": ["...", "..."],
      "scores": {"run_dir": score, ...},
      "num_submissions": 3,
      "timestamps": ["2025-01-01T...", ...]
    },
    "run_type_2": { ... }
  },
  "other_user": { ... }
}

Usage:
    python aggregate_feedback.py [--feedback-dir feedback/] [--output aggregated.json]
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_all_feedback_local(feedback_dir: str = "feedback") -> list[dict]:
    """Walk feedback/{user_slug}/{run_type_slug}.json and collect all entries."""
    root = Path(feedback_dir)
    if not root.exists():
        print(f"Warning: {root} does not exist")
        return []

    all_entries = []
    for user_dir in sorted(root.iterdir()):
        if not user_dir.is_dir():
            continue
        for json_file in sorted(user_dir.glob("*.json")):
            try:
                entries = json.loads(json_file.read_text())
            except (json.JSONDecodeError, OSError) as e:
                print(f"Skipping {json_file}: {e}")
                continue

            if not isinstance(entries, list):
                entries = [entries]

            for entry in entries:
                entry["_user_slug"] = user_dir.name
                entry["_file"] = json_file.name
                all_entries.append(entry)

    return all_entries


def aggregate(entries: list[dict]) -> dict:
    """
    Reshape flat list of feedback entries into:
      { user: { run_type: { networks, whats_good, whats_bad, ... } } }
    """
    result = defaultdict(lambda: defaultdict(lambda: {
        "networks": [],
        "whats_good": [],
        "whats_bad": [],
        "suggested_actions": [],
        "scores": {},
        "timestamps": [],
        "num_submissions": 0,
        "run_directories": [],
        "glob_template": None,
    }))

    for entry in entries:
        user = entry.get("user", entry.get("_user_slug", "unknown"))
        run_type = entry.get("run_type", entry.get("_file", "unknown").replace(".json", ""))
        bucket = result[user][run_type]

        bucket["num_submissions"] += 1

        ts = entry.get("timestamp")
        if ts:
            bucket["timestamps"].append(ts)

        if entry.get("whats_good"):
            bucket["whats_good"].append(entry["whats_good"])
        if entry.get("whats_bad"):
            bucket["whats_bad"].append(entry["whats_bad"])
        if entry.get("suggested_action"):
            bucket["suggested_actions"].append(entry["suggested_action"])

        # Collect network snapshots (one per run directory per submission)
        snapshots = entry.get("network_snapshots", {})
        for run_dir, network in snapshots.items():
            if network and network.get("reactions"):
                bucket["networks"].append({
                    "run_dir": run_dir,
                    "timestamp": ts,
                    "network": network,
                })

        # Merge per-run scores
        for run_dir, score in entry.get("per_run_scores", {}).items():
            bucket["scores"][run_dir] = score

        # Keep track of run directories
        for rd in entry.get("run_directories", []):
            if rd not in bucket["run_directories"]:
                bucket["run_directories"].append(rd)

        if entry.get("glob_template"):
            bucket["glob_template"] = entry["glob_template"]

    # Convert defaultdicts to plain dicts for clean JSON serialization
    return {user: dict(run_types) for user, run_types in result.items()}


def print_summary(aggregated: dict):
    """Print a human-readable summary to stdout."""
    total_users = len(aggregated)
    total_submissions = sum(
        info["num_submissions"]
        for user_data in aggregated.values()
        for info in user_data.values()
    )
    total_networks = sum(
        len(info["networks"])
        for user_data in aggregated.values()
        for info in user_data.values()
    )

    print(f"\n{'='*60}")
    print(f"  Feedback Aggregation Summary")
    print(f"{'='*60}")
    print(f"  Users:        {total_users}")
    print(f"  Submissions:  {total_submissions}")
    print(f"  Networks:     {total_networks}")
    print(f"{'='*60}\n")

    for user, run_types in sorted(aggregated.items()):
        print(f"👤 {user}")
        for run_type, info in sorted(run_types.items()):
            n_nets = len(info["networks"])
            n_subs = info["num_submissions"]
            print(f"   ├─ {run_type}  ({n_subs} submission(s), {n_nets} network(s))")
            for good in info["whats_good"]:
                print(f"   │   ✅ {good[:80]}{'...' if len(good) > 80 else ''}")
            for bad in info["whats_bad"]:
                print(f"   │   ❌ {bad[:80]}{'...' if len(bad) > 80 else ''}")
            for action in info["suggested_actions"]:
                print(f"   │   🔧 {action[:80]}{'...' if len(action) > 80 else ''}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Aggregate chemist feedback from local JSON files")
    parser.add_argument("--feedback-dir", default="feedback", help="Root feedback directory (default: feedback/)")
    parser.add_argument("--output", "-o", default="aggregated_feedback.json", help="Output JSON path")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress summary output")
    args = parser.parse_args()

    entries = load_all_feedback_local(args.feedback_dir)
    if not entries:
        print("No feedback entries found.")
        return

    aggregated = aggregate(entries)

    Path(args.output).write_text(json.dumps(aggregated, indent=2, default=str))
    print(f"Wrote {args.output}")

    if not args.quiet:
        print_summary(aggregated)


if __name__ == "__main__":
    main()