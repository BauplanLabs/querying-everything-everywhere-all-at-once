"""
Launch all pipeline variants in parallel on separate bauplan branches.

For each pipeline project found in src/bpln/, this script:
1. Creates a multiverse_main branch from the current main HEAD
2. Creates variant branches from multiverse_main
3. Runs each pipeline on its branch (materializes by default)
4. Reports results

Usage:
    uv run python src/demo.py                    # all variants, 1M sample
    uv run python src/demo.py --no-sampling      # all variants, full data
    uv run python src/demo.py --max-variants 3   # random 3 variants
    uv run python src/demo.py --max-workers 4    # cap parallel jobs
"""

import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import bauplan
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

BPLN_DIR = Path(__file__).parent / "bpln"
NAMESPACE = os.environ.get("BAUPLAN_NAMESPACE", "apo_multiverse")
BRANCH_PREFIX = "multiverse_v_"
DEMO_MAIN_SUFFIX = "multiverse_main"
RANDOM_SEED = 9


def discover_projects(base_dir):
    """Find all bauplan project folders (contain bauplan_project.yml).

    Folders named 'setup_*' are always excluded (they are one-time setup DAGs).
    """
    projects = []
    for child in sorted(base_dir.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "bauplan_project.yml").exists():
            continue
        if child.name.startswith("setup_"):
            continue
        projects.append(child)
    return projects


def run_pipeline(client, project_dir, branch_name, dry_run, sample_size):
    """Run a single pipeline on its branch. Returns (branch_name, status, error).

    Checks job status after run and extracts error details on failure.
    """
    state = client.run(
        project_dir=str(project_dir),
        ref=branch_name,
        namespace=NAMESPACE,
        dry_run=dry_run,
        cache="off",
        client_timeout=900,
        parameters={"size": sample_size},
    )
    status = state.job_status
    error = None
    if status != "SUCCESS":
        error = state.error
    return branch_name, status, error


MIN_SAMPLE_SIZE = 1_000_000


def launch(dry_run=False, sample_size=0, max_variants=0, max_workers=4):
    """Discover projects, create branches, run pipelines, report results."""
    if max_workers < 1:
        print("Error: --max-workers must be >= 1.")
        sys.exit(1)

    client = bauplan.Client()
    username = client.info().user.username

    # Clean up all previous multiverse branches before doing anything else
    demo_main_name = f"{username}.{DEMO_MAIN_SUFFIX}"
    print("Cleaning up old multiverse branches...")
    old_branches = client.get_branches(user=username, limit=500)
    cleaned = 0
    for b in old_branches:
        if BRANCH_PREFIX in b.name or b.name == demo_main_name:
            client.delete_branch(b.name, if_exists=True)
            cleaned += 1
    print(f"  Deleted {cleaned} old branch(es).")

    # Clear text-to-SQL cache so stale entries don't persist across runs
    cache_dir = Path(__file__).parent / "app" / ".sql_cache"
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
        print("  Cleared SQL cache.")
    print()

    # Pin the exact main HEAD hash
    main_commits = client.get_commits(ref="main", limit=1)
    main_head = list(main_commits)[0]
    main_hash = main_head.ref.hash
    print(f"Starting commit (main HEAD): {main_hash}")
    print(f"  short: {main_hash[:12]}")
    print(f"  message: {main_head.message}")
    print()

    # Create a dedicated demo-main branch from main HEAD.
    # This gives us a clean trunk with no interference from other users,
    # and all variant branches fork from it.
    client.delete_branch(demo_main_name, if_exists=True)
    client.create_branch(demo_main_name, from_ref=f"main@{main_hash}")
    print(f"Demo main branch: {demo_main_name} (from {main_hash[:12]})")
    print()

    # Discover pipeline projects (deterministic shuffle with fixed seed)
    all_projects = discover_projects(BPLN_DIR)
    if not all_projects:
        print(f"No pipeline projects found in {BPLN_DIR}")
        sys.exit(1)

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_projects)

    projects = all_projects if max_variants == 0 else all_projects[:max_variants]

    print(f"Found {len(all_projects)} variant(s), running {len(projects)}:")
    for p in projects:
        print(f"  - {p.name}")
    print()

    # Create fresh branches from demo-main (delete first to be idempotent)
    branches = {}
    for project_dir in projects:
        branch_name = f"{username}.{BRANCH_PREFIX}{project_dir.name}"
        client.delete_branch(branch_name, if_exists=True)
        client.create_branch(branch_name, from_ref=demo_main_name)
        branches[project_dir] = branch_name
        print(f"Branch: {branch_name} (from {demo_main_name})")

    print()
    mode = "DRY-RUN" if dry_run else "MATERIALIZE"
    size_label = f", size={sample_size}" if sample_size > 0 else ", size=ALL"
    print(f"Launching {len(branches)} pipeline(s) in parallel [{mode}{size_label}]...")
    print()

    effective_workers = min(max_workers, len(branches))
    results = {}
    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
        futures = {
            pool.submit(
                run_pipeline, client, project_dir, branch_name, dry_run, sample_size
            ): branch_name
            for project_dir, branch_name in branches.items()
        }
        for future in as_completed(futures):
            branch_name = futures[future]
            try:
                _, status, error = future.result()
                results[branch_name] = status
                if status != "SUCCESS":
                    print(f"  {branch_name}: {status} -- {error}")
                else:
                    print(f"  {branch_name}: {status}")
            except Exception as exc:
                results[branch_name] = f"ERROR: {exc}"
                print(f"  {branch_name}: ERROR - {exc}")

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Main HEAD: {main_hash[:12]} ({main_head.message})")
    print(f"Demo main: {demo_main_name}")
    print()

    all_ok = True
    for project_dir, branch_name in branches.items():
        status = results.get(branch_name, "UNKNOWN")
        marker = "OK" if status == "SUCCESS" else "FAIL"
        if status != "SUCCESS":
            all_ok = False
        print(f"  [{marker}] {branch_name} ({project_dir.name}): {status}")

        # Show branch commits
        try:
            commits = client.get_commits(ref=branch_name, limit=5)
            for c in commits:
                print(f"        {c.ref.hash[:12]} {c.message[:60]}")
        except Exception:
            pass

    print()
    if all_ok:
        print("All pipelines succeeded.")
    else:
        print("Some pipelines failed — check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run pipeline variants in parallel on bauplan branches"
    )
    parser.add_argument(
        "--no-sampling",
        action="store_true",
        help="Use all rows (default is 1M sample for faster iteration)",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=5,
        help="Number of variants to run, 0 = all (default: 5)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum parallel bauplan jobs (default: 4)",
    )
    args = parser.parse_args()
    sample_size = 0 if args.no_sampling else MIN_SAMPLE_SIZE
    launch(
        sample_size=sample_size,
        max_variants=args.max_variants,
        max_workers=args.max_workers,
    )
