"""
Export benchmark data from bauplan branches to local parquet files.

Reads user_predictions from each multiverse branch and saves as local
parquet files for offline benchmarking. The shared ecommerce_users table
is assumed to already exist locally and is NOT re-exported.

Usage:
    uv run python src/benchmarks/export_data.py
    uv run python src/benchmarks/export_data.py --upload   # also upload to S3
"""

import argparse
import json
import os
import sys
from pathlib import Path

import bauplan
import pyarrow.parquet as pq
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

NAMESPACE = os.environ.get("BAUPLAN_NAMESPACE", "apo_multiverse")
BRANCH_PREFIX = "multiverse_v_"
LOCAL_DATA_DIR = Path(__file__).parent / "data"
S3_BUCKET = "alpha-hello-bauplan"
S3_PREFIX = "benchmarks/multiverse/v2"


def main():
    parser = argparse.ArgumentParser(description="Export benchmark data from bauplan")
    parser.add_argument(
        "--upload", action="store_true", help="Also upload to S3 after local export"
    )
    args = parser.parse_args()

    client = bauplan.Client()
    username = client.info().user.username

    # Find all multiverse branches
    all_branches = client.get_branches(user=username, limit=500)
    mv_branches = [
        b for b in all_branches if BRANCH_PREFIX in b.name
    ]
    mv_branches.sort(key=lambda b: b.name)

    if not mv_branches:
        print(f"No branches matching '{BRANCH_PREFIX}' found for user {username}")
        sys.exit(1)

    print(f"Found {len(mv_branches)} multiverse branches")

    # Export user_predictions from each branch
    branches_dir = LOCAL_DATA_DIR / "branches"
    manifest_entries = []

    for i, branch in enumerate(mv_branches, 1):
        # Extract variant name from branch
        # e.g. "jacopo.multiverse_v_v_10m_py_gb" -> "v_10m_py_gb"
        variant = branch.name.split(BRANCH_PREFIX, 1)[1]
        print(f"  [{i}/{len(mv_branches)}] {variant} ...", end="", flush=True)

        try:
            table = client.query(
                f"SELECT user_id, conversion_prob, predicted_label FROM {NAMESPACE}.user_predictions",
                ref=branch.name,
                namespace=NAMESPACE,
                args={"runner2": "true"},
            )
            out_dir = branches_dir / variant
            out_dir.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, out_dir / "user_predictions.parquet")
            print(f" {table.num_rows} rows")
            manifest_entries.append({
                "variant": variant,
                "rows": table.num_rows,
                "columns": table.column_names,
            })
        except Exception as e:
            print(f" FAILED: {e}")

    # Read existing shared table info from current manifest
    old_manifest_path = LOCAL_DATA_DIR / "manifest.json"
    shared_info = {}
    if old_manifest_path.exists():
        old = json.loads(old_manifest_path.read_text())
        shared_info = old.get("shared", {})

    # Write manifest
    manifest = {
        "shared": shared_info,
        "branches": manifest_entries,
    }
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = LOCAL_DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written: {manifest_path}")
    print(f"Exported {len(manifest_entries)} branches")

    if args.upload:
        print("\nUploading to S3...")
        import pyarrow.fs as pafs

        s3 = pafs.S3FileSystem()

        # Upload manifest
        s3_manifest = f"{S3_BUCKET}/{S3_PREFIX}/manifest.json"
        with s3.open_output_stream(s3_manifest) as f:
            f.write(json.dumps(manifest, indent=2).encode())
        print(f"  Uploaded manifest to s3://{s3_manifest}")

        # Upload branch parquet files
        for entry in manifest_entries:
            variant = entry["variant"]
            local_path = branches_dir / variant / "user_predictions.parquet"
            s3_path = f"{S3_BUCKET}/{S3_PREFIX}/branches/{variant}/user_predictions.parquet"
            table = pq.read_table(local_path)
            with s3.open_output_stream(s3_path) as f:
                pq.write_table(table, f)
            print(f"  Uploaded {variant}")

        # Copy shared table to new S3 prefix
        shared_local = LOCAL_DATA_DIR / "shared" / "ecommerce_users.parquet"
        if shared_local.exists():
            s3_shared = f"{S3_BUCKET}/{S3_PREFIX}/shared/ecommerce_users.parquet"
            table = pq.read_table(shared_local)
            with s3.open_output_stream(s3_shared) as f:
                pq.write_table(table, f)
            print("  Uploaded shared/ecommerce_users.parquet")

        print("S3 upload complete.")


if __name__ == "__main__":
    main()
