"""
Refresh docs/assets/waterpark-datasets.json from the live bucket listing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import s3fs

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent.parent / "docs" / "assets" / "waterpark-datasets.json"

ANNOUNCEMENTS = """{{% extends "base.html" %}}
{{% block announce %}}
{announcements}
{{% endblock %}}
"""


def list_buckets(endpoint: str, key: str, secret: str) -> list[str]:
    fs = s3fs.S3FileSystem(
        key=key, secret=secret, client_kwargs={"endpoint_url": endpoint}
    )
    names: list[str] = []
    for root in ("", "/"):
        try:
            items = fs.ls(root, detail=False)
            names = [i.strip("/").split("/")[-1] for i in items if i.strip("/")]
            if names:
                break
        except Exception:
            continue
    if not names:
        raise SystemExit(
            f"could not list buckets from {endpoint} "
            f"(check admin credentials / gateway permissions)"
        )
    return sorted(names)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--endpoint", default="https://s3.waterpark.dkrz.de")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    key = os.environ.get("WATERPARK_S3_KEY")
    secret = os.environ.get("WATERPARK_S3_SECRET")
    if not key or not secret:
        raise SystemExit(
            "set WATERPARK_S3_KEY and WATERPARK_S3_SECRET in the environment"
        )

    buckets = list_buckets(args.endpoint, key, secret)

    # Blacklist
    blacklist = {
        b.strip()
        for b in os.environ.get("WATERPARK_BUCKET_BLACKLIST", "").split(",")
        if b.strip()
    }
    if blacklist:
        buckets = [b for b in buckets if b not in blacklist]

    # check if the JSON is already there.
    existing: dict = {}
    if args.out.exists():
        try:
            existing = json.loads(args.out.read_text()).get("datasets", {})
        except (json.JSONDecodeError, OSError):
            existing = {}

    # Reconcile: keep existing descriptions add new buckets bare, drop the rest.
    datasets = {b: existing.get(b, {"title": b}) for b in buckets}

    added = [b for b in buckets if b not in existing]
    removed = [b for b in existing if b not in buckets]

    payload = {"buckets": buckets, "datasets": datasets}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"wrote {args.out}  ({len(buckets)} buckets)", file=sys.stderr)
    if added:
        print(f"added (no description): {', '.join(added)}", file=sys.stderr)
    if removed:
        print(f"removed: {', '.join(removed)}", file=sys.stderr)
    announ = os.getenv("ANNOUNCEMENTS", "").strip()
    if announ:
        extra = ANNOUNCEMENTS.format(announcements=announ)
        extra_file = HERE.parent.parent / "docs" / "data" / "overrides" / "main.html"
        extra_file.write_text(extra)


if __name__ == "__main__":
    main()
