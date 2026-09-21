#!/usr/bin/env python3
"""Resumable downloader for the Dataset-P&ID annotation folders.

Google Drive rate-limits hard on a few thousand small files: a single flat pass
got 998 of 3500 and then every remaining request came back "Cannot retrieve the
public link of the file". This caches the folder listing once, skips whatever is
already on disk, and retries the rest with exponential backoff, so it can be run
repeatedly until `--report` shows 500 complete sheets.

Usage:
    python tools/fetch_ann.py --workers 4          # one pass
    python tools/fetch_ann.py --report             # just show what is missing
"""
from __future__ import annotations

import argparse
import json
import os
import sys as _sys; _sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from _paths import ROOT as _ROOT  # <ROOT>/PnIDAgent/eval_tools layout
import random
import time
from concurrent.futures import ThreadPoolExecutor

ROOT_URL = ("https://drive.google.com/drive/folders/"
            "1gMm_YKBZtXB3qUKUpI-LF1HE_MgzwfeR")
# File names are case-sensitive on Linux: the Drive folders hold _KeyValue.npy
# and _Table.npy, so the lowercase names only matched on a macOS volume.
SUFFIXES = ("_symbols.npy", "_lines.npy", "_lines2.npy", "_words.npy",
            "_linker.npy", "_KeyValue.npy", "_Table.npy")


def listing(ann: str, refresh: bool = False) -> list[dict]:
    """Folder listing as [{path, id}], cached to ann/_listing.json."""
    cache = os.path.join(ann, "_listing.json")
    if os.path.exists(cache) and not refresh:
        with open(cache) as fh:
            return json.load(fh)
    import gdown
    entries = gdown.download_folder(ROOT_URL, skip_download=True, quiet=True)
    # Folders 245/246/247 hold duplicate uploads, so keep the first of each path.
    seen, out = set(), []
    for e in entries:
        if "/" not in e.path or e.path in seen:
            continue
        seen.add(e.path)
        out.append({"path": e.path, "id": e.id})
    with open(cache, "w") as fh:
        json.dump(out, fh)
    return out


def missing(ann: str, entries: list[dict]) -> list[dict]:
    out = []
    for e in entries:
        dst = os.path.join(ann, e["path"])
        if not os.path.exists(dst) or os.path.getsize(dst) == 0:
            out.append(e)
    return out


_FAILS = [0]                 # consecutive failures across workers
_COOL_AFTER = 12             # this many in a row means Drive is throttling us
_COOL_SECONDS = 180


def fetch(args_tuple) -> bool:
    ann, e, tries = args_tuple
    import gdown
    dst = os.path.join(ann, e["path"])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for k in range(tries):
        if _FAILS[0] >= _COOL_AFTER:
            # Hammering a throttled endpoint only extends the throttle; sit
            # out, then let one request probe whether it has lifted.
            time.sleep(_COOL_SECONDS + random.random() * 30)
            _FAILS[0] = _COOL_AFTER - 1
        try:
            got = gdown.download(id=e["id"], output=dst, quiet=True)
            if got and os.path.getsize(dst) > 0:
                _FAILS[0] = 0
                return True
        except Exception:
            pass
        _FAILS[0] += 1
        # Drive throttles by request rate, so back off with jitter.
        time.sleep((2 ** k) + random.random() * 2)
    return False


def report(ann: str) -> tuple[int, list[int]]:
    ids = sorted(int(d) for d in os.listdir(ann)
                 if d.isdigit() and os.path.isdir(os.path.join(ann, d)))
    bad = [i for i in ids
           if not all(os.path.exists(os.path.join(ann, str(i), f"{i}{s}"))
                      for s in SUFFIXES)]
    return len(ids) - len(bad), bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default=os.path.join(
        _ROOT,
        "dataset", "ann"))
    ap.add_argument("--workers", type=int, default=4,
                    help="keep this low; Drive throttles on concurrency")
    ap.add_argument("--tries", type=int, default=4)
    ap.add_argument("--refresh-listing", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    ann = os.path.abspath(args.ann)
    ok, bad = report(ann)
    print(f"complete sheets: {ok}   incomplete: {len(bad)}")
    if args.report:
        print(f"incomplete ids: {bad}")
        return

    entries = listing(ann, refresh=args.refresh_listing)
    todo = missing(ann, entries)
    print(f"listing has {len(entries)} files; {len(todo)} still to fetch",
          flush=True)
    if not todo:
        return

    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for got in ex.map(fetch, [(ann, e, args.tries) for e in todo]):
            done += 1
            if done % 25 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}", flush=True)

    ok, bad = report(ann)
    print(f"\ncomplete sheets: {ok}   incomplete: {len(bad)}")
    if bad:
        print("rerun to pick up the rest")


if __name__ == "__main__":
    main()
