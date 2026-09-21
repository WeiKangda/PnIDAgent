#!/usr/bin/env python3
"""Adapters between PnIDAgent/pnid_graph.build_topology and the dpid.Graph metric.

The assembler itself lives in the repo (PnIDAgent/pnid_graph.py) so digitize_pnid
uses the same code that is evaluated here.  Two entry points:

  assemble()    call build_topology directly on segments + boxes
  repo_links()  run PnIDAgent/digitize_pnid.digitize_pnid on in-memory inputs
                with either assembler ("topology" default, "chains" legacy)
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys as _sys; _sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from _paths import ROOT as _ROOT  # <ROOT>/PnIDAgent/eval_tools layout
import sys
import tempfile
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpid  # noqa: E402

REPO = os.path.join(_ROOT,
                    "PnIDAgent")
sys.path.insert(0, REPO)
import pnid_graph as PG  # noqa: E402

Seg = Tuple[int, int, int, int]
Box = Tuple[int, int, int, int]

# re-exported for callers that used the old module-level helpers
_project, _len, _in_box, _seg_box_hit = PG.project, PG.seg_len, PG.in_box, PG.seg_box_hit
_point_seg_dist, merge_collinear = PG.point_seg_dist, PG.merge_collinear


def assemble(segs: Sequence[Tuple[Seg, str]], symbols: Sequence[Tuple[str, Box]],
             tol: int = 12, pad: int = 6, merge: bool = True, min_len: float = 0.0,
             drop_inbox_ends: bool = True) -> dpid.Graph:
    nodes, edges = PG.build_topology(segs, symbols, tol=tol, pad=pad, merge=merge,
                                     min_len=min_len, drop_inbox_ends=drop_inbox_ends)
    return dpid.Graph(nodes=nodes, edges=edges)


def repo_links(segs: Sequence[Tuple[Seg, str]], symbols: Sequence[Tuple[str, Box]],
               max_line_distance: float = 50.0, shape=(4561, 7168),
               assembler: str = "topology", snap_tol: int = 12, symbol_pad: int = 6
               ) -> dpid.Graph:
    """Run digitize_pnid.digitize_pnid on in-memory inputs; return a dpid.Graph.

    Symbol nodes keep the caller's ids; junction nodes (topology assembler only)
    become ("jct", i) with their positions.
    """
    import digitize_pnid as DP  # noqa: E402

    ids = [sid for sid, _ in symbols]
    with tempfile.TemporaryDirectory() as td:
        cls_p, txt_p, ln_p = (os.path.join(td, n) for n in ("c.json", "t.json", "l.json"))
        with open(cls_p, "w") as fh:
            json.dump({"symbols": [{"id": i, "bbox": [int(v) for v in b],
                                    "category": "symbol"} for i, (_, b) in enumerate(symbols)]}, fh)
        with open(txt_p, "w") as fh:
            json.dump([], fh)
        with open(ln_p, "w") as fh:
            json.dump({"solid": [[int(v) for v in s] for s, k in segs if k == "solid"],
                       "dashed": [[int(v) for v in s] for s, k in segs if k == "dashed"],
                       "resized_shape": list(shape), "scale": 1.0}, fh)
        with contextlib.redirect_stdout(io.StringIO()):
            full, _ = DP.digitize_pnid(cls_p, txt_p, ln_p, None, 100.0, max_line_distance,
                                       assembler=assembler, snap_tol=snap_tol,
                                       symbol_pad=symbol_pad)

    nodes: Dict[object, Tuple[float, float]] = {}
    key_of: Dict[object, object] = {}
    for n in full["nodes"]:
        if n.get("category") == "junction":
            k = ("jct", int(str(n["id"])[1:]))
            nodes[k] = tuple(n["position"])
        else:
            k = ("sym", ids[n["id"]])
            b = n["bbox"]
            nodes[k] = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
        key_of[n["id"]] = k
    edges = set()
    for l in full["links"]:
        a, b = sorted([key_of[l["source"]], key_of[l["target"]]], key=lambda n: (n[0], str(n[1])))
        edges.add((a, b, l.get("type", "solid")))
    return dpid.Graph(nodes=nodes, edges=sorted(edges, key=lambda e: (str(e[0]), str(e[1]))))
