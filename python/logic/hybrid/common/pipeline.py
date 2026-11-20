"""Run-side pipeline shared by the hybrid methods: evidence alignment, knowledge table, deposits, split, result.

Evidence entries of the parameter dict (built by :class:`logic.hybrid.common.dialogs.KnowledgeTableWidget`)::

    dict(name, data, grid, kind="continuous" | "categorical", process="pathway", direction=+1 | -1,
         scores="2=0.9; 1=0.1" (categorical: expert score 0–1 per class), weight=1.0 (expert weight in the process))

``process_weights`` = {process: weight}. Common keys: deposits, deposits_name, mask, test_fraction, seed,
split_method, thin_distance, negatives, auto_align, min_distance (map units), prefix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from core.methods import MethodResult, OutputRaster
from evaluation.metrics import evaluate, negative_cells
from i18n import tr
from logic.common.training import deposit_point_outputs, split_deposits, thin_deposits
from utils.raster_tools import align_to_grid, points_to_cells

from .knowledge import ExpertLayer, KnowledgeNetwork, KnowledgeTable, negative_candidates, study_cells

PROCESSES = ("source", "pathway", "trap", "deposition", "preservation", "footprint")


def parse_scores(text: str) -> Dict[float, float]:
    """'2=0.9; 1=0.1' -> {2.0: 0.9, 1.0: 0.1} (scores clipped to 0.01 … 0.99)."""
    out = {}
    for part in str(text or "").replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(tr("class scores must look like '2=0.9; 1=0.1'"))
        k, v = part.split("=", 1)
        try:
            out[float(k)] = float(np.clip(float(v), 0.01, 0.99))
        except ValueError:
            raise ValueError(tr("class scores must look like '2=0.9; 1=0.1'")) from None
    if not out:
        raise ValueError(tr("a categorical layer needs a score for at least one class"))
    return out


@dataclass
class HybridContext:
    grid: object
    net: KnowledgeNetwork
    table: KnowledgeTable
    train_pos: np.ndarray                 # positions (in net.cells) of the training deposits
    test_rows: np.ndarray
    test_cols: np.ndarray
    train_rows: np.ndarray
    train_cols: np.ndarray
    known_rows: np.ndarray                # all deposits (also the test ones) — excluded from the negatives
    known_cols: np.ndarray
    neg_cand: np.ndarray                  # positions of cells far enough from every known deposit
    min_distance: float
    messages: List[str] = field(default_factory=list)

    @property
    def n_train(self) -> int:
        return int(self.train_pos.size)


def build_table(params: dict, grid, layers_only: bool = False) -> KnowledgeTable:
    layers = []
    for e in params["evidence"]:
        data = e["data"]
        cat = e.get("kind") == "categorical"
        if not e["grid"].same_as(grid):
            if not params.get("auto_align", True):
                raise ValueError(tr("{layer} is not on the grid of {ref}", layer=e["name"],
                                    ref=params["evidence"][0]["name"]))
            data = align_to_grid(data, e["grid"], grid, "nearest" if cat else "bilinear")
        layers.append(ExpertLayer(e["name"], data, str(e.get("process") or "process").strip() or "process",
                                  -1 if float(e.get("direction", 1)) < 0 else 1, float(e.get("weight", 1.0)),
                                  parse_scores(e.get("scores")) if cat else None))
    return KnowledgeTable(layers, {str(k): float(v) for k, v in (params.get("process_weights") or {}).items()})


def study_mask(params: dict, grid) -> Optional[np.ndarray]:
    if params.get("mask") is None:
        return None
    m, mg = params["mask"]
    if not mg.same_as(grid):
        m = align_to_grid(m, mg, grid, "nearest")
    return ~np.ma.getmaskarray(m)


def prepare(params: dict, progress, min_deposits: int = 3, need_deposits: bool = True) -> HybridContext:
    evs = params["evidence"]
    if not evs:
        raise ValueError(tr("Tick at least one evidence layer."))
    grid = evs[0]["grid"]
    progress(3, tr("preparing evidence"))
    table = build_table(params, grid)
    study = study_cells(table, study_mask(params, grid))
    net = KnowledgeNetwork(table, study, s=params.get("s0", 1.5), c=params.get("c0", 0.5), p=params.get("p", 3.0))
    msgs = []
    progress(6, tr("locating deposits"))
    if params.get("deposits") is not None and len(params["deposits"]):
        rows, cols = points_to_cells(params["deposits"], grid, valid_mask=study)
    else:
        rows = cols = np.zeros(0, int)
    known_r, known_c = rows, cols
    rows, cols, thin_msg = thin_deposits(rows, cols, params.get("thin_distance", 0.0), grid.res)
    msgs += thin_msg
    if need_deposits and len(rows) < min_deposits:
        raise ValueError(tr("Fewer than {n} deposits fall inside the study area / evidence grid.", n=min_deposits))
    tr_r, tr_c, te_r, te_c = split_deposits(rows, cols, params.get("test_fraction", 0.0), params.get("seed", 0),
                                            params.get("split_method", "spatial"))
    pos = net.positions(tr_r, tr_c)
    dist = float(params.get("min_distance", 3 * grid.res[0]))
    cand = negative_candidates(net, known_r, known_c, dist, grid.res) if len(known_r) else np.arange(net.cells.size)
    if cand.size < 10:
        raise ValueError(tr("Too few cells are left for non-deposit samples — reduce the minimum distance."))
    return HybridContext(grid, net, table, pos, te_r, te_c, tr_r, tr_c, known_r, known_c, cand, dist, msgs)


def finish(key: str, ctx: HybridContext, params: dict, outputs: List[OutputRaster], tables: dict,
           messages: List[str], extra: dict, progress, label: str) -> MethodResult:
    """Validate the first output (training / test deposits) and pack the method result."""
    progress(94, tr("validating"))
    score = outputs[0].data
    ev = None
    if len(ctx.train_rows):
        ev = evaluate(score, ctx.train_rows, ctx.train_cols, ctx.test_rows, ctx.test_cols,
                      valid_mask=ctx.net.study, score_name=outputs[0].name,
                      negatives=negative_cells(params.get("negatives"), ctx.grid, ctx.net.study))
    vectors = []
    if params.get("deposits") is not None and len(ctx.train_rows):
        vectors = deposit_point_outputs(params["deposits"], ctx.grid, ctx.train_rows, ctx.train_cols,
                                        ctx.test_rows, ctx.test_cols)
    progress(100, tr("done"))
    extra = dict(extra, deposits_name=params.get("deposits_name"), train_cells=(ctx.train_rows, ctx.train_cols),
                 test_cells=(ctx.test_rows, ctx.test_cols), processes=ctx.table.processes)
    return MethodResult(key, outputs, vectors=vectors, label=params.get("prefix", label), tables=tables,
                        evaluation=ev, messages=ctx.messages + messages, extra=extra)


def knowledge_outputs(ctx: HybridContext, F: np.ndarray, K: np.ndarray, prefix: str = "", with_processes=True,
                      visible=False) -> List[OutputRaster]:
    """Expert map, limiting process (1 … K, see the process table) and optional process maps."""
    net = ctx.net
    outs = [OutputRaster(prefix + "expert", net.grid(K).astype("float32"), ctx.grid, cmap="viridis",
                         description=tr("Expert knowledge favourability K (no deposits used)"), visible=visible)]
    if net.K > 1:
        lim = np.argmin(F, 1).astype("uint8") + 1
        outs.append(OutputRaster(prefix + "limiting_process", net.grid(lim, "uint8"), ctx.grid, categorical=True,
                                 cmap="tab10", visible=False,
                                 description=tr("Limiting (weakest) mineral-system process: {codes}",
                                                codes=", ".join(f"{i + 1} = {p}" for i, p in
                                                                enumerate(ctx.table.processes)))))
    if with_processes:
        for i, p in enumerate(ctx.table.processes):
            outs.append(OutputRaster(f"{prefix}process_{_safe(p)}", net.grid(F[:, i]).astype("float32"), ctx.grid,
                                     cmap="viridis", visible=False,
                                     description=tr("Process score F of '{process}'", process=p)))
    return outs


def _safe(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s.lower())[:24] or "process"


def binomial_deviance(y, eta) -> float:
    return float(2 * np.sum(np.logaddexp(0, eta) - y * eta))
