#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dependency Graph Visualizer (PyQt6) with strict Laplacian Spectral Clustering, 2D K-Means,
DBSCAN (+ DBCV auto-tuning with scale normalization & noise penalty), cluster packing,
auto re-layout, **and experimental TDA / Mapper views for dependency structures.**

Key features:

* List installed packages (pip list), select roots via checkboxes (Select All / Clear).
* Build **dependency** (what they depend on) or **dependents** (who depends on them) graphs.
* Layout on CPU (spring / kamada-kawai) or GPU (cuGraph ForceAtlas2).
* Zoom (wheel) & pan (drag), axes hidden by default, cycles highlighted (red border).
* Coloring modes:

  * Degree colormap (Spectral / viridis) + colorbar legend.
  * Spectral clustering (exact k) with categorical legend.
  * 2D K-Means clustering (exact k) with categorical legend.
  * DBSCAN clustering on 2D positions (variable clusters, noise shown) with categorical legend.
* Spectral clustering pipeline (always exactly k):

  1. Strict Laplacian spectral (normalized Laplacian, Fiedler-based bisection, per component).
  2. sklearn SpectralClustering(affinity="precomputed").
  3. 2D K-Means on current positions (NumPy-only).
  4. Greedy merge of connected components to exactly k.
* DBSCAN:

  * Parameters: eps (QDoubleSpinBox), min_cluster_size (maps to sklearn's min_samples).
  * Optional **“Auto-tune (DBCV)”** grid search (scale-normalized):

    * Evaluate in normalized space (median 1-NN distance ≈ 1) to remove layout scale effects.
    * Score via hdbscan.validity.validity_index (DBCV) if available, else silhouette.
    * Penalize heavy noise: `effective_score = score * sqrt(clustered_fraction)`.
    * Enforce minimum coverage (≥ 30%) to avoid tiny clusters + huge noise.
    * Fallback to 2D K-Means(k) if DBSCAN yields <2 clusters.
* Cluster packing:

  * When enabled, rearranges positions so each cluster forms a visible spatial “blob”.
  * Packs clusters using a quotient graph (clusters as super-nodes), lays out per-cluster subgraphs
    locally, then composes them with spacing and radius heuristics.
* **TDA panel (experimental):**

  * Builds scalar fields on the dependency graph (depth-from-roots, out-degree, betweenness).
  * Runs a lightweight 0-dimensional persistent homology on sublevel sets over that scalar.
  * Shows an H₀ “barcode” so you can see when connected components are born/merged as the
    filtration grows.
  * Uses the same graph you just built — no extra data source required.
* **Mapper-like view (experimental):**

  * Uses the same scalar as a 1D “lens”.
  * Covers the scalar range with overlapping intervals (configurable count & overlap).
  * Inside each interval, takes induced subgraphs and connected components → Mapper nodes.
  * Connects nodes that share original packages → gives you a coarse topological summary of
    “similar dependency regions”.
* Export figure to PNG / SVG.
* Log panel records fallbacks, build stats, exports, errors, **TDA/Mapper updates**, and detailed DBCV decisions.
* Changing Backend/Layout or layout params triggers automatic re-layout + redraw, and the TDA/Mapper
  panels are refreshed on each successful graph build.

"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set
import numpy as np

import matplotlib
import networkx as nx
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

# importlib.metadata (stdlib) or backport
try:
    from importlib import metadata as importlib_metadata
except Exception:  # pragma: no cover
    import importlib_metadata  # type: ignore

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QHeaderView,
    QCheckBox,
    QPushButton,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QGroupBox,
    QFileDialog,
)

# Matplotlib canvas
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import colors as mcolors
from matplotlib.patches import Patch

# Types
Pos = Tuple[float, float]
PosDict = Dict[str, Pos]


# --------------------------- Data structures ---------------------------

@dataclass(frozen=True)
class NodeInfo:
    """Display metadata for a package node.

    Attributes
    ----------
    name : str
        Canonicalized distribution name (lower-case, normalized).
    display : str
        Human-friendly label (e.g., "PkgName==1.2.3" or "name (missing)").
    installed : bool
        True if present in importlib.metadata; False if missing.
    """
    name: str
    display: str
    installed: bool


# --------------------------- Metadata helpers ---------------------------

def _safe_parse_requires(requires: Optional[Iterable[str]]) -> List[Requirement]:
    """Parse a sequence of Requires-Dist strings safely.

    Skips malformed lines and ignores PEP 508 markers that do not apply to the
    current environment.

    Parameters
    ----------
    requires : Optional[Iterable[str]]
        Raw 'Requires-Dist' entries from package metadata.

    Returns
    -------
    List[Requirement]
        Parsed requirements applicable to the current environment.
    """
    parsed: List[Requirement] = []
    if not requires:
        return parsed
    for spec in requires:
        try:
            req = Requirement(spec)
            if req.marker is not None and not req.marker.evaluate():
                continue
            parsed.append(req)
        except Exception:
            continue
    return parsed


def build_dependency_graph_bfs(
    roots: Sequence[str],
    depth_limit: int = -1,
    cancel_flag: Optional[List[bool]] = None,
) -> Tuple[nx.DiGraph, Dict[str, NodeInfo]]:
    """Build a directed dependency graph (DEPENDENCIES) using BFS.

    Edge u→v means: “u depends on v”.

    Parameters
    ----------
    roots : Sequence[str]
        Starting packages (any case).
    depth_limit : int, default -1
        -1 = unlimited, 0 = direct-only, N = depth N.
    cancel_flag : Optional[List[bool]]
        If provided, set cancel_flag[0] = True to cooperatively cancel.

    Returns
    -------
    (G, info) : Tuple[nx.DiGraph, Dict[str, NodeInfo]]
        The dependency graph and node info map.
    """
    dists = list(importlib_metadata.distributions())
    by_name: Dict[str, importlib_metadata.Distribution] = {}
    for d in dists:
        try:
            nm = d.metadata["Name"]
        except Exception:
            continue
        if not nm:
            continue
        by_name[canonicalize_name(nm)] = d

    def get_info(canon_name: str) -> NodeInfo:
        d = by_name.get(canon_name)
        if not d:
            return NodeInfo(canon_name, f"{canon_name} (missing)", False)
        disp = d.metadata.get("Name", canon_name)
        ver = d.version or "?"
        return NodeInfo(canon_name, f"{disp}=={ver}", True)

    G: nx.DiGraph = nx.DiGraph()
    info: Dict[str, NodeInfo] = {}

    from collections import deque

    for r in roots:
        if cancel_flag and cancel_flag[0]:
            break
        root = canonicalize_name(r)
        if root not in info:
            info[root] = get_info(root)
            G.add_node(root)

        q = deque([(root, 0)])
        seen_depth: Dict[str, int] = {root: 0}

        while q:
            if cancel_flag and cancel_flag[0]:
                break
            u, lv = q.popleft()
            if depth_limit >= 0 and lv >= depth_limit:
                continue

            dist = by_name.get(u)
            if not dist:
                continue

            for req in _safe_parse_requires(dist.requires):
                v = canonicalize_name(req.name)
                if v not in info:
                    info[v] = get_info(v)
                    G.add_node(v)
                G.add_edge(u, v, req=str(req), spec=str(req.specifier) if req.specifier else "")

                nxt_lv = lv + 1
                if v not in seen_depth or nxt_lv < seen_depth[v]:
                    seen_depth[v] = nxt_lv
                    q.append((v, nxt_lv))

    return G, info


def build_dependents_graph_bfs(
    targets: Sequence[str],
    depth_limit: int = -1,
    cancel_flag: Optional[List[bool]] = None,
) -> Tuple[nx.DiGraph, Dict[str, NodeInfo]]:
    """Build a directed graph of DEPENDENTS using BFS.

    We traverse “who depends on me?” while keeping edge direction consistent:
    parent→child means “parent depends on child”.

    Parameters
    ----------
    targets : Sequence[str]
        Target packages for which to find dependents.
    depth_limit : int, default -1
        -1 = unlimited, 0 = only direct dependents, N = depth N.
    cancel_flag : Optional[List[bool]]
        Cooperative cancel flag.

    Returns
    -------
    (G, info) : Tuple[nx.DiGraph, Dict[str, NodeInfo]]
        The dependents graph and node info map.
    """
    dists = list(importlib_metadata.distributions())
    by_name: Dict[str, importlib_metadata.Distribution] = {}
    for d in dists:
        try:
            nm = d.metadata["Name"]
        except Exception:
            continue
        if not nm:
            continue
        by_name[canonicalize_name(nm)] = d

    def get_info(canon_name: str) -> NodeInfo:
        d = by_name.get(canon_name)
        if not d:
            return NodeInfo(canon_name, f"{canon_name} (missing)", False)
        disp = d.metadata.get("Name", canon_name)
        ver = d.version or "?"
        return NodeInfo(canon_name, f"{disp}=={ver}", True)

    from collections import defaultdict, deque
    # Reverse adjacency: child -> {parents that require child}
    rdeps: Dict[str, Set[str]] = defaultdict(set)
    edge_req: Dict[Tuple[str, str], str] = {}
    for parent, dist in by_name.items():
        for req in _safe_parse_requires(dist.requires):
            child = canonicalize_name(req.name)
            rdeps[child].add(parent)
            edge_req[(parent, child)] = str(req)

    G: nx.DiGraph = nx.DiGraph()
    info: Dict[str, NodeInfo] = {}

    for t in targets:
        if cancel_flag and cancel_flag[0]:
            break
        root = canonicalize_name(t)
        if root not in info:
            info[root] = get_info(root)
            G.add_node(root)

        q = deque([(root, 0)])
        seen_depth: Dict[str, int] = {root: 0}

        while q:
            if cancel_flag and cancel_flag[0]:
                break
            u, lv = q.popleft()
            if depth_limit >= 0 and lv >= depth_limit:
                continue

            for parent in rdeps.get(u, set()):
                if parent not in info:
                    info[parent] = get_info(parent)
                    G.add_node(parent)
                G.add_edge(parent, u, req=edge_req.get((parent, u), ""))

                nxt_lv = lv + 1
                if parent not in seen_depth or nxt_lv < seen_depth[parent]:
                    seen_depth[parent] = nxt_lv
                    q.append((parent, nxt_lv))

    return G, info


# --------------------------- Shared layout helper ----------------------

def compute_layout_for_graph(
    G: nx.DiGraph,
    backend: str,
    layout: str,
    fa2_iter: int,
    fa2_scaling: float,
    fa2_gravity: float,
    spring_iter: int,
) -> PosDict:
    """Compute 2D positions using the selected backend/layout.

    Parameters
    ----------
    G : nx.DiGraph
        Graph to layout.
    backend : str
        "CPU" or "GPU".
    layout : str
        "spring", "kamada-kawai", or "force-atlas2" (GPU).
    fa2_iter : int
        Iterations for GPU ForceAtlas2.
    fa2_scaling : float
        Scaling ratio for GPU ForceAtlas2.
    fa2_gravity : float
        Gravity for GPU ForceAtlas2.
    spring_iter : int
        Iterations for NetworkX spring layout.

    Returns
    -------
    PosDict
        Mapping node -> (x, y).
    """
    if backend == "GPU" and layout == "force-atlas2":
        try:
            import cudf  # type: ignore
            import cugraph  # type: ignore
        except Exception as e:
            raise RuntimeError("cuGraph/cuDF not available.") from e

        nodes = list(G.nodes())
        id_map: Dict[str, int] = {n: i for i, n in enumerate(nodes)}
        src, dst = [], []
        for u, v in G.edges():
            src.append(id_map[u]); dst.append(id_map[v])

        if not src:
            return {n: (0.0, 0.0) for n in nodes}

        gdf = cudf.DataFrame({"src": src, "dst": dst}).astype({"src": "int32", "dst": "int32"})
        # Undirected for layout stability
        G_cu = cugraph.Graph(directed=False)
        G_cu.from_cudf_edgelist(gdf, source="src", destination="dst", renumber=False)

        pos_df = cugraph.layout.force_atlas2(
            G_cu,
            max_iter=int(fa2_iter),
            outbound_attraction_distribution=True,
            lin_log_mode=False,
            prevent_overlapping=False,
            jitter_tolerance=1.0,
            scaling_ratio=float(fa2_scaling),
            gravity=float(fa2_gravity),
            strong_gravity_mode=False,
        )
        pos: PosDict = {}
        for row in pos_df.to_pandas().itertuples(index=False):
            label = nodes[int(row.vertex)]
            pos[label] = (float(row.x), float(row.y))

        # Place any isolated nodes not present in FA2 output
        if len(pos) < len(nodes):
            missing = [n for n in nodes if n not in pos]
            if pos:
                xs = [xy[0] for xy in pos.values()]
                ys = [xy[1] for xy in pos.values()]
                cx = sum(xs)/len(xs); cy = sum(ys)/len(ys)
                spread_x = (max(xs)-min(xs)) if len(xs) > 1 else 1.0
                spread_y = (max(ys)-min(ys)) if len(ys) > 1 else 1.0
                r = max(spread_x, spread_y); r = max(r*0.6, 1.0)
            else:
                cx = cy = 0.0; r = 1.0
            for i, n in enumerate(missing):
                angle = 2.0 * math.pi * i / len(missing)
                pos[n] = (cx + r*math.cos(angle), cy + r*math.sin(angle))
        return pos

    # CPU
    if layout == "kamada-kawai":
        return nx.kamada_kawai_layout(G)
    return nx.spring_layout(G, seed=42, iterations=int(spring_iter))


# --------------------------- Workers (QThread targets) -----------------

class GraphWorker(QObject):
    """Background worker that resolves graph (deps/dependents) and computes layout."""

    progress = pyqtSignal(int, str)                 # percent, message
    finished = pyqtSignal(object, object, object)   # G, info, pos
    failed = pyqtSignal(str)

    def __init__(self, roots: Sequence[str], depth: int, backend: str, layout: str,
                 fa2_iter: int, fa2_scaling: float, fa2_gravity: float,
                 spring_iter: int, mode: str):
        """Store configuration for graph construction and layout."""
        super().__init__()
        self.roots = list(roots)
        self.depth = depth
        self.backend = backend
        self.layout = layout
        self.fa2_iter = fa2_iter
        self.fa2_scaling = fa2_scaling
        self.fa2_gravity = fa2_gravity
        self.spring_iter = spring_iter
        self.mode = mode
        self.cancel_flag: List[bool] = [False]

    def cancel(self) -> None:
        """Cooperatively request cancellation."""
        self.cancel_flag[0] = True

    def run(self) -> None:
        """Resolve graph, mark cycles, compute layout, and emit results."""
        try:
            self.progress.emit(5, "Resolving dependencies...")
            if self.mode == "rdeps":
                G, info = build_dependents_graph_bfs(self.roots, self.depth, self.cancel_flag)
            else:
                G, info = build_dependency_graph_bfs(self.roots, self.depth, self.cancel_flag)

            if self.cancel_flag[0]:
                self.failed.emit("Cancelled."); return

            self.progress.emit(50, "Detecting cycles...")
            try:
                cycles = list(nx.simple_cycles(G))
                cycle_nodes = set(n for c in cycles for n in c)
            except Exception:
                cycle_nodes = set()
            nx.set_node_attributes(G, {n: (n in cycle_nodes) for n in G.nodes()}, "in_cycle")

            self.progress.emit(70, "Computing layout...")
            pos = compute_layout_for_graph(
                G, self.backend, self.layout, self.fa2_iter, self.fa2_scaling, self.fa2_gravity, self.spring_iter
            )
            if self.cancel_flag[0]:
                self.failed.emit("Cancelled."); return

            self.progress.emit(100, "Done.")
            self.finished.emit(G, info, pos)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class LayoutWorker(QObject):
    """Background worker that recomputes layout for an existing graph."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)   # pos
    failed = pyqtSignal(str)

    def __init__(self, G: nx.DiGraph, backend: str, layout: str,
                 fa2_iter: int, fa2_scaling: float, fa2_gravity: float,
                 spring_iter: int):
        """Store parameters and reference to the graph to (re)layout."""
        super().__init__()
        self.G = G
        self.backend = backend
        self.layout = layout
        self.fa2_iter = fa2_iter
        self.fa2_scaling = fa2_scaling
        self.fa2_gravity = fa2_gravity
        self.spring_iter = spring_iter

    def run(self) -> None:
        """Compute positions only and emit results."""
        try:
            self.progress.emit(10, "Recomputing layout...")
            pos = compute_layout_for_graph(
                self.G, self.backend, self.layout,
                self.fa2_iter, self.fa2_scaling, self.fa2_gravity, self.spring_iter
            )
            self.progress.emit(100, "Done.")
            self.finished.emit(pos)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


# --------------------------- Canvas ---------------------------

class MplCanvas(FigureCanvasQTAgg):
    """A Matplotlib canvas embedded in Qt with mouse wheel zoom."""

    def __init__(self, parent: Optional[QWidget] = None):
        """Create a single-axes figure and hide axes by default."""
        fig = Figure(constrained_layout=True)
        super().__init__(fig)
        self.setParent(parent)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_axis_off()  # hide ticks/frames before anything is drawn
        self._zoom_factor = 1.2

    def wheelEvent(self, event):
        """Zoom the axes keeping the mouse position as focal point."""
        if self.ax is None:
            return

        x = event.position().x()
        y = event.position().y()

        inv = self.ax.transData.inverted()
        try:
            xdata, ydata = inv.transform((x, y))
        except Exception:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            xdata = (xlim[0] + xlim[1]) / 2
            ydata = (ylim[0] + ylim[1]) / 2

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        scale = 1 / self._zoom_factor if event.angleDelta().y() > 0 else self._zoom_factor

        new_xlim = [
            xdata - (xdata - xlim[0]) * scale,
            xdata + (xlim[1] - xdata) * scale,
        ]
        new_ylim = [
            ydata - (ydata - ylim[0]) * scale,
            ydata + (ylim[1] - ydata) * scale,
        ]

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.draw_idle()


# ---------------------------------------------------------------------------
# helpers: build dependency graph
# ---------------------------------------------------------------------------

def get_installed_distributions() -> Dict[str, im.Distribution]:
    dists: Dict[str, im.Distribution] = {}
    for dist in im.distributions():
        name = dist.metadata["Name"] or dist.metadata["Summary"] or dist.locate_file
        key = name.lower().replace("_", "-")
        dists[key] = dist
    return dists


def build_dep_graph(selected: Optional[List[str]] = None) -> nx.DiGraph:
    """
    Build a directed graph pkg -> its requirement.
    If `selected` is given, we BFS from those roots; otherwise include all.
    """
    dists = get_installed_distributions()
    G = nx.DiGraph()

    if not selected:
        # all packages
        for name, dist in dists.items():
            G.add_node(name)
            requires = dist.requires or []
            for req in requires:
                depname = req.split(";")[0].strip().split()[0].lower().replace("_", "-")
                G.add_edge(name, depname)
    else:
        # only those reachable from selected
        seen: Set[str] = set()
        q: List[str] = [nm.lower().replace("_", "-") for nm in selected]
        while q:
            cur = q.pop()
            if cur in seen:
                continue
            seen.add(cur)
            G.add_node(cur)
            dist = dists.get(cur)
            if not dist:
                continue
            requires = dist.requires or []
            for req in requires:
                depname = req.split(";")[0].strip().split()[0].lower().replace("_", "-")
                G.add_node(depname)
                G.add_edge(cur, depname)
                if depname not in seen:
                    q.append(depname)

    return G


# ---------------------------------------------------------------------------
# Layer 1: scalar field
# ---------------------------------------------------------------------------

def compute_depth_scalar(G: nx.DiGraph, roots: List[str]) -> Dict[str, float]:
    if not roots:
        if G.number_of_nodes() == 0:
            return {}
        roots = [list(G.nodes())[0]]
    from collections import deque
    depth = {n: math.inf for n in G.nodes()}
    q = deque()
    for r in roots:
        if r in depth:
            depth[r] = 0.0
            q.append(r)
    while q:
        u = q.popleft()
        du = depth[u]
        for v in G.successors(u):
            if depth[v] > du + 1:
                depth[v] = du + 1
                q.append(v)
    finite = [d for d in depth.values() if d < math.inf]
    maxd = max(finite) if finite else 0.0
    for n in depth:
        if depth[n] == math.inf:
            depth[n] = maxd + 1.0
    return depth


def compute_outdeg_scalar(G: nx.DiGraph) -> Dict[str, float]:
    return {n: float(G.out_degree(n)) for n in G.nodes()}


def compute_betweenness_scalar(G: nx.DiGraph) -> Dict[str, float]:
    if G.number_of_nodes() == 0:
        return {}
    # depend graphs are not huge → exact is OK; undirected is enough for H0
    bc = nx.betweenness_centrality(G.to_undirected(), normalized=True)
    return {n: float(bc.get(n, 0.0)) for n in G.nodes()}


def compute_scalar_field(G: nx.DiGraph, mode: str, roots: List[str]) -> Dict[str, float]:
    if mode == "depth-from-roots":
        return compute_depth_scalar(G, roots)
    elif mode == "out-degree":
        return compute_outdeg_scalar(G)
    elif mode == "betweenness":
        return compute_betweenness_scalar(G)
    return {n: 0.0 for n in G.nodes()}


# ---------------------------------------------------------------------------
# Layer 2: 0D PH (very small)
# ---------------------------------------------------------------------------

def compute_h0_barcodes(G: nx.DiGraph, scalars: Dict[str, float]) -> List[Tuple[float, float]]:
    """
    Sublevel filtration on an undirected version of G.
    Each node is born at its scalar value.
    When two components get connected, the younger (bigger birth) dies.
    Return list of (birth, death); "infinite" deaths will be max+δ.
    """
    if G.number_of_nodes() == 0 or not scalars:
        return []
    H = G.to_undirected()
    # union-find
    parent = {n: n for n in H.nodes()}
    birth = {n: scalars[n] for n in H.nodes()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b, level, bars):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # older has smaller birth
        if birth[ra] <= birth[rb]:
            older, younger = ra, rb
        else:
            older, younger = rb, ra
        # younger dies at this level
        bars.append((birth[younger], level))
        parent[younger] = older

    # sort nodes by scalar
    nodes_sorted = sorted(H.nodes(), key=lambda x: scalars[x])
    bars: List[Tuple[float, float]] = []
    active: Set[str] = set()
    for n in nodes_sorted:
        active.add(n)
        lvl = scalars[n]
        for nb in H.neighbors(n):
            if nb in active:
                union(n, nb, lvl, bars)

    maxv = max(scalars.values())
    inf_death = maxv + (0.1 * maxv if maxv > 0 else 1.0)
    # remaining reps
    reps = set(find(n) for n in H.nodes())
    for r in reps:
        bars.append((birth[r], inf_death))
    return bars


# ---------------------------------------------------------------------------
# Layer 3: Mapper-like view (interval cover + CC)
# ---------------------------------------------------------------------------

def build_mapper_graph(
    G: nx.DiGraph,
    scalars: Dict[str, float],
    num_intervals: int = 6,
    overlap: float = 0.25,
) -> nx.Graph:
    """
    Super lightweight Mapper:
    - lens: given by `scalars`
    - cover: num_intervals, with given overlap (0..0.9)
    - clustering: within each interval, take induced subgraph and split into CC
    - result: nodes = (interval_id, component_id), edges = "non-empty intersection of clusters"
    """
    M = nx.Graph()
    if not G or not scalars:
        return M

    vals = np.array(list(scalars.values()), dtype=float)
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmin == vmax:
        # all the same → single node
        M.add_node((0, 0), members=list(G.nodes()))
        return M

    total = vmax - vmin
    step = total / num_intervals
    ov = max(0.0, min(overlap, 0.9))  # clamp
    intervals = []
    for i in range(num_intervals):
        start = vmin + i * step
        end = start + step
        # apply overlap
        start = start - ov * step
        end = end + ov * step
        intervals.append((start, end))

    # for each interval, pick nodes, split into CC -> cluster nodes
    cluster_nodes: Dict[Tuple[int, int], Set[str]] = {}
    for idx, (a, b) in enumerate(intervals):
        # pick nodes whose value is in [a, b]
        bucket = [n for n, v in scalars.items() if a <= v <= b]
        if not bucket:
            continue
        # induced subgraph
        H = G.to_undirected().subgraph(bucket).copy()
        cc_list = list(nx.connected_components(H))
        for j, comp in enumerate(cc_list):
            node_id = (idx, j)
            M.add_node(node_id, members=comp, interval=idx)
            cluster_nodes[node_id] = set(comp)

    # connect clusters if they share at least one original node
    node_ids = list(cluster_nodes.keys())
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            a_id, b_id = node_ids[i], node_ids[j]
            if cluster_nodes[a_id] & cluster_nodes[b_id]:
                M.add_edge(a_id, b_id)

    return M


# ---------------------------------------------------------------------------
# Qt Widgets
# ---------------------------------------------------------------------------


class TDAPanel(QWidget):
    """Right-side small panel for scalar + H0 barcode."""
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)

        title = QLabel("TDA (H₀) — experimental")
        title.setStyleSheet("font-weight: 600;")
        v.addWidget(title)

        # scalar selector
        h = QHBoxLayout()
        h.addWidget(QLabel("Scalar:"))
        self.combo_scalar = QComboBox()
        self.combo_scalar.addItems(["depth-from-roots", "out-degree", "betweenness"])
        h.addWidget(self.combo_scalar, 1)
        v.addLayout(h)

        # canvas
        self.fig = Figure(figsize=(3.5, 1.7), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        v.addWidget(self.canvas, 1)

        self.G: Optional[nx.DiGraph] = None
        self.roots: List[str] = []
        self.scalars: Dict[str, float] = {}

        self.combo_scalar.currentIndexChanged.connect(self.redraw)

    def set_graph(self, G: nx.DiGraph, roots: List[str]):
        self.G = G
        self.roots = roots
        self.redraw()

    def redraw(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_title("H₀ barcode")
        if self.G is None or self.G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "no graph", ha="center", va="center")
            self.canvas.draw_idle()
            return

        mode = self.combo_scalar.currentText()
        scalars = compute_scalar_field(self.G, mode, self.roots)
        self.scalars = scalars
        bars = compute_h0_barcodes(self.G, scalars)
        if not bars:
            ax.text(0.5, 0.5, "no bars", ha="center", va="center")
            self.canvas.draw_idle()
            return

        # sort by birth
        bars = sorted(bars, key=lambda x: (x[0], x[1]))
        for i, (b, d) in enumerate(bars):
            ax.hlines(i, b, d, linewidth=2)
            ax.plot([b], [i], "o", markersize=3)
        ax.set_xlabel("filtration")
        ax.set_ylabel("component")
        ax.set_ylim(-1, len(bars) + 1)
        self.canvas.draw_idle()


class MapperPanel(QWidget):
    """Right-side panel for Mapper-like view."""
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Mapper-like view")
        title.setStyleSheet("font-weight: 600;")
        v.addWidget(title)

        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Scalar (lens):"))
        self.combo_scalar = QComboBox()
        self.combo_scalar.addItems(["depth-from-roots", "out-degree", "betweenness"])
        h1.addWidget(self.combo_scalar, 1)
        v.addLayout(h1)

        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Intervals:"))
        self.spin_intervals = QSpinBox()
        self.spin_intervals.setRange(2, 24)
        self.spin_intervals.setValue(6)
        h2.addWidget(self.spin_intervals)
        h2.addWidget(QLabel("Overlap:"))
        self.dspin_overlap = QDoubleSpinBox()
        self.dspin_overlap.setRange(0.0, 0.9)
        self.dspin_overlap.setSingleStep(0.05)
        self.dspin_overlap.setValue(0.25)
        h2.addWidget(self.dspin_overlap)
        v.addLayout(h2)

        self.fig = Figure(figsize=(3.5, 2.0), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        v.addWidget(self.canvas, 1)

        self.G: Optional[nx.DiGraph] = None
        self.roots: List[str] = []

        self.combo_scalar.currentIndexChanged.connect(self.redraw)
        self.spin_intervals.valueChanged.connect(self.redraw)
        self.dspin_overlap.valueChanged.connect(self.redraw)

    def set_graph(self, G: nx.DiGraph, roots: List[str]):
        self.G = G
        self.roots = roots
        self.redraw()

    def redraw(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        if self.G is None or self.G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "no graph", ha="center", va="center")
            self.canvas.draw_idle()
            return

        mode = self.combo_scalar.currentText()
        scalars = compute_scalar_field(self.G, mode, self.roots)
        mi = self.spin_intervals.value()
        ov = self.dspin_overlap.value()

        M = build_mapper_graph(self.G, scalars, num_intervals=mi, overlap=ov)
        if M.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "empty cover", ha="center", va="center")
            self.canvas.draw_idle()
            return

        # layout by interval (x) and component (y)
        pos = {}
        for (interval, comp_id) in M.nodes():
            x = interval
            y = -comp_id
            pos[(interval, comp_id)] = (x, y)

        nx.draw_networkx(
            M,
            pos=pos,
            ax=ax,
            with_labels=False,
            node_size=200,
            node_color=[M.nodes[n].get("interval", 0) for n in M.nodes()],
        )
        ax.set_title(f"Mapper nodes: {M.number_of_nodes()}")
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw_idle()


# --------------------------- Main Window -------------------------------

class MainWindow(QMainWindow):
    """Main window: controls, worker orchestration, clustering, rendering, logging, and TDA/Mapper rendering."""

    def __init__(self):
        """Construct UI, connect signals, and load package table."""
        super().__init__()
        self.setWindowTitle("DepAtlas: Python Package Dependency Graph")
        self.resize(1540, 1080)

        # Left: controls + table + missing list + log
        left = QWidget(self)
        left_v = QVBoxLayout(left)

        # Filter row
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self.edit_filter = QLineEdit()
        self.edit_filter.setPlaceholderText("Type to filter packages (case-insensitive)")
        filter_row.addWidget(self.edit_filter, 1)
        btn_clear_filter = QPushButton("Clear")
        filter_row.addWidget(btn_clear_filter)
        left_v.addLayout(filter_row)

        # Table: checkbox + name + version
        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(["Select", "Package", "Version"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        left_v.addWidget(self.table, 2)

        # Controls box
        ctrl_box = QGroupBox("Graph Controls")
        ctrl = QVBoxLayout(ctrl_box)
        ctrl.setSpacing(6)

        # Row A: Depth, Backend, Layout, Mode
        rowA = QHBoxLayout()
        rowA.addWidget(QLabel("Depth:"))
        self.spin_depth = QSpinBox()
        self.spin_depth.setRange(-1, 32)
        self.spin_depth.setValue(-1)
        self.spin_depth.setToolTip("-1 = unlimited, 0 = direct only, N = depth N")
        rowA.addWidget(self.spin_depth)

        rowA.addWidget(QLabel("Backend:"))
        self.combo_backend = QComboBox()
        self.combo_backend.addItems(["CPU", "GPU"])
        rowA.addWidget(self.combo_backend)

        rowA.addWidget(QLabel("Layout:"))
        self.combo_layout = QComboBox()
        self._populate_layout_options()
        rowA.addWidget(self.combo_layout)

        rowA.addWidget(QLabel("Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Dependencies", "Dependents"])
        self.combo_mode.setCurrentText("Dependencies")
        rowA.addWidget(self.combo_mode)

        rowA.addStretch(1)
        ctrl.addLayout(rowA)

        # Row B: Spring iters, FA2 iters, FA2 scaling, FA2 gravity
        rowB = QHBoxLayout()
        rowB.addWidget(QLabel("Spring iters:"))
        self.spin_spring_iter = QSpinBox()
        self.spin_spring_iter.setRange(10, 5000)
        self.spin_spring_iter.setValue(50)
        rowB.addWidget(self.spin_spring_iter)

        rowB.addWidget(QLabel("FA2 iters:"))
        self.spin_fa2_iter = QSpinBox()
        self.spin_fa2_iter.setRange(50, 5000)
        self.spin_fa2_iter.setValue(500)
        rowB.addWidget(self.spin_fa2_iter)

        rowB.addWidget(QLabel("FA2 scaling:"))
        self.spin_fa2_scaling = QSpinBox()
        self.spin_fa2_scaling.setRange(1, 50)
        self.spin_fa2_scaling.setValue(2)
        rowB.addWidget(self.spin_fa2_scaling)

        rowB.addWidget(QLabel("FA2 gravity:"))
        self.spin_fa2_gravity = QSpinBox()
        self.spin_fa2_gravity.setRange(0, 50)
        self.spin_fa2_gravity.setValue(1)
        rowB.addWidget(self.spin_fa2_gravity)

        rowB.addStretch(1)
        ctrl.addLayout(rowB)

        # Row C: Colormap, Arrows, Node size, Font size
        rowC = QHBoxLayout()
        rowC.addWidget(QLabel("Colormap:"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(["Spectral", "viridis"])
        self.combo_cmap.setCurrentText("Spectral")
        rowC.addWidget(self.combo_cmap)

        self.chk_arrows = QCheckBox("Arrows")
        self.chk_arrows.setChecked(True)
        rowC.addWidget(self.chk_arrows)

        rowC.addWidget(QLabel("Node size:"))
        self.spin_node_size = QSpinBox()
        self.spin_node_size.setRange(10, 4000)
        self.spin_node_size.setValue(100)
        rowC.addWidget(self.spin_node_size)

        rowC.addWidget(QLabel("Font size:"))
        self.spin_font = QSpinBox()
        self.spin_font.setRange(4, 24)
        self.spin_font.setValue(9)
        rowC.addWidget(self.spin_font)

        rowC.addStretch(1)
        ctrl.addLayout(rowC)

        # Row C2: Clustering controls
        rowC2 = QHBoxLayout()
        rowC2.addWidget(QLabel("Clustering:"))
        self.combo_cluster = QComboBox()
        self.combo_cluster.addItems(["None", "Spectral", "2D K-Means", "DBSCAN"])
        self.combo_cluster.setCurrentText("None")
        rowC2.addWidget(self.combo_cluster)

        rowC2.addWidget(QLabel("Clusters k:"))
        self.spin_clusters = QSpinBox()
        self.spin_clusters.setRange(2, 64)
        self.spin_clusters.setValue(8)
        rowC2.addWidget(self.spin_clusters)

        self.chk_pack = QCheckBox("Pack clusters")
        self.chk_pack.setChecked(True)
        rowC2.addWidget(self.chk_pack)

        rowC2.addStretch(1)
        ctrl.addLayout(rowC2)

        # Row C3: DBSCAN params
        rowC3 = QHBoxLayout()
        rowC3.addWidget(QLabel("DBSCAN eps:"))
        self.spin_db_eps = QDoubleSpinBox()
        self.spin_db_eps.setDecimals(6)
        self.spin_db_eps.setRange(1e-6, 1e6)
        self.spin_db_eps.setSingleStep(0.05)
        self.spin_db_eps.setValue(0.8)
        rowC3.addWidget(self.spin_db_eps)

        rowC3.addWidget(QLabel("min_cluster_size:"))
        self.spin_db_min = QSpinBox()
        self.spin_db_min.setRange(2, 9999)
        self.spin_db_min.setValue(5)
        rowC3.addWidget(self.spin_db_min)

        self.chk_db_autotune = QCheckBox("Auto-tune (DBCV)")
        self.chk_db_autotune.setChecked(False)
        rowC3.addStretch(1)
        rowC3.addWidget(self.chk_db_autotune)
        ctrl.addLayout(rowC3)

        # Row D: Export buttons
        rowD = QHBoxLayout()
        self.btn_export_png = QPushButton("Export PNG")
        self.btn_export_svg = QPushButton("Export SVG")
        rowD.addStretch(1)
        rowD.addWidget(self.btn_export_png)
        rowD.addWidget(self.btn_export_svg)
        ctrl.addLayout(rowD)

        # Row E: Refresh / Select All / Clear / Cancel / Show Graph
        rowE = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh (pip list)")
        self.btn_select_all = QPushButton("Select All")
        self.btn_clear_sel = QPushButton("Clear Selection")
        self.btn_show = QPushButton("Show Graph ▶")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)

        rowE.addWidget(self.btn_refresh)
        rowE.addWidget(self.btn_select_all)
        rowE.addWidget(self.btn_clear_sel)
        rowE.addStretch(1)
        rowE.addWidget(self.btn_cancel)
        rowE.addWidget(self.btn_show)
        ctrl.addLayout(rowE)

        left_v.addWidget(ctrl_box)

        # Missing list
        miss_box = QGroupBox("Missing dependencies")
        miss_v = QVBoxLayout(miss_box)
        self.txt_missing = QPlainTextEdit()
        self.txt_missing.setReadOnly(True)
        miss_v.addWidget(self.txt_missing)
        left_v.addWidget(miss_box, 1)

        # Log panel
        log_box = QGroupBox("Log")
        log_v = QVBoxLayout(log_box)
        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        log_v.addWidget(self.txt_log)
        left_v.addWidget(log_box, 1)

        # Right pane: canvas + TDA + Mapper + progress
        right = QWidget(self)
        right_v = QVBoxLayout(right)
        self.canvas = MplCanvas(self)
        right_v.addWidget(self.canvas, 3)

        # TDA panel (new)
        self.tda_panel = TDAPanel(self)
        right_v.addWidget(self.tda_panel, 1)

        # Mapper panel (new)
        self.mapper_panel = MapperPanel(self)
        right_v.addWidget(self.mapper_panel, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        right_v.addWidget(self.progress)

        # Root layout
        root = QWidget(self)
        root_h = QHBoxLayout(root)
        root_h.addWidget(left, 3)
        root_h.addWidget(right, 4)
        self.setCentralWidget(root)

        # State
        self._cbar = None        # colorbar handle
        self._legend = None      # cluster legend handle
        self.G: Optional[nx.DiGraph] = None
        self.info: Dict[str, NodeInfo] = {}
        self.pos: PosDict = {}
        self.root_names: List[str] = []
        self.mode: str = "deps"  # "deps" or "rdeps"
        self.cluster_labels: Dict[str, int] = {}
        self.cluster_palette: Dict[int, str] = {}
        self.dbscan_meta: Optional[Tuple[int, int, float, int, Optional[float], str]] = None

        # Connections
        self.btn_refresh.clicked.connect(self.populate_table_from_pip)
        self.btn_select_all.clicked.connect(self.select_all)
        self.btn_clear_sel.clicked.connect(self.clear_selection)
        self.btn_show.clicked.connect(self.show_graph)
        self.btn_cancel.clicked.connect(self.cancel_worker)
        self.edit_filter.textChanged.connect(self.apply_filter)
        btn_clear_filter.clicked.connect(self.edit_filter.clear)

        # Backend/Layout and layout params → auto re-layout
        self.combo_backend.currentIndexChanged.connect(self._on_backend_changed)
        self.combo_layout.currentIndexChanged.connect(self._on_layout_changed)
        self.spin_spring_iter.valueChanged.connect(self._on_layout_params_changed)
        self.spin_fa2_iter.valueChanged.connect(self._on_layout_params_changed)
        self.spin_fa2_scaling.valueChanged.connect(self._on_layout_params_changed)
        self.spin_fa2_gravity.valueChanged.connect(self._on_layout_params_changed)

        # Redraw triggers (style only)
        self.chk_arrows.stateChanged.connect(self._draw_graph)
        self.spin_node_size.valueChanged.connect(self._draw_graph)
        self.spin_font.valueChanged.connect(self._draw_graph)
        self.combo_cmap.currentIndexChanged.connect(self._draw_graph)
        self.combo_mode.currentIndexChanged.connect(self._draw_graph)

        # Clustering triggers
        self.combo_cluster.currentIndexChanged.connect(self._on_cluster_controls_changed)
        self.spin_clusters.valueChanged.connect(self._on_cluster_controls_changed)
        self.spin_db_eps.valueChanged.connect(self._on_cluster_controls_changed)
        self.spin_db_min.valueChanged.connect(self._on_cluster_controls_changed)
        self.chk_db_autotune.stateChanged.connect(self._on_cluster_controls_changed)
        self.chk_pack.stateChanged.connect(self._on_cluster_controls_changed)

        # Export
        self.btn_export_png.clicked.connect(self.export_png)
        self.btn_export_svg.clicked.connect(self.export_svg)

        # Style: highlight "Show Graph" button
        self._style_show_button()

        # Initial load
        self.populate_table_from_pip()
        self._enable_cluster_controls()  # set initial UI enable/disable

    # -------- Logging helper --------
    def _log(self, msg: str) -> None:
        if not hasattr(self, "txt_log"):
            return
        ts = datetime.now().strftime("%H:%M:%S")
        self.txt_log.appendPlainText(f"[{ts}] {msg}")

    # -------- Style helpers --------
    def _style_show_button(self) -> None:
        self.btn_show.setObjectName("btnShowGraph")
        self.setStyleSheet("""
            QPushButton#btnShowGraph {
                background-color: #8ED6FF;
                color: #0B2942;
                border: 1px solid #59B9F3;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: 600;
            }
            QPushButton#btnShowGraph:hover { background-color: #9BDCFF; }
            QPushButton#btnShowGraph:pressed { background-color: #7FCFFF; }
            QPushButton#btnShowGraph:disabled {
                background-color: #CFEEFC; color: #6B7A88; border-color: #CFEEFC;
            }
        """)
        self.btn_show.setMinimumHeight(32)
        self.btn_show.setCursor(Qt.CursorShape.PointingHandCursor)

    # -------- Package table --------
    def populate_table_from_pip(self) -> None:
        try:
            res = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True, text=True, check=True
            )
            pkgs = json.loads(res.stdout)
        except Exception as e:
            QMessageBox.critical(self, "pip error", f"Failed to run pip list:\n{e}")
            pkgs = []

        self.table.setRowCount(0)
        for p in sorted(pkgs, key=lambda x: x["name"].lower()):
            name = p.get("name", "")
            ver = p.get("version", "")
            if not name:
                continue
            row = self.table.rowCount()
            self.table.insertRow(row)
            cb = QCheckBox()
            cb.setChecked(False)
            self.table.setCellWidget(row, 0, cb)
            self.table.setItem(row, 1, QTableWidgetItem(name))
            vitem = QTableWidgetItem(ver)
            vitem.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 2, vitem)
        self.table.sortItems(1)
        self._log(f"[pip] listed {self.table.rowCount()} packages")

    def apply_filter(self) -> None:
        q = self.edit_filter.text().strip().lower()
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 1)
            name = name_item.text().lower() if name_item else ""
            self.table.setRowHidden(row, q not in name)

    def _iter_selected_packages(self) -> List[str]:
        names: List[str] = []
        for row in range(self.table.rowCount()):
            if self.table.isRowHidden(row):
                continue
            w = self.table.cellWidget(row, 0)
            if isinstance(w, QCheckBox) and w.isChecked():
                item = self.table.item(row, 1)
                if item:
                    names.append(item.text())
        return names

    def select_all(self) -> None:
        for row in range(self.table.rowCount()):
            if self.table.isRowHidden(row):
                continue
            w = self.table.cellWidget(row, 0)
            if isinstance(w, QCheckBox):
                w.setChecked(True)

    def clear_selection(self) -> None:
        for row in range(self.table.rowCount()):
            w = self.table.cellWidget(row, 0)
            if isinstance(w, QCheckBox):
                w.setChecked(False)

    # -------- Worker orchestration --------
    def _populate_layout_options(self) -> None:
        backend = self.combo_backend.currentText()
        self.combo_layout.blockSignals(True)
        self.combo_layout.clear()
        if backend == "GPU":
            self.combo_layout.addItems(["force-atlas2"])
        else:
            self.combo_layout.addItems(["spring", "kamada-kawai"])
        self.combo_layout.blockSignals(False)

    def show_graph(self) -> None:
        roots = self._iter_selected_packages()
        if not roots:
            QMessageBox.information(self, "No selection", "Please select at least one package.")
            return

        self.root_names = [canonicalize_name(r) for r in roots]
        depth = self.spin_depth.value()
        self.mode = "rdeps" if self.combo_mode.currentText().lower().startswith("dependents") else "deps"

        self.progress.setValue(0)
        self.progress.setFormat("Starting...")
        self.btn_show.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        self._log(f"[build] mode={self.mode}, depth={depth}, backend={self.combo_backend.currentText()}, "
                  f"layout={self.combo_layout.currentText()}, roots={len(roots)}")

        self.thread = QThread()
        self.worker = GraphWorker(
            roots=roots,
            depth=depth,
            backend=self.combo_backend.currentText(),
            layout=self.combo_layout.currentText(),
            fa2_iter=self.spin_fa2_iter.value(),
            fa2_scaling=float(self.spin_fa2_scaling.value()),
            fa2_gravity=float(self.spin_fa2_gravity.value()),
            spring_iter=self.spin_spring_iter.value(),
            mode=self.mode,
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_worker_progress)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.failed.connect(self.on_worker_failed)
        self.worker.finished.connect(lambda *_: self._cleanup_worker())
        self.worker.failed.connect(lambda *_: self._cleanup_worker())
        self.thread.start()

    def _start_layout_worker(self) -> None:
        if self.G is None:
            return
        self.progress.setValue(0)
        self.progress.setFormat("Re-layout...")
        self.btn_show.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        self._log(f"[layout] backend={self.combo_backend.currentText()}, layout={self.combo_layout.currentText()} "
                  f"(iters: spring={self.spin_spring_iter.value()}, fa2={self.spin_fa2_iter.value()})")

        self.lthread = QThread()
        self.lworker = LayoutWorker(
            G=self.G,
            backend=self.combo_backend.currentText(),
            layout=self.combo_layout.currentText(),
            fa2_iter=self.spin_fa2_iter.value(),
            fa2_scaling=float(self.spin_fa2_scaling.value()),
            fa2_gravity=float(self.spin_fa2_gravity.value()),
            spring_iter=self.spin_spring_iter.value(),
        )
        self.lworker.moveToThread(self.lthread)
        self.lthread.started.connect(self.lworker.run)
        self.lworker.progress.connect(self.on_worker_progress)
        self.lworker.finished.connect(self.on_layout_finished)
        self.lworker.failed.connect(self.on_worker_failed)
        self.lworker.finished.connect(lambda *_: self._cleanup_layout_worker())
        self.lworker.failed.connect(lambda *_: self._cleanup_layout_worker())
        self.lthread.start()

    def _cleanup_worker(self) -> None:
        try:
            self.thread.quit()
            self.thread.wait()
        except Exception:
            pass
        self.btn_show.setEnabled(True)
        self.btn_cancel.setEnabled(False)

    def _cleanup_layout_worker(self) -> None:
        try:
            self.lthread.quit()
            self.lthread.wait()
        except Exception:
            pass
        self.btn_show.setEnabled(True)
        self.btn_cancel.setEnabled(False)

    def cancel_worker(self) -> None:
        if hasattr(self, "worker") and getattr(self, "worker", None):
            try:
                self.worker.cancel()
            except Exception:
                pass
        self._log("[build/layout] cancellation requested")

    def on_worker_progress(self, pct: int, msg: str) -> None:
        self.progress.setValue(pct)
        self.progress.setFormat(msg)

    def on_worker_finished(self, G: nx.DiGraph, info: Dict[str, NodeInfo],
                           pos: PosDict) -> None:
        self._log(f"[ready] nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
        self.G = G
        self.info = info
        self.pos = pos
        self._update_missing_list()
        self._update_clusters()
        self._draw_graph()

        # --- TDA / Mapper update ---
        try:
            if hasattr(self, "tda_panel") and self.tda_panel is not None:
                self.tda_panel.set_graph(G, self.root_names)
                self._log("[tda] H₀ barcode updated.")
        except Exception as e:
            self._log(f"[tda] update failed: {e}")

        try:
            if hasattr(self, "mapper_panel") and self.mapper_panel is not None:
                self.mapper_panel.set_graph(G, self.root_names)
                self._log("[mapper] view updated.")
        except Exception as e:
            self._log(f"[mapper] update failed: {e}")

    def on_layout_finished(self, pos: PosDict) -> None:
        self.pos = pos
        if self.combo_cluster.currentText().lower().startswith(("2d", "dbscan")):
            self._update_clusters()
        self._draw_graph()

        # keep TDA/Mapper in sync (optional)
        if hasattr(self, "tda_panel") and self.tda_panel is not None and self.G is not None:
            self.tda_panel.set_graph(self.G, self.root_names)
        if hasattr(self, "mapper_panel") and self.mapper_panel is not None and self.G is not None:
            self.mapper_panel.set_graph(self.G, self.root_names)

    def on_worker_failed(self, error_msg: str) -> None:
        self._log(f"[error] {error_msg}")
        QMessageBox.warning(self, "Operation failed", error_msg)

    # -------- Auto re-layout triggers --------
    def _on_backend_changed(self) -> None:
        self._populate_layout_options()
        self._start_layout_worker()

    def _on_layout_changed(self) -> None:
        self._start_layout_worker()

    def _on_layout_params_changed(self) -> None:
        self._start_layout_worker()

    # -------- Clustering (strict-k + DBSCAN) --------

    def _enable_cluster_controls(self) -> None:
        """Enable/disable UI controls according to selected clustering method."""
        mode = self.combo_cluster.currentText().lower()
        is_db = mode == "dbscan"
        is_k = mode.startswith("2d")
        is_spec = mode == "spectral"
        # Enable degree colormap only when 'None'
        self.combo_cmap.setEnabled(mode == "none")
        # k only for Spectral / 2D K-Means
        self.spin_clusters.setEnabled(is_k or is_spec)
        # DBSCAN params only for DBSCAN
        for w in (self.spin_db_eps, self.spin_db_min, self.chk_db_autotune):
            w.setEnabled(is_db)

    def _on_cluster_controls_changed(self) -> None:
        """Recompute clustering when controls change; update UI and redraw."""
        self._enable_cluster_controls()
        self._update_clusters()
        self._draw_graph()

    def _spectral_strict_laplacian(self, G_und: nx.Graph, k: int) -> Dict[str, int]:
        """Exact-k spectral clustering via normalized Laplacian and recursive bisection.

        Works per connected component, using Fiedler vector bisection.
        Requires: numpy, scipy.sparse, scipy.sparse.linalg.eigsh
        """
        from scipy.sparse import csr_matrix, identity
        from scipy.sparse.linalg import eigsh

        clusters: List[List[str]] = [list(c) for c in nx.connected_components(G_und)]
        clusters = [c for c in clusters if len(c) > 0]
        if not clusters:
            return {}

        def bipartition(nodes: List[str]) -> Optional[Tuple[List[str], List[str]]]:
            if len(nodes) < 2:
                return None
            H = G_und.subgraph(nodes)
            order = list(H.nodes())
            try:
                A = nx.to_scipy_sparse_array(H, nodelist=order, dtype=float, weight=None, format="csr")
            except Exception:
                from networkx.convert_matrix import to_scipy_sparse_matrix
                A = to_scipy_sparse_matrix(H, nodelist=order, dtype=float, weight=None, format="csr")
            deg = (A.sum(axis=1)).A.ravel()
            if (deg == 0).all():
                return None
            with np.errstate(divide="ignore"):
                dinv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
            Dinv = csr_matrix(np.diag(dinv_sqrt))
            Lsym = identity(A.shape[0], dtype=float, format="csr") - Dinv @ A @ Dinv
            try:
                vals, vecs = eigsh(Lsym, k=2, which="SM")
            except Exception:
                return None
            fiedler = vecs[:, 1]
            thr = float(np.median(fiedler))
            left_mask = fiedler <= thr
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                left_mask = fiedler <= 0
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    return None
            left = [order[i] for i in range(len(order)) if left_mask[i]]
            right = [order[i] for i in range(len(order)) if right_mask[i]]
            return left, right

        import heapq
        heap = [(-len(c), i, c) for i, c in enumerate(clusters)]
        heapq.heapify(heap)

        while len(heap) < k:
            if not heap:
                break
            _, _, big = heapq.heappop(heap)
            split = bipartition(big)
            if split is None:
                heapq.heappush(heap, (-len(big), id(big), big))
                break
            a, b = split
            if len(a) == 0 or len(b) == 0:
                heapq.heappush(heap, (-len(big), id(big), big))
                break
            heapq.heappush(heap, (-len(a), id(a), a))
            heapq.heappush(heap, (-len(b), id(b), b))

        bins = [c for _, _, c in heap]

        def simple_split(lst: List[str]) -> Tuple[List[str], List[str]]:
            mid = max(1, len(lst) // 2)
            return lst[:mid], lst[mid:]

        while len(bins) < k:
            bins.sort(key=len, reverse=True)
            big = bins.pop(0)
            if len(big) < 2:
                bins.insert(0, big)
                break
            a, b = simple_split(big)
            bins.extend([a, b])

        if len(bins) > k:
            bins.sort(key=len)
            while len(bins) > k:
                x = bins.pop(0); y = bins.pop(0)
                bins.insert(0, x + y)

        labels: Dict[str, int] = {}
        for cid, bucket in enumerate(bins[:k]):
            for n in bucket:
                labels[n] = cid
        return labels

    def _cluster_kmeans_on_positions(self, nodes: List[str], k: int) -> Dict[str, int]:
        """Cluster nodes into exactly k groups using K-Means on 2D positions (NumPy-only)."""

        if any(n not in self.pos for n in nodes):
            self.pos = self._ensure_positions(self.G, self.pos)

        X = np.array([self.pos[n] for n in nodes], dtype=float)
        n = len(nodes)
        k = max(1, min(k, n))

        rng = np.random.default_rng(42)
        centers = []
        first_idx = rng.integers(0, n)
        centers.append(X[first_idx])

        for _ in range(1, k):
            d2 = np.min(((X[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(axis=2), axis=1)
            s = d2.sum()
            if s <= 0:
                idx = rng.integers(0, n)
            else:
                probs = d2 / s
                idx = rng.choice(n, p=probs)
            centers.append(X[idx])
        centers = np.array(centers)

        labels = np.zeros(n, dtype=int)
        for _ in range(50):
            dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = dists.argmin(axis=1)
            new_centers = centers.copy()
            for j in range(k):
                pts = X[new_labels == j]
                if len(pts) > 0:
                    new_centers[j] = pts.mean(axis=0)
                else:
                    far_idx = int(np.argmax(np.min(dists, axis=1)))
                    new_centers[j] = X[far_idx]
            if np.all(new_labels == labels) and np.allclose(new_centers, centers):
                labels = new_labels
                centers = new_centers
                break
            labels = new_labels
            centers = new_centers

        return {nodes[i]: int(labels[i]) for i in range(n)}

    def _cluster_merge_components_to_k(self, G: nx.Graph, k: int) -> Dict[str, int]:
        """Greedily merge connected components into exactly k clusters (size-balanced)."""
        comps = [list(c) for c in nx.connected_components(G)]
        total_nodes = sum(len(c) for c in comps)
        k = max(1, min(k, total_nodes))

        comps.sort(key=len, reverse=True)
        buckets = [[] for _ in range(k)]
        sizes = [0] * k
        for comp in comps:
            idx = sizes.index(min(sizes))
            buckets[idx].extend(comp)
            sizes[idx] += len(comp)

        labels: Dict[str, int] = {}
        for cid, bucket in enumerate(buckets):
            for n in bucket:
                labels[n] = cid
        return labels

    def _dbscan_on_positions(self, nodes: List[str], eps: float, min_size: int) -> Tuple[Dict[str, int], int, int]:
        """Run DBSCAN on current 2D positions.

        Parameters
        ----------
        nodes : List[str]
            Nodes to cluster.
        eps : float
            Neighborhood radius in 2D layout units.
        min_size : int
            Minimum cluster size (mapped to sklearn's min_samples).

        Returns
        -------
        (labels_map, n_clusters, n_noise)
            labels_map contains -1 for noise, cluster ids are remapped to 0..C-1.
        """
        try:
            from sklearn.cluster import DBSCAN
        except Exception as e:
            raise RuntimeError("scikit-learn is required for DBSCAN") from e

        if any(n not in self.pos for n in nodes):
            self.pos = self._ensure_positions(self.G, self.pos)
        X = np.array([self.pos[n] for n in nodes], dtype=float)

        if eps <= 0.0:
            raise ValueError("eps must be positive")

        db = DBSCAN(eps=float(eps), min_samples=int(max(2, min_size))).fit(X)
        labels = db.labels_.astype(int)
        # Remap labels >=0 to 0..C-1
        uniq = sorted([u for u in set(labels) if u >= 0])
        remap = {u: i for i, u in enumerate(uniq)}
        mapped = [remap.get(int(l), -1) for l in labels]
        n_clusters = len(uniq)
        n_noise = int((np.array(mapped) == -1).sum())
        labels_map = {nodes[i]: int(mapped[i]) for i in range(len(nodes))}
        return labels_map, n_clusters, n_noise

    def _nn_median_distance(self, X) -> float:
        """Estimate a typical neighborhood scale via median nearest-neighbor distance."""
        n = len(X)
        if n <= 1:
            return 1.0
        # brute-force pairwise distances; fine for typical graph sizes
        D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
        # set diagonal to large to exclude self
        np.fill_diagonal(D, np.inf)
        nn = np.min(D, axis=1)
        nn = nn[np.isfinite(nn)]
        if nn.size == 0:
            return 1.0
        return float(np.median(nn))

    def _dbcv_score(self, X, labels) -> Optional[Tuple[float, str]]:
        """Compute DBCV score if available; else silhouette score as fallback.

        Returns
        -------
        (score, metric_name) or None if cannot compute a valid score.
        """
        labels = np.asarray(labels, dtype=int)
        mask = labels >= 0
        if mask.sum() < 2 or len(set(labels[mask])) < 2:
            return None

        # Try hdbscan.validity.validity_index
        try:
            from hdbscan.validity import validity_index as hdbscan_validity_index
            score = float(hdbscan_validity_index(X, labels))
            return score, "DBCV"
        except Exception:
            pass

        # Fallback: silhouette
        try:
            from sklearn.metrics import silhouette_score
            score = float(silhouette_score(X[mask], labels[mask]))
            return score, "silhouette"
        except Exception:
            return None

    def _dbscan_autotune(
        self, nodes: List[str], base_min: int
    ) -> Tuple[Dict[str, int], int, int, float, int, Optional[float], str]:
        """Auto-tune DBSCAN with scale normalization, coverage constraint, and noise penalty.

        Strategy
        --------
        - Normalize coordinates so that median 1-NN distance ≈ 1 (scale invariance).
        - Explore relative eps in normalized space: {0.8, 1.0, 1.2, 1.5, 2.0, 2.5}.
        - Try min_cluster_size in {base_min//2, base_min, 2*base_min} (≥ 2).
        - Score via DBCV (hdbscan) if available, else silhouette.
        - Penalize excessive noise: effective_score = score * sqrt(clustered_fraction).
        - Enforce minimum coverage (≥ 30%); skip candidates below this threshold.

        Returns
        -------
        (labels_map, n_clusters, n_noise, best_eps, best_min, best_score, metric_name)
        """
        try:
            from sklearn.cluster import DBSCAN  # noqa: F401
        except Exception as e:
            raise RuntimeError("scikit-learn is required for DBSCAN auto-tune") from e

        # Prepare data (original positions)
        if any(n not in self.pos for n in nodes):
            self.pos = self._ensure_positions(self.G, self.pos)
        X = np.array([self.pos[n] for n in nodes], dtype=float)
        n = len(nodes)

        # Scale normalization for evaluation
        d_med = self._nn_median_distance(X)
        s = (d_med if math.isfinite(d_med) and d_med > 0 else 1.0)
        Xs = X / s  # normalized space (typical NN distance ≈ 1)

        # Relative eps candidates in normalized space
        eps_grid_scaled = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
        mins = sorted(set([max(2, base_min), max(2, base_min // 2), max(2, base_min * 2)]))

        MIN_COVERAGE = 0.30  # require ≥30% of nodes not noise

        def effective_key(score: Optional[float], n_clusters: int, n_noise: int) -> Tuple[float, int, int]:
            """Combine score with a penalty for noise to rank candidates."""
            clustered_fraction = (n - n_noise) / max(1, n)
            if score is None:
                eff = -1e9
            else:
                eff = float(score) * (clustered_fraction ** 0.5)
            return (eff, n_clusters, -n_noise)

        best = None  # (key, labels_map, n_clusters, n_noise, eps_unscaled, min_size, raw_score, metric, coverage)

        for eps_s in eps_grid_scaled:
            eps_unscaled = float(eps_s * s)  # DBSCAN uses original-scale eps
            for ms in mins:
                try:
                    labels_map, n_clusters, n_noise = self._dbscan_on_positions(nodes, eps_unscaled, ms)
                except Exception:
                    continue

                labels_array = [labels_map[nm] for nm in nodes]
                coverage = (n - n_noise) / max(1, n)
                if coverage < MIN_COVERAGE:
                    continue  # discard too-noisy solutions before scoring

                score_info = self._dbcv_score(Xs, labels_array)
                if score_info is None:
                    key = effective_key(None, n_clusters, n_noise)
                    metric = "none"
                    raw_score = None
                else:
                    raw_score, metric = score_info
                    key = effective_key(raw_score, n_clusters, n_noise)

                if best is None or key > best[0]:
                    best = (key, labels_map, n_clusters, n_noise, eps_unscaled, int(ms), raw_score, metric, coverage)

        if best is None:
            raise RuntimeError("Auto-tuning failed to find a suitable DBSCAN configuration (try increasing eps/min size)")

        (_, labels_map, n_clusters, n_noise, eps, ms, raw, metric, coverage) = best
        return labels_map, n_clusters, n_noise, float(eps), int(ms), (None if raw is None else float(raw)), metric

    def _pack_layout_by_clusters(self) -> None:
        """Re-layout nodes so that clusters become spatially separated (“packed”).

        Steps
        -----
        1) Build a cluster-quotient graph (clusters as super-nodes; edge weight = inter-cluster edge count).
        2) Layout the cluster graph (spring) to get cluster centers.
        3) For each cluster, layout its induced subgraph locally (spring) to get internal coordinates.
        4) Scale/translate local coords to the cluster center; expand to reduce overlaps.
        5) Place noise (-1) nodes, if any, on an outer ring around cluster centers.
        """
        if not self.cluster_labels or self.G is None:
            return

        G = self.G
        labels = self.cluster_labels
        # cluster -> nodes
        cl2nodes: Dict[int, List[str]] = {}
        for n, cid in labels.items():
            cl2nodes.setdefault(cid, []).append(n)
        cids = sorted([c for c in cl2nodes.keys() if c >= 0])  # noise(-1) handled later

        if not cids:
            return

        # cluster-quotient graph
        C = nx.Graph()
        for cid in cids:
            C.add_node(cid, size=len(cl2nodes[cid]))
        for u, v in G.edges():
            cu, cv = labels.get(u, -1), labels.get(v, -1)
            if cu >= 0 and cv >= 0 and cu != cv:
                w = C.get_edge_data(cu, cv, {}).get("weight", 0) + 1
                C.add_edge(cu, cv, weight=w)

        # layout cluster graph (centers)
        Cpos = nx.spring_layout(C, seed=42, weight="weight", k=None, iterations=100)

        # local layouts per cluster
        new_pos: Dict[str, Tuple[float, float]] = {}
        sep = 6.0  # spacing factor between cluster centers
        for cid in cids:
            nodes = cl2nodes[cid]
            H = G.subgraph(nodes)
            local = nx.spring_layout(H, seed=42, iterations=50)
            # normalize local coords to radius=1 around origin
            loc = np.array([local[n] for n in nodes], dtype=float)
            if len(loc) == 0:
                continue
            loc -= loc.mean(axis=0, keepdims=True)
            r = np.linalg.norm(loc, axis=1).max()
            if not np.isfinite(r) or r == 0:
                r = 1.0
            loc /= r
            # cluster size → radius
            radius = 1.0 + 0.25 * math.sqrt(len(nodes))
            center = np.array(Cpos[cid]) * sep
            loc = center + loc * radius
            for i, n in enumerate(nodes):
                new_pos[n] = (float(loc[i, 0]), float(loc[i, 1]))

        # noise (-1) to outer ring
        noise_nodes = [n for n, c in labels.items() if c == -1]
        if noise_nodes:
            cx = np.mean([xy[0] for xy in new_pos.values()]) if new_pos else 0.0
            cy = np.mean([xy[1] for xy in new_pos.values()]) if new_pos else 0.0
            if new_pos:
                dmax = max(math.hypot(x - cx, y - cy) for x, y in new_pos.values())
            else:
                dmax = 3.0
            r = dmax + 2.0
            for i, n in enumerate(noise_nodes):
                ang = 2 * math.pi * i / max(1, len(noise_nodes))
                new_pos[n] = (cx + r * math.cos(ang), cy + r * math.sin(ang))

        self.pos = new_pos
        self._log("[layout] cluster packing applied")

    def _update_clusters(self) -> None:
        """Compute node→cluster mapping according to the clustering controls.

        - 'None': clears cluster labels (degree colormap applies).
        - '2D K-Means': strict k on current positions.
        - 'Spectral': strict Laplacian spectral → sklearn spectral → 2D K-Means → component merge.
        - 'DBSCAN': run with (eps, min_cluster_size) on 2D positions;
            if Auto-tune is on, grid search to maximize DBCV (or silhouette) with noise penalties.
            If DBSCAN yields <2 clusters, fallback to 2D K-Means(k).
        """
        clustering = self.combo_cluster.currentText().lower()
        self.combo_cmap.setEnabled(clustering == "none")

        self.cluster_labels = {}
        self.cluster_palette = {}
        self.dbscan_meta = None
        if self.G is None:
            return

        G_und = self.G.to_undirected()
        nodes = list(G_und.nodes())
        n = len(nodes)
        if n < 2:
            return

        if clustering == "none":
            return

        if clustering.startswith("2d"):
            k_req = self.spin_clusters.value()
            k = max(2, min(k_req, n))
            try:
                self.cluster_labels = self._cluster_kmeans_on_positions(nodes, k)
                self._log(f"[clustering] 2D K-Means (k={k}) succeeded")
            except Exception:
                self.cluster_labels = self._cluster_merge_components_to_k(G_und, k)
                self._log(f"[fallback] 2D K-Means failed → merged components to exactly k={k}")

        elif clustering == "spectral":
            k_req = self.spin_clusters.value()
            k = max(2, min(k_req, n))
            try:
                labels_map = self._spectral_strict_laplacian(G_und, k)
                uniq = sorted(set(labels_map.values()))
                if labels_map and len(uniq) == k:
                    self.cluster_labels = labels_map
                    self._log(f"[clustering] Strict Laplacian spectral (k={k}) succeeded")
                else:
                    raise RuntimeError("strict spectral returned wrong cluster count")
            except Exception:
                spectral_ok = False
                try:
                    from scipy import sparse
                    from sklearn.cluster import SpectralClustering
                    try:
                        A = nx.to_scipy_sparse_array(G_und, nodelist=nodes, dtype=float, weight=None, format="csr")
                    except Exception:
                        from networkx.convert_matrix import to_scipy_sparse_matrix
                        A = to_scipy_sparse_matrix(G_und, nodelist=nodes, dtype=float, weight=None, format="csr")
                    deg = (A.sum(axis=1)).A.ravel()
                    if (deg == 0).any():
                        A = A + sparse.identity(n, dtype=float, format="csr") * 1e-6
                    sc = SpectralClustering(
                        n_clusters=k, affinity="precomputed",
                        assign_labels="kmeans", random_state=42, n_init=10,
                    )
                    labels = sc.fit_predict(A)
                    self.cluster_labels = {nodes[i]: int(labels[i]) for i in range(n)}
                    spectral_ok = True
                    self._log(f"[clustering] sklearn Spectral (k={k}) succeeded")
                except Exception:
                    spectral_ok = False

                if not spectral_ok:
                    try:
                        self.cluster_labels = self._cluster_kmeans_on_positions(nodes, k)
                        self._log(f"[fallback] Spectral failed/unavailable → using 2D K-Means (k={k})")
                    except Exception:
                        self.cluster_labels = self._cluster_merge_components_to_k(G_und, k)
                        self._log(f"[fallback] Clustering libs unavailable → merged components to exactly k={k}")

        elif clustering == "dbscan":
            base_eps = float(self.spin_db_eps.value())
            base_min = int(self.spin_db_min.value())
            try:
                if self.chk_db_autotune.isChecked():
                    labels_map, n_clusters, n_noise, best_eps, best_min, score, metric = self._dbscan_autotune(nodes, base_min)
                    self.cluster_labels = labels_map
                    self.dbscan_meta = (n_clusters, n_noise, best_eps, best_min, score, metric)
                    # Coverage and effective score for logging
                    total = max(1, self.G.number_of_nodes())
                    coverage = 1.0 - (n_noise / total)
                    eff = None
                    if score is not None:
                        eff = float(score) * (coverage ** 0.5)

                    # Detailed logs including final parameters and coverage/effective score
                    self._log(f"[dbcv] auto-tuned DBSCAN → eps={best_eps:.6g}, min={best_min}, "
                              f"clusters={n_clusters}, noise={n_noise}, "
                              f"score={score if score is not None else 'NA'} ({metric})")
                    self._log(f"[dbcv] final parameters: eps={best_eps:.6g}, min_cluster_size={best_min}")
                    if score is not None and eff is not None:
                        self._log(f"[dbcv] coverage={coverage:.2%}, effective_score={eff:.3f}")
                else:
                    labels_map, n_clusters, n_noise = self._dbscan_on_positions(nodes, base_eps, base_min)
                    self.cluster_labels = labels_map
                    self.dbscan_meta = (n_clusters, n_noise, base_eps, base_min, None, "manual")
                    total = max(1, self.G.number_of_nodes())
                    coverage = 1.0 - (n_noise / total)
                    self._log(f"[clustering] DBSCAN (eps={base_eps:.6g}, min={base_min}) → "
                              f"clusters={n_clusters}, noise={n_noise}, coverage={coverage:.2%}")
            except Exception as e:
                self._log(f"[error] DBSCAN failed: {e}")
                # fallback to 2D K-Means(k)
                k_req = self.spin_clusters.value()
                k = max(2, min(k_req, n))
                try:
                    self.cluster_labels = self._cluster_kmeans_on_positions(nodes, k)
                    self._log(f"[fallback] DBSCAN failed → using 2D K-Means (k={k})")
                except Exception:
                    self.cluster_labels = self._cluster_merge_components_to_k(G_und, k)
                    self._log(f"[fallback] Both DBSCAN and 2D K-Means failed → merged components to k={k}")

            # If DBSCAN produced <2 clusters, fallback to 2D K-Means(k)
            if self.cluster_labels:
                uniq = [c for c in set(self.cluster_labels.values()) if c >= 0]
                if len(uniq) < 2:
                    k_req = self.spin_clusters.value()
                    k = max(2, min(k_req, n))
                    self._log(f"[fallback] DBSCAN produced <2 clusters → 2D K-Means (k={k})")
                    try:
                        self.cluster_labels = self._cluster_kmeans_on_positions(nodes, k)
                    except Exception:
                        self.cluster_labels = self._cluster_merge_components_to_k(G_und, k)
                        self._log(f"[fallback] 2D K-Means failed → merged components to exactly k={k}")

        # After clustering, optionally pack clusters into spatial blobs
        if self.cluster_labels and self.chk_pack.isChecked():
            self._pack_layout_by_clusters()

        # Build categorical palette (tab20) for any clustering mode (including DBSCAN)
        if self.cluster_labels:
            cmap = matplotlib.colormaps.get_cmap("tab20")
            uniq = sorted({cid for cid in self.cluster_labels.values() if cid >= 0})
            self.cluster_palette = {cid: mcolors.to_hex(cmap(cid % cmap.N)) for cid in uniq}

    # -------- Drawing --------

    def _selected_cmap(self):
        """Return a continuous Matplotlib colormap based on UI selection."""
        name = self.combo_cmap.currentText()
        if name.lower() == "spectral":
            return matplotlib.colormaps.get_cmap("Spectral")
        return matplotlib.colormaps.get_cmap("viridis")

    def _ensure_positions(self, G: nx.DiGraph, pos: PosDict) -> PosDict:
        """Ensure all nodes have coordinates; place missing ones on a circle."""
        out = dict(pos)
        missing_nodes = [n for n in G.nodes() if n not in out]
        if not missing_nodes:
            return out

        if out:
            xs = [xy[0] for xy in out.values()]
            ys = [xy[1] for xy in out.values()]
            cx = sum(xs)/len(xs); cy = sum(ys)/len(ys)
            spread_x = (max(xs)-min(xs)) if len(xs) > 1 else 1.0
            spread_y = (max(ys)-min(ys)) if len(ys) > 1 else 1.0
            r = max(spread_x, spread_y); r = max(r*0.6, 1.0)
        else:
            cx = cy = 0.0; r = 1.0

        for i, n in enumerate(missing_nodes):
            angle = 2.0 * math.pi * i / len(missing_nodes)
            out[n] = (cx + r*math.cos(angle), cy + r*math.sin(angle))
        return out

    def _compute_degree_colors(self, G: nx.DiGraph, cmap) -> Tuple[Dict[str, str], mcolors.Normalize]:
        """Compute per-node hex colors from total degree using a continuous colormap."""
        total_deg = dict(G.degree())
        if total_deg:
            vmin = min(total_deg.values()); vmax = max(total_deg.values())
        else:
            vmin = vmax = 0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        vrange = max(1e-9, float(vmax - vmin))

        def color_for(n: str) -> str:
            t = (total_deg.get(n, 0) - vmin) / vrange
            return mcolors.to_hex(cmap(t), keep_alpha=False)

        return ({n: color_for(n) for n in G.nodes()}, norm)

    def _draw_colorbar(self, norm: mcolors.Normalize, cmap) -> None:
        """Attach or refresh the continuous colorbar legend (degree)."""
        if getattr(self, "_legend", None) is not None:
            try: self._legend.remove()
            except Exception: pass
            self._legend = None
        if getattr(self, "_cbar", None) is not None:
            try: self._cbar.remove()
            except Exception: pass
            self._cbar = None
        self._cbar = self.canvas.figure.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=self.canvas.ax, fraction=0.046, pad=0.04
        )
        self._log("[legend] degree colorbar shown")
        self._cbar.set_label("Total degree (in + out)")

    def _draw_cluster_legend(self) -> None:
        """Attach or refresh the categorical cluster legend (includes 'Noise' if present)."""
        # remove colorbar when categorical legend is used
        if getattr(self, "_cbar", None) is not None:
            try: self._cbar.remove()
            except Exception: pass
            self._cbar = None
        if getattr(self, "_legend", None) is not None:
            try: self._legend.remove()
            except Exception: pass
            self._legend = None

        handles = [
            Patch(facecolor=color, edgecolor="#333333", label=f"Cluster {cid}")
            for cid, color in sorted(self.cluster_palette.items())
        ]
        # Add Noise if any node has label -1
        if any(cid == -1 for cid in self.cluster_labels.values()):
            handles.append(Patch(facecolor="#BDBDBD", edgecolor="#333333", label="Noise"))

        if handles:
            self._legend = self.canvas.ax.legend(
                handles=handles, title="Clusters", loc="upper right", frameon=True
            )
            self._log("[legend] cluster legend shown")

    def _draw_graph(self) -> None:
        """Render the current graph with either degree colormap or cluster colors."""
        if self.G is None:
            self.canvas.ax.set_axis_off()
            self.canvas.draw_idle()
            return

        G, info, pos = self.G, self.info, self._ensure_positions(self.G, self.pos)
        ax = self.canvas.ax
        ax.clear()
        ax.set_axis_off()

        root_set = set(self.root_names)

        # Decide coloring mode
        clustering = self.combo_cluster.currentText().lower()
        node_colors: List[str] = []
        node_edges: List[str] = []
        node_lws: List[float] = []
        node_labels: Dict[str, str] = {}

        if clustering in ("spectral", "2d k-means", "dbscan") and self.cluster_labels:
            for n in G.nodes():
                ni = info[n]
                in_cycle = G.nodes[n].get("in_cycle", False)
                edge_col = "#ff0000" if in_cycle else ("#2f2f2f" if ni.installed else "#d62728")
                node_edges.append(edge_col)
                node_lws.append(3.0 if n in root_set else 1.2)
                cid = self.cluster_labels.get(n, -1)
                if cid == -1:
                    color = "#BDBDBD"  # Noise
                else:
                    color = self.cluster_palette.get(cid, "#9ecae1")
                node_colors.append(color)
                node_labels[n] = ni.display
            self._draw_cluster_legend()
        else:
            cmap = self._selected_cmap()
            node_color_map, norm = self._compute_degree_colors(G, cmap)
            for n in G.nodes():
                ni = info[n]
                in_cycle = G.nodes[n].get("in_cycle", False)
                edge_col = "#ff0000" if in_cycle else ("#2f2f2f" if ni.installed else "#d62728")
                node_edges.append(edge_col)
                node_lws.append(3.0 if n in root_set else 1.2)
                node_colors.append(node_color_map[n])
                node_labels[n] = ni.display
            self._draw_colorbar(norm, cmap)

        solid_edges, dashed_edges = [], []
        for u, v in G.edges():
            if info[v].installed:
                solid_edges.append((u, v))
            else:
                dashed_edges.append((u, v))

        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            edgecolors=node_edges,
            linewidths=node_lws,
            node_size=self.spin_node_size.value(),
        )
        nx.draw_networkx_edges(
            G, pos, edgelist=solid_edges, ax=ax,
            arrows=self.chk_arrows.isChecked(),
            arrowstyle="-|>" if self.chk_arrows.isChecked() else "-"
        )
        nx.draw_networkx_edges(
            G, pos, edgelist=dashed_edges, ax=ax,
            arrows=self.chk_arrows.isChecked(),
            arrowstyle="-|>" if self.chk_arrows.isChecked() else "-",
            style="dashed"
        )
        nx.draw_networkx_labels(
            G, pos, labels=node_labels, ax=ax, font_size=self.spin_font.value(),
        )

        title = "Dependency Graph (A → B means A depends on B)" if self.mode == "deps" \
            else "Dependents of selected packages (A → B means A depends on B)"

        if clustering == "spectral" and self.cluster_labels:
            title += f"  |  Clustering: Spectral (k={self.spin_clusters.value()})"
        elif clustering == "2d k-means" and self.cluster_labels:
            title += f"  |  Clustering: 2D K-Means (k={self.spin_clusters.value()})"
        elif clustering == "dbscan" and self.cluster_labels:
            if self.dbscan_meta:
                C, Nnoise, eps, msz, score, metric = self.dbscan_meta
                frac = (100.0 * Nnoise / max(1, G.number_of_nodes()))
                score_txt = f", {metric}={score:.3f}" if (score is not None and metric) else ""
                title += f"  |  Clustering: DBSCAN (C={C}, noise={frac:.1f}%, eps={eps:.3g}, min={msz}{score_txt})"
            else:
                title += "  |  Clustering: DBSCAN"

        if self.chk_pack.isChecked() and self.cluster_labels:
            title += "  |  Packed"

        ax.set_title(title)
        self.canvas.draw_idle()

    # -------- Export --------

    def export_png(self) -> None:
        """Export the current figure to a PNG image."""
        path, _ = QFileDialog.getSaveFileName(self, "Export PNG", "", "PNG Image (*.png)")
        if not path:
            return
        try:
            self.canvas.figure.savefig(path, dpi=200)
            self._log(f"[export] PNG → {path}")
        except Exception as e:
            self._log(f"[error] export PNG failed: {e}")
            QMessageBox.critical(self, "Export failed", f"Failed to export PNG:\n{e}")

    def export_svg(self) -> None:
        """Export the current figure to an SVG vector file."""
        path, _ = QFileDialog.getSaveFileName(self, "Export SVG", "", "SVG Vector (*.svg)")
        if not path:
            return
        try:
            self.canvas.figure.savefig(path)
            self._log(f"[export] SVG → {path}")
        except Exception as e:
            self._log(f"[error] export SVG failed: {e}")
            QMessageBox.critical(self, "Export failed", f"Failed to export SVG:\n{e}")

    # -------- Misc --------

    def _update_missing_list(self) -> None:
        """List nodes that are not installed (missing distributions)."""
        if self.G is None:
            self.txt_missing.setPlainText("")
            return
        info = self.info
        missing = sorted([n for n in self.G.nodes() if not info[n].installed])
        self.txt_missing.setPlainText("(none)" if not missing else "\n".join(missing))


# --------------------------- Entrypoint ----------------------------

def main() -> None:
    """Qt application entry point: create the main window and start the event loop."""
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
