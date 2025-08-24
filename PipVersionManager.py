#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pip Version Manager with Dependency Graph (PyQt6)

New in this build:
- "Include yanked" checkbox (default unchecked). When off, yanked releases are hidden from the list.
- If a yanked version is selected (only visible when "Include yanked" is ON), the "Install selected" button is disabled.

Also includes:
- Wider default Package column / left pane
- Node size control
- Drag-to-pan, wheel zoom (cursor-centered), click-to-focus
- Label font size control
- Depth default -1 (unlimited)
- Self-loop removal
- Versions & Info with install (confirmation dialog), live log
All UI strings and comments are in English.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ---- Optional imports: graceful degradation ----
try:
    import networkx as nx  # type: ignore
    HAVE_NX = True
except Exception:
    HAVE_NX = False

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # type: ignore
    from matplotlib.figure import Figure  # type: ignore
    from matplotlib import rcParams  # type: ignore
    from matplotlib.colors import to_rgba  # type: ignore
    rcParams.setdefault('font.family', ['DejaVu Sans'])
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False
    FigureCanvas = None  # type: ignore
    Figure = None  # type: ignore

try:
    from packaging.requirements import Requirement  # type: ignore
    from packaging.version import Version, InvalidVersion  # type: ignore
    HAVE_PKG = True
except Exception:
    HAVE_PKG = False
    class Version:  # crude fallback
        def __init__(self, s: str) -> None: self._s = s
        def __lt__(self, other: "Version") -> bool: return str(self) < str(other)
        def __str__(self) -> str: return self._s
        @property
        def is_prerelease(self) -> bool: return any(tag in self._s for tag in ("a", "b", "rc", "dev"))
    class InvalidVersion(Exception): ...

# ---- Importlib metadata for installed packages ----
try:
    import importlib.metadata as ilm  # Python 3.8+
except Exception:
    try:
        import importlib_metadata as ilm  # type: ignore
    except Exception:
        ilm = None  # type: ignore

# ---- PyQt6 ----
from PyQt6.QtCore import (
    Qt,
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    pyqtSignal,
    QProcess,
)
from PyQt6.QtGui import QAction, QKeySequence, QColor, QBrush
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableView,
    QLineEdit,
    QLabel,
    QAbstractItemView,
    QHeaderView,
    QSplitter,
    QGroupBox,
    QTextEdit,
    QPlainTextEdit,
    QTabWidget,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QCheckBox,
    QMessageBox,
)


# -------------------- Data structures --------------------

@dataclass
class PackageRow:
    name: str
    version: str


# -------------------- Utility functions --------------------

_CANON_RE = re.compile(r"[-_.]+")

def canon(name: str) -> str:
    """PEP 503-like canonicalization for comparing package names."""
    return _CANON_RE.sub("-", (name or "").strip().lower())


def list_installed_packages() -> List[PackageRow]:
    """Return installed packages using importlib.metadata only."""
    rows: List[PackageRow] = []
    if ilm is None:
        return rows
    seen: set[str] = set()
    for dist in ilm.distributions():  # type: ignore[attr-defined]
        try:
            md = getattr(dist, "metadata", None)
            n = (md.get("Name") if md else None) or getattr(dist, "name", None) or ""
            v = getattr(dist, "version", "") or ""
            if not n:
                continue
            key = canon(n)
            if key in seen:
                continue
            seen.add(key)
            rows.append(PackageRow(name=n, version=v))
        except Exception:
            continue
    rows.sort(key=lambda r: r.name.casefold())
    return rows


def build_dependency_graph(installed: List[PackageRow]) -> Tuple["nx.DiGraph | None", Dict[str, str]]:
    """
    Build a directed dependency graph G where edge A -> B means "A depends on B".
    Only includes nodes that are installed (present in `installed`).
    Returns (graph, canon_to_display).
    """
    if not HAVE_NX or ilm is None:
        return None, {}

    # Canonical name <-> display name mapping for installed set
    canon_to_display: Dict[str, str] = {canon(r.name): r.name for r in installed}
    installed_canon = set(canon_to_display.keys())

    G = nx.DiGraph()
    for c in installed_canon:
        G.add_node(c)

    for dist in ilm.distributions():  # type: ignore[attr-defined]
        try:
            meta = getattr(dist, "metadata", None)
            raw_name = (meta.get("Name") if meta else None) or getattr(dist, "name", None) or ""
            src = canon(raw_name)
            if src not in installed_canon:
                continue
            reqs = getattr(dist, "requires", None) or []
            for rline in reqs:
                target_name = None
                if HAVE_PKG:
                    try:
                        target_name = Requirement(rline).name  # type: ignore
                    except Exception:
                        target_name = None
                if not target_name:
                    # Fallback rough parse: split at specifiers/markers/extras
                    target_name = re.split(r"[ ;<>=!~\[\]\(]", rline.strip())[0]
                tgt = canon(target_name)

                # Skip invalid or self-dependency edges
                if not tgt or tgt == src:
                    continue

                if tgt in installed_canon:
                    G.add_edge(src, tgt)
        except Exception:
            continue

    # Remove any self-loops just in case
    try:
        G.remove_edges_from(nx.selfloop_edges(G))
    except Exception:
        for u, v in list(G.edges()):
            if u == v:
                G.remove_edge(u, v)

    # Attach display labels
    try:
        nx.set_node_attributes(G, {c: {"label": canon_to_display.get(c, c)} for c in G.nodes})
    except Exception:
        pass
    return G, canon_to_display


def safe_label(text: object, ascii_only: bool = False) -> str:
    """Return a glyph-safe label. Normalize and optionally strip to ASCII."""
    s = "" if text is None else str(text)
    s = unicodedata.normalize("NFKC", s)
    if ascii_only:
        return s.encode("ascii", "ignore").decode("ascii")
    return "".join(ch if ch.isprintable() else " " for ch in s)


# -------------------- Fetch versions from PyPI --------------------

def fetch_pypi_versions(project_name: str) -> Tuple[List[str], Dict[str, Dict]]:
    """
    Fetch versions from PyPI JSON API.
    Returns (version_strings, releases_meta)
      - version_strings: list of all version tags (as strings)
      - releases_meta: mapping version -> aggregated info:
            {
              "is_prerelease": bool,
              "yanked": bool,
              "upload_time": str | None
            }
    """
    project = canon(project_name)
    url = f"https://pypi.org/pypi/{project}/json"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return [], {}

    releases = data.get("releases", {}) or {}
    versions = list(releases.keys())
    meta: Dict[str, Dict] = {}
    for v in versions:
        files = releases.get(v) or []
        times = [f.get("upload_time_iso_8601") for f in files if f.get("upload_time_iso_8601")]
        newest = max(times) if times else None
        yanked_flags = [bool(f.get("yanked")) for f in files]
        yanked = all(yanked_flags) if files else False
        try:
            ver = Version(v)  # type: ignore
            is_pre = bool(getattr(ver, "is_prerelease", False))
        except Exception:
            is_pre = any(t in v for t in ("a", "b", "rc", "dev"))
        meta[v] = {"is_prerelease": is_pre, "yanked": yanked, "upload_time": newest}
    return versions, meta


def sort_versions_desc(versions: List[str]) -> List[str]:
    def key(v: str):
        try:
            return Version(v)  # type: ignore
        except Exception:
            return Version("0!" + v) if HAVE_PKG else Version(v)  # type: ignore
    return sorted(versions, key=key, reverse=True)


# -------------------- Qt models & widgets --------------------

class InstalledTableModel(QAbstractTableModel):
    COL_NAME = 0
    COL_VERSION = 1
    HEADERS = ["Package", "Version"]

    def __init__(self, rows: Optional[List[PackageRow]] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._rows: List[PackageRow] = rows or []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # type: ignore[override]
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # type: ignore[override]
        return 0 if parent.isValid() else 2

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):  # type: ignore[override]
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return section + 1

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):  # type: ignore[override]
        if not index.isValid():
            return None
        r = self._rows[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            if index.column() == self.COL_NAME:
                return r.name
            if index.column() == self.COL_VERSION:
                return r.version
        return None

    def set_rows(self, rows: List[PackageRow]) -> None:
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()

    def row_at(self, proxy_index: QModelIndex, proxy: QSortFilterProxyModel) -> Optional[PackageRow]:
        if not proxy_index.isValid():
            return None
        src = proxy.mapToSource(proxy_index)
        if not src.isValid():
            return None
        i = src.row()
        if i < 0 or i >= len(self._rows):
            return None
        return self._rows[i]



class CaseInsensitiveStableProxy(QSortFilterProxyModel):
    """
    QSortFilterProxyModel with case-insensitive, *stable* sorting.
    - Compare strings with .casefold() for all columns.
    - Ties are broken by original source row index to preserve input order.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.setSortCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:  # type: ignore[override]
        src = self.sourceModel()
        if src is None:
            return super().lessThan(left, right)
        ltxt = src.data(left, Qt.ItemDataRole.DisplayRole) or ""
        rtxt = src.data(right, Qt.ItemDataRole.DisplayRole) or ""
        lc = str(ltxt).casefold()
        rc = str(rtxt).casefold()
        if lc == rc:
            # Stable: keep original order from the source model
            return left.row() < right.row()
        return lc < rc

class GraphPanel(QWidget):
    """
    NetworkX dependency visualization panel.
    Controls: Mode (deps/rdeps), Layout, Depth (-1=unlimited), Label size, Node size.
    Features: click node to refocus, mouse wheel zoom, drag to pan.
    """
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.G: Optional["nx.DiGraph"] = None
        self.canon_to_display: Dict[str, str] = {}
        self.root: Optional[str] = None  # canonical name

        # Controls row
        self.mode = QComboBox()
        self.mode.addItem("Dependencies ->", userData="deps")
        self.mode.addItem("Dependents <-", userData="rdeps")

        self.layoutSel = QComboBox()
        self.layoutSel.addItem("kamada-kawai")
        self.layoutSel.addItem("spring")

        self.depth = QSpinBox()
        self.depth.setRange(-1, 32)
        self.depth.setValue(-1)
        self.depth.setToolTip("Max BFS depth from root (-1 = unlimited)")

        self.labelSize = QSpinBox()
        self.labelSize.setRange(6, 24)
        self.labelSize.setValue(11)
        self.labelSize.setToolTip("Label font size")

        self.nodeSize = QSpinBox()
        self.nodeSize.setRange(10, 4000)
        self.nodeSize.setSingleStep(10)
        self.nodeSize.setValue(400)
        self.nodeSize.setToolTip("Node circle size (root is 2×)")

        
        # Node alpha (fill transparency)
        self.nodeAlpha = QDoubleSpinBox()
        self.nodeAlpha.setRange(0.0, 1.0)
        self.nodeAlpha.setDecimals(2)
        self.nodeAlpha.setSingleStep(0.05)
        self.nodeAlpha.setValue(1.0)
        self.nodeAlpha.setToolTip("Node fill alpha (0=transparent, 1=opaque)")
        ctrl = QWidget()
        ctrl_h = QHBoxLayout(ctrl)
        ctrl_h.setContentsMargins(0, 0, 0, 0)
        ctrl_h.addWidget(QLabel("Mode:"))
        ctrl_h.addWidget(self.mode)
        ctrl_h.addSpacing(10)
        ctrl_h.addWidget(QLabel("Layout:"))
        ctrl_h.addWidget(self.layoutSel)
        ctrl_h.addSpacing(10)
        ctrl_h.addWidget(QLabel("Depth (-1=unlimited):"))
        ctrl_h.addWidget(self.depth)
        ctrl_h.addSpacing(10)
        ctrl_h.addWidget(QLabel("Label size:"))
        ctrl_h.addWidget(self.labelSize)
        ctrl_h.addSpacing(10)
        ctrl_h.addWidget(QLabel("Node size:"))
        ctrl_h.addWidget(self.nodeSize)
        ctrl_h.addSpacing(8)
        ctrl_h.addWidget(QLabel("Node alpha:"))
        ctrl_h.addWidget(self.nodeAlpha)
        ctrl_h.addStretch(1)

        # Host widget (canvas or info)
        self._view = QWidget()
        self._view_layout = QVBoxLayout(self._view)
        self._view_layout.setContentsMargins(0, 0, 0, 0)

        self.info = QLabel()
        self.info.setWordWrap(True)
        self._current_view_widget: Optional[QWidget] = None

        self.figure: Optional["Figure"] = None
        self.canvas: Optional["FigureCanvas"] = None

        # State for interactions
        self._ax = None
        self._last_pos: Dict[str, Tuple[float, float]] = {}
        self._mpl_cids: Dict[str, int] = {}

        # Drag/pan state
        self._panning: bool = False
        self._pan_button: Optional[int] = None
        self._pan_start_data: Optional[Tuple[float, float]] = None
        self._pan_start_xlim: Optional[Tuple[float, float]] = None
        self._pan_start_ylim: Optional[Tuple[float, float]] = None

        # Click-to-focus state
        self._press_px: Optional[Tuple[float, float]] = None
        self._mouse_moved: bool = False

        v = QVBoxLayout(self)
        v.addWidget(ctrl)
        v.addWidget(self._view, 1)

        # Hooks
        self.mode.currentIndexChanged.connect(self._redraw)
        self.layoutSel.currentIndexChanged.connect(self._redraw)
        self.depth.valueChanged.connect(self._redraw)
        self.labelSize.valueChanged.connect(self._redraw)
        self.nodeSize.valueChanged.connect(self._redraw)
        self.nodeAlpha.valueChanged.connect(self._redraw)

        self._show_info_if_needed()

    # ---- view helpers ----
    def _show_info_if_needed(self, message: Optional[str] = None) -> None:
        if self._current_view_widget is not self.info:
            if self._current_view_widget:
                self._view_layout.removeWidget(self._current_view_widget)
                self._current_view_widget.setParent(None)
            self._current_view_widget = self.info
            self._view_layout.addWidget(self.info)
        if message:
            self.info.setText(message)
        else:
            miss = []
            if not HAVE_NX:
                miss.append("networkx")
            if not HAVE_MPL:
                miss.append("matplotlib")
            if miss:
                self.info.setText(
                    "Graph view requires: "
                    + ", ".join(miss)
                    + "\nInstall with: python -m pip install "
                    + " ".join(miss)
                )
            else:
                self.info.setText("Select a package to visualize")

    def _ensure_canvas(self) -> bool:
        if not HAVE_MPL or Figure is None or FigureCanvas is None:
            self._show_info_if_needed()
            return False
        created = False
        if self.figure is None:
            try:
                self.figure = Figure(figsize=(5, 4), constrained_layout=True)
                created = True
            except Exception:
                self._show_info_if_needed("Failed to create matplotlib Figure")
                return False
        if self.canvas is None:
            try:
                self.canvas = FigureCanvas(self.figure)  # type: ignore[arg-type]
                created = True
            except Exception:
                self._show_info_if_needed("Failed to create matplotlib Canvas")
                return False
        if created or (self._current_view_widget is not self.canvas):
            if self._current_view_widget:
                self._view_layout.removeWidget(self._current_view_widget)
                self._current_view_widget.setParent(None)
            self._current_view_widget = self.canvas  # type: ignore[assignment]
            self._view_layout.addWidget(self.canvas)  # type: ignore[arg-type]

        # Connect mpl events once
        if self.canvas and 'press' not in self._mpl_cids:
            self._mpl_cids['press'] = self.canvas.mpl_connect('button_press_event', self._on_press)
        if self.canvas and 'release' not in self._mpl_cids:
            self._mpl_cids['release'] = self.canvas.mpl_connect('button_release_event', self._on_release)
        if self.canvas and 'motion' not in self._mpl_cids:
            self._mpl_cids['motion'] = self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        if self.canvas and 'scroll' not in self._mpl_cids:
            self._mpl_cids['scroll'] = self.canvas.mpl_connect('scroll_event', self._on_scroll)
        return True

    # Public API
    def set_dataset(self, G: "nx.DiGraph", canon_to_display: Dict[str, str]) -> None:
        if not HAVE_NX:
            self._show_info_if_needed()
            return
        self.G = G
        self.canon_to_display = dict(canon_to_display)
        self._redraw()

    def set_focus(self, root_canon: Optional[str]) -> None:
        if not HAVE_NX:
            return
        self.root = root_canon if (self.G is not None and root_canon in self.G) else None
        self._redraw()

    # Internal helpers
    def _subgraph(self) -> Optional["nx.DiGraph"]:
        if self.G is None or self.root is None:
            return None
        mode = self.mode.currentData()
        depth_raw = int(self.depth.value())
        cutoff = None if depth_raw < 0 else depth_raw  # -1 means unlimited
        Gx = self.G if mode == "deps" else self.G.reverse(copy=False)
        try:
            lengths = nx.single_source_shortest_path_length(Gx, self.root, cutoff=cutoff)  # type: ignore
        except Exception:
            return None
        nodes = list(lengths.keys())
        H = self.G.subgraph(nodes).copy()
        # Safety: drop self-loops at view time too
        try:
            H.remove_edges_from(nx.selfloop_edges(H))
        except Exception:
            for u, v in list(H.edges()):
                if u == v:
                    H.remove_edge(u, v)
        return H

    def _redraw(self) -> None:
        if not HAVE_NX:
            self._show_info_if_needed()
            return
        if not self._ensure_canvas():
            return

        H = self._subgraph()
        self.figure.clear()  # do not chain with add_subplot
        ax = self.figure.add_subplot(111)
        ax.axis("off")
        ax.set_aspect("equal", adjustable="datalim")
        self._ax = ax
        self._last_pos.clear()

        if not H:
            ax.text(0.5, 0.5, "Select a package to visualize", ha="center", va="center")
            self.canvas.draw_idle()
            return

        # Layout
        layout = self.layoutSel.currentText()
        try:
            if layout == "kamada-kawai":
                pos = nx.kamada_kawai_layout(H)  # type: ignore
            else:
                pos = nx.spring_layout(H, seed=42)  # type: ignore
        except Exception:
            pos = nx.spring_layout(H, seed=42)

        # Node styling
        # Node styling
        root = self.root
        base = int(self.nodeSize.value())
        sizes = [base*2 if n == root else base for n in H.nodes()]
        alpha = float(getattr(self, "nodeAlpha", None).value()) if hasattr(self, "nodeAlpha") else 1.0
        if HAVE_MPL and alpha < 1.0:
            colors = [to_rgba("#ffcc00", alpha=alpha) if n == root else to_rgba("#b9d7fb", alpha=alpha) for n in H.nodes()]
        else:
            colors = ["#ffcc00" if n == root else "#b9d7fb" for n in H.nodes()]

        nx.draw_networkx_nodes(H, pos, ax=ax, node_size=sizes, node_color=colors,
                               edgecolors="#333", linewidths=0.8)

        # --- Directed edges: separate straight vs. reciprocal (curved) ---
        edges = list(H.edges())
        processed_pairs = set()  # frozenset({u,v}) for reciprocal pairs handled
        straight = []
        curved_a = []
        curved_b = []
        for (u, v) in edges:
            if H.has_edge(v, u):
                key = frozenset((u, v))
                if key in processed_pairs:
                    continue
                processed_pairs.add(key)
                curved_a.append((u, v))
                curved_b.append((v, u))
            else:
                straight.append((u, v))

        # Draw straight (single-direction) edges
        if straight:
            nx.draw_networkx_edges(
                H, pos, ax=ax, edgelist=straight,
                arrows=True, arrowstyle='->', arrowsize=12, width=1.0, alpha=0.9
            )

        # Draw reciprocal pairs as two curved arcs to make direction clear
        if curved_a:
            nx.draw_networkx_edges(
                H, pos, ax=ax, edgelist=curved_a,
                arrows=True, arrowstyle='->', arrowsize=12, width=1.0, alpha=0.9,
                connectionstyle='arc3,rad=0.18'
            )
        if curved_b:
            nx.draw_networkx_edges(
                H, pos, ax=ax, edgelist=curved_b,
                arrows=True, arrowstyle='->', arrowsize=12, width=1.0, alpha=0.9,
                connectionstyle='arc3,rad=-0.18'
            )
        # Labels (single-pass; avoid duplicates)
        # Clear any pre-existing text artists on this axes just in case
        try:
            for _t in list(ax.texts):
                _t.remove()
        except Exception:
            pass
        raw_labels = {n: H.nodes[n].get("label", self.canon_to_display.get(n, n)) for n in H.nodes()}
        labels = {n: safe_label(v, ascii_only=False) for n, v in raw_labels.items()}
        fsize = int(self.labelSize.value())
        labels_drawn = False
        try:
            nx.draw_networkx_labels(H, pos, labels=labels, ax=ax, font_size=fsize)
            labels_drawn = True
        except Exception:
            pass
        if not labels_drawn:
            labels_ascii = {n: safe_label(v, ascii_only=True) for n, v in raw_labels.items()}
            try:
                nx.draw_networkx_labels(H, pos, labels=labels_ascii, ax=ax, font_size=fsize)
            except Exception:
                pass
        # Save positions for interaction
        self._last_pos = {str(n): (float(x), float(y)) for n, (x, y) in pos.items()}

        # Fit view
        ax.relim()
        ax.autoscale_view()
        self.canvas.draw_idle()

    # ---- interactions ----
    def _nearest_node(self, event, radius_px: float = 12.0) -> Tuple[Optional[str], float]:
        """Return (node_id, squared_pixel_distance) for nearest node to event; None if too far."""
        if self._ax is None or not self._last_pos:
            return None, float("inf")
        inv = self._ax.transData
        min_d2 = None
        hit_node = None
        for n, (x, y) in self._last_pos.items():
            px, py = inv.transform((x, y))
            dx = px - event.x
            dy = py - event.y
            d2 = dx*dx + dy*dy
            if min_d2 is None or d2 < min_d2:
                min_d2 = d2
                hit_node = n
        if min_d2 is None or min_d2 > radius_px*radius_px:
            return None, float("inf")
        return hit_node, float(min_d2)

    def _on_press(self, event) -> None:
        if self._ax is None or event.inaxes is not self._ax:
            return
        self._mouse_moved = False
        self._press_px = (event.x, event.y)

        # Middle(2) or Right(3): always pan
        if event.button in (2, 3):
            self._start_pan(event)
            return

        # Left(1): if near a node, wait for click; else start pan
        if event.button == 1:
            node, d2 = self._nearest_node(event, radius_px=12.0)
            if node is None:
                self._start_pan(event)

    def _start_pan(self, event) -> None:
        if self._ax is None:
            return
        self._panning = True
        self._pan_button = event.button
        self._pan_start_data = (event.xdata, event.ydata)
        self._pan_start_xlim = tuple(self._ax.get_xlim())
        self._pan_start_ylim = tuple(self._ax.get_ylim())

    def _on_motion(self, event) -> None:
        if self._ax is None or event.inaxes is not self._ax:
            return
        if self._press_px is not None:
            if abs(event.x - self._press_px[0]) > 2 or abs(event.y - self._press_px[1]) > 2:
                self._mouse_moved = True

        if not self._panning:
            return
        if self._pan_start_data is None or self._pan_start_xlim is None or self._pan_start_ylim is None:
            return
        x0, y0 = self._pan_start_data
        if x0 is None or y0 is None or event.xdata is None or event.ydata is None:
            return

        dx = event.xdata - x0
        dy = event.ydata - y0

        xlim0 = self._pan_start_xlim
        ylim0 = self._pan_start_ylim
        self._ax.set_xlim((xlim0[0] - dx, xlim0[1] - dx))
        self._ax.set_ylim((ylim0[0] - dy, ylim0[1] - dy))
        self.canvas.draw_idle()

    def _on_release(self, event) -> None:
        # Finish panning
        self._panning = False
        self._pan_button = None
        self._pan_start_data = None
        self._pan_start_xlim = None
        self._pan_start_ylim = None

        # Left-click without significant movement -> focus node
        if event.button == 1 and not self._mouse_moved and self._ax is not None and event.inaxes is self._ax:
            node, d2 = self._nearest_node(event, radius_px=12.0)
            if node is not None:
                self.set_focus(node)

        # Reset click state
        self._press_px = None
        self._mouse_moved = False

    def _on_scroll(self, event) -> None:
        if self._ax is None or event.inaxes is not self._ax:
            return
        # Zoom towards cursor
        base_scale = 1.2
        scale = (1 / base_scale) if event.button == 'up' else base_scale
        xlim = list(self._ax.get_xlim())
        ylim = list(self._ax.get_ylim())
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return
        new_w = (xlim[1] - xlim[0]) * scale
        new_h = (ylim[1] - ylim[0]) * scale
        relx = (xdata - xlim[0]) / (xlim[1] - xlim[0] + 1e-12)
        rely = (ydata - ylim[0]) / (ylim[1] - ylim[0] + 1e-12)
        self._ax.set_xlim([xdata - new_w * relx, xdata + new_w * (1 - relx)])
        self._ax.set_ylim([ydata - new_h * rely, ydata + new_h * (1 - rely)])
        self.canvas.draw_idle()


class VersionsInfoPanel(QWidget):
    """
    Shows metadata and available versions for the selected package.
    - Fetches version list from PyPI JSON API
    - Color-codes items
    - Install selected version with confirmation
    - Emits refreshRequested after successful install
    """
    refreshRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.current_name: Optional[str] = None
        self.current_version: Optional[str] = None
        self._cache_versions: Dict[str, Tuple[List[str], Dict[str, Dict]]] = {}
        self.proc: Optional[QProcess] = None

        self.name_label = QLabel("Package: -")
        self.version_label = QLabel("Version: -")
        self.meta = QTextEdit()
        self.meta.setReadOnly(True)

        # Top row
        top = QWidget()
        th = QHBoxLayout(top)
        th.setContentsMargins(0, 0, 0, 0)
        th.addWidget(self.name_label)
        th.addSpacing(12)
        th.addWidget(self.version_label)
        th.addStretch(1)

        # Details group
        box = QGroupBox("Details")
        vb = QVBoxLayout(box)
        vb.addWidget(self.meta)

        # Versions group
        self.verBox = QGroupBox("Available Versions (PyPI)")
        ver_top = QWidget()
        vh = QHBoxLayout(ver_top)
        vh.setContentsMargins(0, 0, 0, 0)
        self.btnRefresh = QPushButton("Refresh")
        self.chkPre = QCheckBox("Include pre-releases")
        self.chkPre.setChecked(False)
        self.chkYanked = QCheckBox("Include yanked")
        self.chkYanked.setChecked(False)
        self.btnInstall = QPushButton("Install selected")
        self.btnInstall.setEnabled(False)
        vh.addWidget(self.btnRefresh)
        vh.addSpacing(10)
        vh.addWidget(self.chkPre)
        vh.addSpacing(10)
        vh.addWidget(self.chkYanked)
        vh.addStretch(1)
        vh.addWidget(self.btnInstall)

        self.versionList = QListWidget()
        self.installLog = QPlainTextEdit()
        self.installLog.setReadOnly(True)
        self.installLog.setPlaceholderText("Install output will appear here...")

        ver_l = QVBoxLayout(self.verBox)
        ver_l.addWidget(ver_top)
        ver_l.addWidget(self.versionList, 2)
        ver_l.addWidget(self.installLog, 1)

        v = QVBoxLayout(self)
        v.addWidget(top)
        v.addWidget(box, 1)
        v.addWidget(self.verBox, 2)

        # Hooks
        self.btnRefresh.clicked.connect(self._refetch_versions)
        self.chkPre.toggled.connect(lambda _checked: self._populate_versions_from_cache())
        self.chkYanked.toggled.connect(lambda _checked: self._populate_versions_from_cache())
        self.versionList.currentItemChanged.connect(self._selection_changed)
        self.btnInstall.clicked.connect(self._do_install)

    # ---- public API ----
    def show_package(self, name: Optional[str], version: Optional[str]) -> None:
        self.current_name = name
        self.current_version = version
        if not name:
            self.name_label.setText("Package: -")
            self.version_label.setText("Version: -")
            self.meta.setPlainText("")
            self.versionList.clear()
            self.installLog.clear()
            self.btnInstall.setEnabled(False)
            self.btnInstall.setToolTip("")
            return
        self.name_label.setText(f"Package: {name}")
        self.version_label.setText(f"Version: {version or '-'}")
        self._populate_metadata(name)
        self._ensure_versions(name)

    # ---- metadata ----
    def _populate_metadata(self, name: str) -> None:
        text = []
        if ilm is not None:
            for dist in ilm.distributions():  # type: ignore[attr-defined]
                try:
                    meta = getattr(dist, "metadata", None)
                    dname = (meta.get("Name") if meta else None) or getattr(dist, "name", None)
                    if canon(dname or "") != canon(name):
                        continue
                    if meta:
                        for k in ("Summary", "Home-page", "Author", "License", "Requires-Python"):
                            v = meta.get(k)
                            if v:
                                text.append(f"{k}: {v}")
                    break
                except Exception:
                    continue
        self.meta.setPlainText("\n".join(text) if text else "(no metadata)")

    # ---- versions ----
    def _ensure_versions(self, name: str) -> None:
        c = canon(name)
        if c not in self._cache_versions:
            versions, meta = fetch_pypi_versions(name)
            self._cache_versions[c] = (versions, meta)
        self._populate_versions_from_cache()

    def _refetch_versions(self) -> None:
        if not self.current_name:
            return
        versions, meta = fetch_pypi_versions(self.current_name)
        self._cache_versions[canon(self.current_name)] = (versions, meta)
        self._populate_versions_from_cache()

    def _populate_versions_from_cache(self) -> None:
        self.versionList.clear()
        name = self.current_name
        if not name:
            self._update_install_enabled()
            return
        c = canon(name)
        tup = self._cache_versions.get(c)
        if not tup:
            self.versionList.addItem("(failed to fetch or no data)")
            self._update_install_enabled()
            return
        versions, meta = tup

        def is_pre(v: str) -> bool:
            return bool(meta.get(v, {}).get("is_prerelease"))

        def is_yanked(v: str) -> bool:
            return bool(meta.get(v, {}).get("yanked"))

        include_pre = bool(self.chkPre.isChecked())
        include_yanked = bool(self.chkYanked.isChecked())

        # Determine latest stable (non-yanked)
        latest_stable: Optional[str] = None
        stables = [v for v in versions if not is_pre(v) and not is_yanked(v)]
        stables_sorted = sort_versions_desc(stables)
        if stables_sorted:
            latest_stable = stables_sorted[0]

        # Filter shown list by toggles
        shown = [v for v in versions if (include_pre or not is_pre(v)) and (include_yanked or not is_yanked(v))]
        shown = sort_versions_desc(shown)

        inst = (self.current_version or "").strip()

        for v in shown:
            m = meta.get(v, {})
            tags = []
            if v == inst:
                tags.append("installed")
            if v == latest_stable:
                tags.append("latest")
            if is_pre(v):
                tags.append("pre")
            if is_yanked(v):
                tags.append("yanked")
            tagtxt = "  [" + ", ".join(tags) + "]" if tags else ""
            t = f"{v}{tagtxt}"
            it = QListWidgetItem(t)

            # Color rules
            if v == inst:
                it.setBackground(QBrush(QColor("#d9fdd3")))  # light green
            if v == latest_stable and v != inst:
                it.setBackground(QBrush(QColor("#d6eaff")))  # light blue
            if is_pre(v):
                it.setForeground(QBrush(QColor("#b26a00")))   # dark orange
            if is_yanked(v):
                it.setForeground(QBrush(QColor("#b00020")))   # red

            # Store raw version in item data for quick access
            it.setData(Qt.ItemDataRole.UserRole, v)
            self.versionList.addItem(it)

        self._update_install_enabled()

    def _selection_changed(self, _cur, _prev) -> None:
        self._update_install_enabled()

    def _selected_version(self) -> Optional[str]:
        it = self.versionList.currentItem()
        if not it:
            return None
        v = it.data(Qt.ItemDataRole.UserRole)
        if v:
            return str(v)
        # Fallback parse
        text = it.text()
        return text.split()[0] if text else None

    def _selected_is_yanked(self) -> bool:
        name = self.current_name
        if not name:
            return False
        c = canon(name)
        tup = self._cache_versions.get(c)
        if not tup:
            return False
        _versions, meta = tup
        v = self._selected_version()
        if not v:
            return False
        return bool(meta.get(v, {}).get("yanked"))

    def _update_install_enabled(self) -> None:
        enable = False
        tooltip = ""
        if self.current_name and self._selected_version():
            if self._selected_is_yanked():
                enable = False
                tooltip = "Cannot install a yanked version from this UI. Select a non-yanked version."
            else:
                enable = True
        self.btnInstall.setEnabled(enable)
        self.btnInstall.setToolTip(tooltip)

    # ---- install ----
    def _do_install(self) -> None:
        if not self.current_name:
            return
        v = self._selected_version()
        if not v:
            return
        # Prevent installing yanked version (double-check)
        if self._selected_is_yanked():
            QMessageBox.warning(self, "Yanked release",
                                "This version is yanked on PyPI. Please choose a non-yanked version.")
            self._update_install_enabled()
            return
        # Confirmation
        cmd_display = f'{sys.executable} -m pip install "{self.current_name}=={v}"'
        resp = QMessageBox.question(
            self, "Confirm installation",
            f"Run the following command?\n\n{cmd_display}\n\nThis may change your Python environment.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if resp != QMessageBox.StandardButton.Yes:
            return

        # Launch QProcess
        self.installLog.clear()
        self.btnInstall.setEnabled(False)
        self.btnRefresh.setEnabled(False)
        self.chkPre.setEnabled(False)
        self.chkYanked.setEnabled(False)

        self.proc = QProcess(self)
        # Merge stderr into stdout for simpler capture
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._proc_ready)
        self.proc.finished.connect(self._proc_finished)

        self.proc.start(sys.executable, ["-m", "pip", "install", f"{self.current_name}=={v}"])

    def _proc_ready(self) -> None:
        if not self.proc:
            return
        data = self.proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
        if data:
            self.installLog.appendPlainText(data.rstrip())

    def _proc_finished(self, exitCode: int, exitStatus) -> None:
        ok = (exitCode == 0)
        self.installLog.appendPlainText(f"\n[Finished] exitCode={exitCode}")
        QMessageBox.information(self, "Installation finished",
                                "Success." if ok else "Failed. Check the log above.")
        self.btnInstall.setEnabled(True)
        self.btnRefresh.setEnabled(True)
        self.chkPre.setEnabled(True)
        self.chkYanked.setEnabled(True)
        # Suggest refresh of installed list
        if ok:
            self.refreshRequested.emit()


# -------------------- Main window --------------------

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Pip Version Manager (with Graph)")
        self.resize(1300, 880)

        # Data caches
        self.rows: List[PackageRow] = []
        self.dep_graph: Optional["nx.DiGraph"] = None
        self.canon_to_display: Dict[str, str] = {}

        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        # Actions
        actRefresh = QAction("Refresh", self)
        actRefresh.setShortcut(QKeySequence.StandardKey.Refresh)
        actRefresh.triggered.connect(self._refresh)

        self.addAction(actRefresh)
        self.menuBar().addMenu("&Actions").addAction(actRefresh)

        # Left: search + table
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter packages (type to search)")
        self.search.textChanged.connect(self._apply_filter)

        self.model = InstalledTableModel([])
        self.proxy = CaseInsensitiveStableProxy(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.proxy.setFilterKeyColumn(InstalledTableModel.COL_NAME)

        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(InstalledTableModel.COL_NAME, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(InstalledTableModel.COL_VERSION, QHeaderView.ResizeMode.ResizeToContents)
        header.setMinimumSectionSize(220)  # prevent overly narrow columns
        self.table.setColumnWidth(InstalledTableModel.COL_NAME, 500)

        self.table.verticalHeader().setVisible(False)
        self.table.selectionModel().selectionChanged.connect(self._on_selection_changed)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(self.search)
        lv.addWidget(self.table, 1)

        # Right: tabs
        self.infoPanel = VersionsInfoPanel(self)
        self.infoPanel.refreshRequested.connect(self._refresh)
        self.graphPanel = GraphPanel(self)

        tabs = QTabWidget()
        tabs.addTab(self.infoPanel, "Versions & Info")
        tabs.addTab(self.graphPanel, "Dependency Graph")

        # Splitter
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(tabs)
        split.setStretchFactor(0, 3)  # favor left
        split.setStretchFactor(1, 4)
        split.setSizes([780, 600])

        # Top-level
        central = QWidget()
        cv = QVBoxLayout(central)
        cv.addWidget(split, 1)
        self.setCentralWidget(central)

        self.statusBar().showMessage("Ready")

    def _apply_filter(self, text: str) -> None:
        self.proxy.setFilterFixedString(text)

    def _refresh(self) -> None:
        self.statusBar().showMessage("Refreshing...")
        self.rows = list_installed_packages()
        self.model.set_rows(self.rows)
        # Initial sort: Package name A→Z, case-insensitive, stable
        try:
            hdr = self.table.horizontalHeader()
            hdr.setSortIndicator(InstalledTableModel.COL_NAME, Qt.SortOrder.AscendingOrder)
            hdr.setSortIndicatorShown(True)
            self.table.sortByColumn(InstalledTableModel.COL_NAME, Qt.SortOrder.AscendingOrder)
        except Exception:
            try:
                self.proxy.sort(InstalledTableModel.COL_NAME, Qt.SortOrder.AscendingOrder)
            except Exception:
                pass
        self.statusBar().showMessage(f"Found {len(self.rows)} packages")

        # Build dependency graph
        G, c2d = build_dependency_graph(self.rows)
        self.dep_graph = G
        self.canon_to_display = c2d
        if G is not None:
            self.graphPanel.set_dataset(G, c2d)

        self.infoPanel.show_package(None, None)

    def _on_selection_changed(self) -> None:
        idxs = self.table.selectionModel().selectedRows()
        if not idxs:
            self.infoPanel.show_package(None, None)
            if self.dep_graph is not None:
                self.graphPanel.set_focus(None)
            return
        row = self.model.row_at(idxs[0], self.proxy)
        if not row:
            return
        self.infoPanel.show_package(row.name, row.version)
        if self.dep_graph is not None:
            self.graphPanel.set_focus(canon(row.name))


# -------------------- Entry point --------------------

def main() -> int:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
