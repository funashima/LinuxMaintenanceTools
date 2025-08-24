#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQt6 GUI for managing pip package versions in the *current* Python interpreter.

Features
--------
- Left: table of installed packages (from `pip list --format=json`).
- Filter box: incremental search for package names (case-insensitive).
- Right: available versions for the selected package (`pip index versions`),
  with fallback to the PyPI JSON API.
- Install/downgrade to a chosen version, or uninstall.
- Options: index URL, extra index URL, `--pre`, `--user`, `--no-deps`.
- Environment indicator: shows System / Virtualenv / Conda and active prefix.
- Version color hints relative to the installed version:
    * Red    — different MAJOR
    * Orange — different MINOR
    * Blue   — different PATCH
- FIFO queue for pip tasks to avoid concurrent runs; visible progress + Cancel.

Notes
-----
- All operations target this app's interpreter (`sys.executable -m pip`).
- Prefer using a virtual environment or a conda environment for safety.
- All comments and docstrings are in English for maintainability.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

from PyQt6.QtCore import Qt, QProcess, pyqtSignal, QSortFilterProxyModel, QRegularExpression
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QTableView, QListWidget, QListWidgetItem, QLabel, QLineEdit, QPushButton,
    QAbstractItemView, QTextEdit, QMessageBox, QCheckBox, QHeaderView, QGroupBox,
    QToolBar, QStatusBar, QMenu, QProgressBar
)
from PyQt6.QtGui import QStandardItemModel, QStandardItem

try:
    from packaging.version import Version
except Exception:
    class Version:  # lightweight fallback if packaging is unavailable
        """Minimal version-like object just to detect pre-releases."""
        def __init__(self, v: str) -> None:
            self._v = v
        @property
        def is_prerelease(self) -> bool:
            return bool(re.search(r"[abrc]|dev|post", self._v))
        def __str__(self) -> str:
            return self._v


@dataclass
class Package:
    """Represents one installed package."""
    name: str
    version: str


def get_environment_summary() -> dict:
    """Return a summary of the current Python environment.

    Returns
    -------
    dict with keys:
      type: {"conda", "virtualenv", "system"}
      name: environment name (if available)
      prefix: sys.prefix
      base_prefix: sys.base_prefix
      python_executable: path to sys.executable
      display: human-readable summary string
    """
    prefix = sys.prefix
    base_prefix = getattr(sys, "base_prefix", prefix)
    pyexec = sys.executable

    conda_prefix = os.environ.get("CONDA_PREFIX")
    conda_env_name = os.environ.get("CONDA_DEFAULT_ENV")
    virtual_env = os.environ.get("VIRTUAL_ENV")

    is_conda = bool(conda_prefix)
    is_venv = (prefix != base_prefix) or bool(virtual_env) or hasattr(sys, "real_prefix")

    if is_conda:
        env_type = "conda"
        name = conda_env_name or os.path.basename(conda_prefix)
    elif is_venv:
        env_type = "virtualenv"
        name = os.path.basename(prefix)
    else:
        env_type = "system"
        name = "system"

    display = f"{env_type.capitalize()} ({name}) — prefix: {prefix}"
    return {
        "type": env_type,
        "name": name,
        "prefix": prefix,
        "base_prefix": base_prefix,
        "python_executable": pyexec,
        "display": display,
    }


class PipRunner(QProcess):
    """
    Run `pip` commands asynchronously to keep the UI responsive.

    Signals
    -------
    outputReady(str): emitted when new output text arrives
    finishedOk(int, str): exit code 0 and full combined stdout/stderr
    finishedErr(int, str): non-zero exit code and full combined stdout/stderr
    """

    outputReady = pyqtSignal(str)
    finishedOk = pyqtSignal(int, str)
    finishedErr = pyqtSignal(int, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.readyReadStandardOutput.connect(self._on_read)
        self.finished.connect(self._on_finished)
        self._buffer: List[str] = []

    def run(self, args: List[str]) -> None:
        """Start `python -m pip <args>`."""
        self._buffer.clear()
        self.start(sys.executable, ["-m", "pip"] + args)

    def _on_read(self) -> None:
        """Collect output progressively and forward it to the log."""
        data = self.readAllStandardOutput().data().decode(errors="replace")
        if data:
            self._buffer.append(data)
            self.outputReady.emit(data)

    def _on_finished(self, code: int, _status) -> None:
        """Emit success/error signal with full text on process termination."""
        text = "".join(self._buffer)
        if code == 0:
            self.finishedOk.emit(code, text)
        else:
            self.finishedErr.emit(code, text)


class InstalledModel(QStandardItemModel):
    """Table model holding the installed package list."""

    COL_NAME = 0
    COL_VER = 1

    def __init__(self) -> None:
        super().__init__(0, 2)
        self.setHorizontalHeaderLabels(["Package", "Installed Version"])

    def set_packages(self, pkgs: List[Package]) -> None:
        """Replace table rows with `pkgs`."""
        self.setRowCount(0)
        for p in pkgs:
            name_item = QStandardItem(p.name)
            ver_item = QStandardItem(p.version)
            name_item.setEditable(False)
            ver_item.setEditable(False)
            self.appendRow([name_item, ver_item])


class MainWindow(QMainWindow):
    """Main application window for the pip version manager."""

    def __init__(self) -> None:
        super().__init__()
        # Ensure environment info is available before building the UI
        self.env_info = get_environment_summary()

        self.setWindowTitle("Pip Version Manager (PyQt6)")
        self.resize(1150, 740)

        self._build_ui()
        self._wire_actions()

        # FIFO queue and state flags
        self._queue: deque[List[str]] = deque()
        self._busy = False
        self._cancelled = False

        self.refresh_installed()

    # ---------- UI ----------

    def _build_ui(self) -> None:
        """Create widgets, lay them out, and prepare long-lived helpers."""
        self.statusBar = QStatusBar(self)
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage(
            f"Python: {self.env_info['python_executable']} | {self.env_info['display']}"
        )

        # Indeterminate progress bar shown while pip is running
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedWidth(140)
        self.progress.setVisible(False)
        self.statusBar.addPermanentWidget(self.progress)

        # Cancel button to terminate the current pip process (and clear queue)
        self.btnCancel = QPushButton("Cancel")
        self.btnCancel.setVisible(False)
        self.btnCancel.clicked.connect(self._on_cancel)
        self.statusBar.addPermanentWidget(self.btnCancel)

        # Toolbar
        tb = QToolBar("Main", self)
        tb.setMovable(False)
        self.addToolBar(tb)
        self.actRefresh = QAction("Refresh", self)
        self.actInstall = QAction("Install selected version", self)
        self.actUninstall = QAction("Uninstall package", self)
        tb.addAction(self.actRefresh)
        tb.addSeparator()
        tb.addAction(self.actInstall)
        tb.addAction(self.actUninstall)

        # Top options: Environment + pip options
        optBox = QGroupBox("Environment & pip options")
        optLayout = QHBoxLayout()
        self.lblEnv = QLabel(self.env_info["display"])  # explicit environment indicator
        self.lblEnv.setToolTip(
            f"type={self.env_info['type']}, name={self.env_info['name']}\n"
            f"prefix={self.env_info['prefix']}\nbase_prefix={self.env_info['base_prefix']}"
        )
        self.txtIndex = QLineEdit()
        self.txtIndex.setPlaceholderText("Index URL (-i), e.g. https://pypi.org/simple")
        self.txtExtraIndex = QLineEdit()
        self.txtExtraIndex.setPlaceholderText("Extra Index URL (--extra-index-url)")
        self.chkPre = QCheckBox("--pre")
        self.chkUser = QCheckBox("--user")
        # Disable or annotate --user depending on environment
        env_type = self.env_info["type"]
        if env_type == "virtualenv":
            # pip --user is not available inside virtual environments
            self.chkUser.setChecked(False)
            self.chkUser.setEnabled(False)
            self.chkUser.setToolTip("--user is disabled inside virtual environments (pip will refuse).")
        elif env_type == "conda":
            # Technically allowed, but installs to user site outside the env (confusing)
            self.chkUser.setToolTip(
                "Not recommended in Conda env: --user installs to the user site outside this environment."
            )
        else:
            # system Python: commonly useful
            self.chkUser.setToolTip(
                "Install to the per-user site-packages (no admin rights needed)."
            )
        self.chkNoDeps = QCheckBox("--no-deps")
        optLayout.addWidget(QLabel("Environment:"))
        optLayout.addWidget(self.lblEnv)
        optLayout.addSpacing(12)
        optLayout.addWidget(QLabel("Index:"))
        optLayout.addWidget(self.txtIndex)
        optLayout.addWidget(QLabel("Extra:"))
        optLayout.addWidget(self.txtExtraIndex)
        optLayout.addWidget(self.chkPre)
        optLayout.addWidget(self.chkUser)
        optLayout.addWidget(self.chkNoDeps)
        optBox.setLayout(optLayout)

        # Left panel: filter + table
        self.model = InstalledModel()
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setSortCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.proxy.setFilterKeyColumn(InstalledModel.COL_NAME)

        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        # Search box (incremental filter)
        self.searchEdit = QLineEdit()
        self.searchEdit.setPlaceholderText("Filter packages… (incremental)")

        # Right panel: versions + install button
        rightBox = QWidget()
        rightLayout = QVBoxLayout(rightBox)
        self.lblTarget = QLabel("Versions: -")
        self.listVersions = QListWidget()
        self.listVersions.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.btnInstall = QPushButton("Install selected version")
        self.btnInstall.setEnabled(False)
        rightLayout.addWidget(self.lblTarget)
        rightLayout.addWidget(self.listVersions, 1)
        rightLayout.addWidget(self.btnInstall)

        # Splitter layout
        leftBox = QWidget()
        leftLayout = QVBoxLayout(leftBox)
        leftLayout.addWidget(optBox)
        leftLayout.addWidget(QLabel("Filter:"))
        leftLayout.addWidget(self.searchEdit)
        leftLayout.addWidget(self.table, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(leftBox)
        splitter.addWidget(rightBox)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        # Log area
        self.txtLog = QTextEdit()
        self.txtLog.setReadOnly(True)
        self.txtLog.setPlaceholderText("pip logs will appear here…")

        # Central layout
        central = QWidget()
        mainLayout = QVBoxLayout(central)
        mainLayout.addWidget(splitter, 1)
        mainLayout.addWidget(self.txtLog, 1)
        self.setCentralWidget(central)

        # Process runner
        self.proc = PipRunner(self)
        self.proc.outputReady.connect(self._append_log)
        self.proc.finishedOk.connect(self._pip_ok)
        self.proc.finishedErr.connect(self._pip_err)

    def _wire_actions(self) -> None:
        """Connect UI signals to behavior."""
        self.table.selectionModel().selectionChanged.connect(self._on_row_selected)
        self.listVersions.itemSelectionChanged.connect(self._on_version_selected)
        self.btnInstall.clicked.connect(self.install_selected)
        self.actInstall.triggered.connect(self.install_selected)
        self.actUninstall.triggered.connect(self.uninstall_current)
        self.actRefresh.triggered.connect(self.refresh_installed)
        self.table.customContextMenuRequested.connect(self._on_table_ctx_menu)
        self.searchEdit.textChanged.connect(self._on_search_changed)

    # ---------- Helpers ----------

    def _append_log(self, text: str) -> None:
        """Append text to the log view and keep the cursor at the end."""
        self.txtLog.moveCursor(self.txtLog.textCursor().MoveOperation.End)
        self.txtLog.insertPlainText(text)
        self.txtLog.moveCursor(self.txtLog.textCursor().MoveOperation.End)

    def _on_search_changed(self, text: str) -> None:
        """Incremental filter by package name."""
        if text.strip():
            pattern = QRegularExpression(
                re.escape(text),
                QRegularExpression.PatternOption.CaseInsensitiveOption
            )
        else:
            pattern = QRegularExpression()
        self.proxy.setFilterRegularExpression(pattern)
        self.proxy.setFilterKeyColumn(InstalledModel.COL_NAME)

    # ---------- Process orchestration (queue, busy state, cancel) ----------

    def _set_busy(self, busy: bool) -> None:
        """Toggle busy UI state and controls while pip is running."""
        self._busy = busy
        self.progress.setVisible(busy)
        self.btnCancel.setVisible(busy)
        # Disable actions that would spawn more pip work
        self.actRefresh.setEnabled(not busy)
        self.actInstall.setEnabled(not busy)
        self.actUninstall.setEnabled(not busy)
        self.btnInstall.setEnabled(not busy and bool(self.listVersions.selectedItems()))
        self.table.setEnabled(not busy)

    def _on_cancel(self) -> None:
        """Cancel the running pip process and clear the pending queue."""
        self._cancelled = True
        # Clear any queued operations so nothing restarts automatically
        self._queue.clear()
        if self.proc.state() != QProcess.ProcessState.NotRunning:
            self.txtLog.append("(!) Cancelling current pip process…\n")
            self.proc.terminate()
            self.proc.waitForFinished(1500)
            if self.proc.state() != QProcess.ProcessState.NotRunning:
                self.proc.kill()
        # Busy state will be reset in the finished handler via _drain_queue()

    def _run_pip(self, args: List[str]) -> None:
        """Start a pip command or enqueue if another is running."""
        if self.proc.state() != QProcess.ProcessState.NotRunning:
            self._queue.append(list(args))  # copy to avoid aliasing
            self.txtLog.append(f"(queued) pip {' '.join(args)}\n")
            return
        self._start_pip(args)

    def _start_pip(self, args: List[str]) -> None:
        """Apply UI options and run pip now; show busy indicator."""
        idx = self.txtIndex.text().strip()
        exidx = self.txtExtraIndex.text().strip()
        real_args: List[str] = []
        if idx:
            real_args += ["-i", idx]
        if exidx:
            real_args += ["--extra-index-url", exidx]
        real_args += args
        self.txtLog.append(f"$ {sys.executable} -m pip {' '.join(real_args)}\n")
        self._set_busy(True)
        self.proc.run(real_args)

    def _drain_queue(self) -> None:
        """Start next queued pip command or clear busy state if none pending."""
        if self.proc.state() == QProcess.ProcessState.NotRunning:
            if self._queue:
                next_args = self._queue.popleft()
                self._start_pip(next_args)
            else:
                self._set_busy(False)

    # ---------- Core operations ----------

    def _run_pip_op(self, args: List[str]) -> None:
        """Helper to run a pip op (kept for readability)."""
        self._run_pip(args)

    def _current_pkg(self) -> Optional[str]:
        """Return the currently selected package name in the table."""
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return None
        name_idx = self.proxy.index(sel[0].row(), InstalledModel.COL_NAME)
        return self.proxy.data(name_idx)

    def _current_version(self) -> Optional[str]:
        """Return the currently selected available version (right list)."""
        items = self.listVersions.selectedItems()
        if not items:
            return None
        return items[0].text()

    def _current_installed_version(self) -> Optional[str]:
        """Return the installed version of the selected package."""
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return None
        ver_idx = self.proxy.index(sel[0].row(), InstalledModel.COL_VER)
        return self.proxy.data(ver_idx)

    @staticmethod
    def _version_parts(v: Optional[str]) -> Optional[Tuple[int, int, int]]:
        """Parse `v` into (major, minor, patch) integers.

        Uses `packaging.version.Version` if available, otherwise a simple
        numeric extractor that collects up to three integer groups.
        """
        if not v:
            return None
        try:
            pv = Version(v)
            major = getattr(pv, "major", None)
            minor = getattr(pv, "minor", None)
            micro = getattr(pv, "micro", None)
            if None not in (major, minor, micro):
                return int(major), int(minor), int(micro)
        except Exception:
            pass
        # Fallback: grab first 1-3 integer groups
        nums: List[int] = []
        token = ""
        for ch in v:
            if ch.isdigit():
                token += ch
            elif token:
                nums.append(int(token))
                token = ""
            if len(nums) >= 3:
                break
        if token and len(nums) < 3:
            nums.append(int(token))
        while len(nums) < 3:
            nums.append(0)
        return tuple(nums[:3])

    # ---------- Actions ----------

    def refresh_installed(self) -> None:
        """Refresh the installed package list (pip list)."""
        self.txtLog.append("\n=== Refresh installed packages ===\n")
        self._run_pip(["list", "--format=json"])  # handled in _pip_ok

    def _on_row_selected(self) -> None:
        """When a row is selected, fetch its available versions."""
        pkg = self._current_pkg()
        if pkg:
            self.fetch_versions(pkg)

    def fetch_versions(self, pkg: str) -> None:
        """Fetch available versions for `pkg` using pip (with JSON fallback)."""
        self.lblTarget.setText(f"Versions: {pkg}")
        self.listVersions.clear()
        self.btnInstall.setEnabled(False)
        self.txtLog.append(f"\n=== Fetch versions for {pkg} ===\n")
        self._run_pip(["index", "versions", pkg])

    def _on_version_selected(self) -> None:
        """Enable the install button when a version is selected."""
        self.btnInstall.setEnabled(bool(self.listVersions.selectedItems()))

    def install_selected(self) -> None:
        """Install the chosen version for the selected package."""
        pkg = self._current_pkg()
        ver = self._current_version()
        if not (pkg and ver):
            return
        args = ["install", f"{pkg}=={ver}"]
        if self.chkPre.isChecked():
            args.append("--pre")
        if self.chkUser.isChecked():
            args.append("--user")
        if self.chkNoDeps.isChecked():
            args.append("--no-deps")
        self._run_pip_op(args)

    def uninstall_current(self) -> None:
        """Uninstall the selected package after confirmation."""
        pkg = self._current_pkg()
        if not pkg:
            return
        if QMessageBox.question(self, "Uninstall", f"Uninstall '{pkg}' ?") \
                != QMessageBox.StandardButton.Yes:
            return
        self._run_pip_op(["uninstall", "-y", pkg])

    def _on_table_ctx_menu(self, pos) -> None:
        """Context menu for the installed packages table."""
        idx = self.table.indexAt(pos)
        if not idx.isValid():
            return
        pkg = self._current_pkg()
        menu = QMenu(self)
        actCopyName = QAction("Copy package name", self)
        actFetch = QAction("Fetch versions", self)
        menu.addAction(actCopyName)
        menu.addAction(actFetch)

        def copy_name():
            QApplication.clipboard().setText(pkg or "")

        actCopyName.triggered.connect(copy_name)
        actFetch.triggered.connect(lambda: self.fetch_versions(pkg))
        menu.exec(self.table.viewport().mapToGlobal(pos))

    # ---------- QProcess callbacks ----------

    def _pip_ok(self, _code: int, text: str) -> None:
        """Dispatch success handlers based on the last pip command."""
        last_cmd = self._last_cmd_in_log()
        if not last_cmd:
            self._drain_queue()
            return
        if last_cmd.startswith("list "):
            self._handle_list_ok(text)
        elif last_cmd.startswith("index versions "):
            pkg = last_cmd.split(" ", 2)[-1].strip()
            self._handle_versions_ok(pkg, text)
        elif last_cmd.startswith("install "):
            self.txtLog.append("\n✅ Install finished successfully.\n")
            self.refresh_installed()
        elif last_cmd.startswith("uninstall "):
            self.txtLog.append("\n✅ Uninstall finished successfully.\n")
            self.refresh_installed()
        self._drain_queue()

    def _pip_err(self, code: int, text: str) -> None:
        """Handle pip failures; use the JSON API for versions if needed."""
        last_cmd = self._last_cmd_in_log()
        self.txtLog.append(f"\n❌ pip exited with {code}.\n")
        if last_cmd and last_cmd.startswith("index versions "):
            pkg = last_cmd.split(" ", 2)[-1].strip()
            self.txtLog.append("Falling back to PyPI JSON API...\n")
            versions = self._fetch_versions_from_pypi_json(pkg)
            self._populate_version_list(versions)
            self._drain_queue()
            return
        self._drain_queue()

    def _last_cmd_in_log(self) -> Optional[str]:
        """Return the most recent `$ python -m pip ...` line we logged."""
        for line in reversed(self.txtLog.toPlainText().splitlines()):
            if line.startswith("$ ") and " -m pip " in line:
                return line.split(" -m pip ", 1)[1]
        return None

    # ---------- Handlers ----------

    def _handle_list_ok(self, text: str) -> None:
        """Parse/Render `pip list --format=json` into the table."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Some platforms merge progress noise; try to find a JSON block
            m = re.search(r"(\[.*\])", text, re.S)
            if not m:
                QMessageBox.warning(self, "Error", "Failed to parse pip list output.")
                return
            data = json.loads(m.group(1))
        pkgs = [Package(d.get("name"), d.get("version")) for d in data]
        pkgs.sort(key=lambda x: x.name.lower())
        self.model.set_packages(pkgs)
        self.proxy.sort(0, Qt.SortOrder.AscendingOrder)
        self.statusBar.showMessage(
            f"Loaded {len(pkgs)} packages | {self.env_info['display']}"
        )

    def _handle_versions_ok(self, pkg: str, text: str) -> None:
        """Parse `pip index versions` plain text; fallback to JSON if needed."""
        versions = self._parse_versions_from_pip_index(text)
        if not versions:
            self.txtLog.append(
                "Could not parse 'pip index versions' output. Trying PyPI JSON...\n"
            )
            versions = self._fetch_versions_from_pypi_json(pkg)
        self._populate_version_list(versions)

    def _populate_version_list(self, versions: List[str]) -> None:
        """
        Fill the right-side version list, sorting (desc) and colorizing
        relative to the installed version (major/minor/patch).
        """
        # Filter pre-releases unless --pre is checked
        if not self.chkPre.isChecked():
            versions = [v for v in versions if not Version(v).is_prerelease]
        # Unique + sort (best-effort semantic sort)
        try:
            versions = sorted(set(versions), key=lambda v: Version(v), reverse=True)
        except Exception:
            versions = sorted(set(versions), reverse=True)

        self.listVersions.clear()
        cur_v = self._current_installed_version()
        cur_parts = self._version_parts(cur_v)

        for v in versions:
            item = QListWidgetItem(v)
            if cur_parts:
                parts = self._version_parts(v)
                if parts:
                    if parts[0] != cur_parts[0]:
                        item.setForeground(Qt.GlobalColor.red)       # major difference
                    elif parts[1] != cur_parts[1]:
                        item.setForeground(Qt.GlobalColor.darkYellow) # minor difference (closest to orange)
                    elif parts[2] != cur_parts[2]:
                        item.setForeground(Qt.GlobalColor.blue)      # patch difference
            self.listVersions.addItem(item)

        self.btnInstall.setEnabled(bool(versions))
        self.statusBar.showMessage(
            f"Found {len(versions)} versions for selection | {self.env_info['display']}"
        )

    # ---------- Parsing / network ----------

    @staticmethod
    def _parse_versions_from_pip_index(text: str) -> List[str]:
        """
        Extract version strings from `pip index versions` output lines.

        Supports both single-line "Available versions: ..." and multi-line
        formats printed by different pip versions.
        """
        versions: List[str] = []
        for line in text.splitlines():
            if "Available versions:" in line:
                right = line.split("Available versions:", 1)[1]
                parts = [p.strip() for p in right.split(",")]
                versions.extend([p for p in parts if p])
        if not versions:
            # Fallback heuristic: collect tokens that look like versions
            tokens = re.findall(r"\b\d+[^\s,;]*", text)
            versions = tokens
        return versions

    @staticmethod
    def _fetch_versions_from_pypi_json(pkg: str) -> List[str]:
        """Fetch available versions from the PyPI JSON API."""
        url = f"https://pypi.org/pypi/{pkg}/json"
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                data = json.loads(r.read().decode())
            return list(data.get("releases", {}).keys())
        except Exception:
            return []


def main() -> None:
    """Application entrypoint."""
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
