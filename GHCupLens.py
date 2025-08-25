#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ghcup Manager (PyQt6)
=====================
Left pane width is locked to exactly the Tool column width (no extra padding).
- Left table: Tool only (unique)
- Right panel: versions (numeric-desc), install, optional set default
- Non-blocking QProcess, robust `ghcup list` parsing (ANSI, headers, ✗/✓/✔)

Run:
  pip install PyQt6
  python ghcup_manager.py
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from PyQt6.QtCore import (
    Qt,
    QAbstractTableModel,
    QModelIndex,
    QVariant,
    QRegularExpression,
)
from PyQt6.QtGui import QAction, QColor, QBrush
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTableView,
    QHeaderView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import QSortFilterProxyModel, pyqtSignal, QObject
from PyQt6.QtCore import QProcess


# -------------------------------
# Data structures
# -------------------------------

@dataclass
class ToolEntry:
    tool: str
    version: str
    tags: List[str]

    def tag_str(self) -> str:
        return ", ".join(self.tags)


# -------------------------------
# Parsing `ghcup list` output (robust)
# -------------------------------

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
LEAD_GLYPH_RE = re.compile(r"^\s*(?:[^\w]*(?:✗|✓|✔✔|✔)\s+)?")
LINE_RE = re.compile(
    r"^\s*(?P<tool>ghc|hls|cabal|stack|ghcup)\s+"
    r"(?P<ver>[A-Za-z0-9_.+\-]+)\s*"
    r"(?P<rest>.*)$",
    re.IGNORECASE,
)
TAG_TOKENS = ("installed", "recommended", "latest", "set", "default", "hls-powered")


def parse_ghcup_list(text: str) -> List[ToolEntry]:
    def strip_ansi(s: str) -> str:
        return ANSI_RE.sub("", s)

    text = strip_ansi(text)
    entries: List[ToolEntry] = []

    for raw in text.splitlines():
        raw_no_ansi = strip_ansi(raw)
        line = raw_no_ansi.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("[ warn ") or low.startswith("[ info "):
            continue
        if low.startswith("tool ") and "version" in low:
            continue

        cleaned = LEAD_GLYPH_RE.sub("", line)
        m = LINE_RE.match(cleaned)
        if not m:
            continue

        tool = m.group("tool").lower()
        ver  = m.group("ver")
        rest = m.group("rest").strip()

        tags: List[str] = []
        low_rest = rest.lower()
        for t in TAG_TOKENS:
            if t in low_rest and t not in tags:
                tags.append(t)

        lead = raw_no_ansi.lstrip()
        if lead.startswith("✓") or lead.startswith("✔"):
            if "installed" not in tags:
                tags.append("installed")
            if lead.startswith("✔✔") and ("set" not in tags and "default" not in tags):
                tags.append("set")

        entries.append(ToolEntry(tool=tool, version=ver, tags=tags))

    return entries


# -------------------------------
# Qt Models
# -------------------------------

class EntriesTableModel(QAbstractTableModel):
    """Compact table: **Tool** only (unique)."""
    HEADERS = ["Tool"]

    def __init__(self, entries: List[ToolEntry]):
        super().__init__()
        self._entries = entries

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # type: ignore[override]
        return len(self._entries)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # type: ignore[override]
        return 1

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):  # type: ignore[override]
        if not index.isValid() or role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            return QVariant()
        e = self._entries[index.row()]
        if index.column() == 0:
            return e.tool
        return QVariant()

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):  # type: ignore[override]
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return QVariant()

    def entry_at(self, row: int) -> Optional[ToolEntry]:
        if 0 <= row < len(self._entries):
            return self._entries[row]
        return None

    def set_entries(self, entries: List[ToolEntry]):
        self.beginResetModel()
        self._entries = entries
        self.endResetModel()


class AllColumnsFilterProxy(QSortFilterProxyModel):
    def __init__(self):
        super().__init__()
        self.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._regex = QRegularExpression("")

    def setFilterText(self, text: str):
        self._regex = QRegularExpression(re.escape(text), QRegularExpression.PatternOption.CaseInsensitiveOption)
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:  # type: ignore[override]
        if not self._regex.pattern():
            return True
        model = self.sourceModel()
        for col in range(model.columnCount()):
            idx = model.index(source_row, col, source_parent)
            val = model.data(idx, Qt.ItemDataRole.DisplayRole)
            if val is not None and self._regex.match(str(val)).hasMatch():
                return True
        return False


# -------------------------------
# Command runner (QProcess)
# -------------------------------

class CommandRunner(QObject):
    output = pyqtSignal(str)
    finished = pyqtSignal(int, str)  # exitCode, command name

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.proc: Optional[QProcess] = None
        self.current_cmd_name: str = ""

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.state() != QProcess.ProcessState.NotRunning

    def run(self, cmd_name: str, args: List[str], cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None):
        if self.is_running():
            self.output.emit("\n[Error] Another process is still running. Please wait or cancel.\n")
            return
        self.current_cmd_name = cmd_name
        self.proc = QProcess(self)
        if env:
            qenv = self.proc.processEnvironment()
            for k, v in env.items():
                qenv.insert(k, v)
            self.proc.setProcessEnvironment(qenv)
        if cwd:
            self.proc.setWorkingDirectory(cwd)

        self.proc.setProgram(args[0])
        self.proc.setArguments(args[1:])
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read_output)
        self.proc.finished.connect(self._finished)
        try:
            self.proc.start()
        except Exception as e:
            self.output.emit(f"[Error] Failed to start: {' '.join(args)}\n{e}\n")
            self.proc = None

    def send_stdin(self, text: str):
        if self.proc:
            self.proc.write(text.encode('utf-8'))
            self.proc.write(b"\n")

    def _read_output(self):
        if not self.proc:
            return
        data = self.proc.readAllStandardOutput().data().decode('utf-8', errors='replace')
        if data:
            self.output.emit(data)

    def _finished(self, exitCode: int, _status):
        name = self.current_cmd_name
        self.current_cmd_name = ""
        self.proc = None
        self.finished.emit(exitCode, name)


# -------------------------------
# Main Window
# -------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ghcup Manager (PyQt6)")
        self.resize(1100, 720)

        # Top actions
        self.btnUpgrade = QPushButton("ghcup upgrade")
        self.btnUpdate = QPushButton("ghcup update")
        self.btnRefresh = QPushButton("Refresh list")

        self.filterLabel = QLabel("Search:")
        self.filterEdit = QLineEdit()
        self.filterEdit.setPlaceholderText("type to filter …")

        topRow = QHBoxLayout()
        topRow.addWidget(self.btnUpgrade)
        topRow.addWidget(self.btnUpdate)
        topRow.addWidget(self.btnRefresh)
        topRow.addStretch(1)
        topRow.addWidget(self.filterLabel)
        topRow.addWidget(self.filterEdit, 1)

        # Center: split between table (left) and versions (right)
        self.table = QTableView()
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # compact view for single-column table
        self.table.verticalHeader().setVisible(False)
        hdr = self.table.horizontalHeader()
        hdr.setStretchLastSection(True)
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)

        self.model = EntriesTableModel([])
        self.proxy = AllColumnsFilterProxy()
        self.proxy.setSourceModel(self.model)
        self.table.setModel(self.proxy)
        self.table.sortByColumn(0, Qt.SortOrder.AscendingOrder)

        # Right pane: details & install
        self.toolLabel = QLabel("Selected tool:")
        self.lblSelectedTool = QLabel("-")

        self.versionsList = QListWidget()
        self.btnInstall = QPushButton("Install selected version")
        self.cbSetDefault = QCheckBox("Set as default after install")

        rightBox = QVBoxLayout()
        r1 = QHBoxLayout()
        r1.addWidget(self.toolLabel)
        r1.addWidget(self.lblSelectedTool)
        r1.addStretch(1)
        rightBox.addLayout(r1)
        rightBox.addWidget(QLabel("Available versions (desc):"))
        rightBox.addWidget(self.versionsList, 1)
        rightBox.addWidget(self.cbSetDefault)
        rightBox.addWidget(self.btnInstall)

        rightWidget = QWidget(); rightWidget.setLayout(rightBox)

        self.splitter = QSplitter()
        self.leftPane = QWidget(); leftLayout = QVBoxLayout()
        leftLayout.setContentsMargins(0, 0, 0, 0)
        leftLayout.setSpacing(0)
        leftLayout.addWidget(self.table, 1)
        self.leftPane.setLayout(leftLayout)

        self.splitter.addWidget(self.leftPane)
        self.splitter.addWidget(rightWidget)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)

        # Log output
        self.log = QTextEdit(); self.log.setReadOnly(True)

        # Central layout
        central = QWidget(); layout = QVBoxLayout(central)
        layout.addLayout(topRow)
        layout.addWidget(self.splitter, 1)
        layout.addWidget(QLabel("Console:"))
        layout.addWidget(self.log, 1)
        self.setCentralWidget(central)

        # Status bar
        self.setStatusBar(QStatusBar())

        # Command runner
        self.runner = CommandRunner(self)
        self.runner.output.connect(self._append_log)
        self.runner.finished.connect(self._cmd_finished)

        # store full entries for right pane
        self._all_entries: List[ToolEntry] = []

        # Signals
        self.btnUpgrade.clicked.connect(self.on_upgrade)
        self.btnUpdate.clicked.connect(self.on_update)
        self.btnRefresh.clicked.connect(self.on_refresh)
        self.filterEdit.textChanged.connect(self.proxy.setFilterText)
        self.table.selectionModel().selectionChanged.connect(self.on_selection_changed)
        self.btnInstall.clicked.connect(self.on_install)

        # Menu: send 'y' to stdin if ghcup prompts
        sendYesAct = QAction("Send 'y' to process", self)
        sendYesAct.triggered.connect(lambda: self.runner.send_stdin("y"))
        self.menuBar().addAction(sendYesAct)

        # Initial load
        self.run_list()

    # ------------- helpers -------------
    def _append_log(self, s: str):
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)
        self.log.insertPlainText(s)
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)

    def _exact_left_width(self) -> int:
        """Compute *exact* width that fits Tool column with no extra padding.
        = columnWidth(0) + frame*2 + (vscroll width if visible)
        """
        # Ensure column size hint is applied
        self.table.resizeColumnsToContents()
        col_w = self.table.columnWidth(0)
        w = col_w + self.table.frameWidth() * 2
        try:
            if self.table.verticalScrollBar().isVisible():
                w += self.table.verticalScrollBar().sizeHint().width()
        except Exception:
            pass
        return max(w, 80)

    def adjust_left_width(self):
        w = self._exact_left_width()
        # Fix table and pane to exactly this width
        self.table.setMinimumWidth(w)
        self.table.setMaximumWidth(w)
        self.leftPane.setMinimumWidth(w)
        self.leftPane.setMaximumWidth(w)
        # Ensure the single Tool column stretches to fill any rounding remainder
        self.table.horizontalHeader().setStretchLastSection(True)
        # Update splitter sizes (no arbitrary padding)
        total = max(self.width() - w, 1)
        self.splitter.setSizes([w, total])

    def resizeEvent(self, event):  # keep left width exact on window resize
        super().resizeEvent(event)
        self.adjust_left_width()

    def _cmd_finished(self, exitCode: int, name: str):
        self.statusBar().showMessage(f"[{name}] finished with exit code {exitCode}", 5000)
        if name in ("upgrade", "update", "install", "set"):
            self.run_list()

    def ghcup_path(self) -> str:
        env = os.environ.get("GHCUP_BIN")
        if env:
            p = os.path.expanduser(env)
            if os.path.exists(p):
                return p
        home_candidate = os.path.expanduser("~/.ghcup/bin/ghcup")
        if os.path.exists(home_candidate):
            return home_candidate
        return "ghcup"

    def run_cmd(self, name: str, args: List[str]):
        self.statusBar().showMessage(f"Running: {' '.join(args)}")
        self._append_log(f"\n$ {' '.join(args)}\n")
        env = {"NO_COLOR": "1", "CLICOLOR": "0"}
        self.runner.run(name, args, env=env)

    # ------------- actions -------------
    def on_upgrade(self):
        self.run_cmd("upgrade", [self.ghcup_path(), "upgrade", "-y"])  # -y to avoid prompt

    def on_update(self):
        self.run_cmd("update", [self.ghcup_path(), "update"])

    def on_refresh(self):
        self.run_list()

    def run_list(self):
        proc = QProcess(self)
        program = self.ghcup_path()
        proc.setProgram(program)
        proc.setArguments(["list"])
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        qenv = proc.processEnvironment()
        qenv.insert("NO_COLOR", "1")
        qenv.insert("CLICOLOR", "0")
        proc.setProcessEnvironment(qenv)

        buffer: List[str] = []

        def on_out():
            data = proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
            if data:
                buffer.append(data)

        def on_error(_err):
            msg = (
                f"Failed to start '{program}'.\n\n"
                "Ensure ghcup is installed and on PATH, or set env var GHCUP_BIN to the absolute path.\n"
                "Typical path: ~/.ghcup/bin/ghcup"
            )
            self._append_log("\n[Error] " + msg + "\n")
            QMessageBox.critical(self, "ghcup not found", msg)

        def on_finished(_code, _status):
            raw_text = "".join(buffer)
            self._append_log("\n$ ghcup list\n" + raw_text + "\n")
            entries = parse_ghcup_list(raw_text)

            self._all_entries = entries

            seen = set()
            unique: List[ToolEntry] = []
            for e in entries:
                if e.tool not in seen:
                    seen.add(e.tool)
                    unique.append(e)

            self.model.set_entries(unique)
            self.table.resizeColumnsToContents()
            self.adjust_left_width()

            if self.model.rowCount() > 0:
                self.table.selectRow(0)
            self.populate_versions_from_selection()

        proc.readyReadStandardOutput.connect(on_out)
        proc.errorOccurred.connect(on_error)
        proc.finished.connect(on_finished)
        proc.start()

    def on_selection_changed(self, *_):
        self.populate_versions_from_selection()
        self.adjust_left_width()  # keep left width synced if font/selection changes

    def populate_versions_from_selection(self):
        self.versionsList.clear()
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            self.lblSelectedTool.setText("-")
            return
        source_row = self.proxy.mapToSource(sel[0]).row()
        entry = self.model.entry_at(source_row)
        if not entry:
            self.lblSelectedTool.setText("-")
            return
        tool = entry.tool
        self.lblSelectedTool.setText(tool)

        all_entries = getattr(self, "_all_entries", [])
        seen = set()
        versions: List[Tuple[str, str]] = []
        for e in all_entries:
            if e.tool == tool and e.version not in seen:
                seen.add(e.version)
                versions.append((e.version, e.tag_str()))

        import re as _re
        def _ver_key(ver: str):
            nums = _re.findall(r"\d+", ver)
            return tuple(int(n) for n in nums) if nums else tuple()

        versions.sort(key=lambda v: _ver_key(v[0]), reverse=True)

        for ver, tags in versions:
            item = QListWidgetItem(f"{ver}  [{tags}]" if tags else ver)
            item.setData(Qt.ItemDataRole.UserRole, ver)
            # colorize: set/default -> light blue; installed (not set) -> light green
            low = (tags or "").lower()
            if ("set" in low) or ("default" in low):
                item.setBackground(QBrush(QColor("#dbeafe")))  # light blue
            elif "installed" in low:
                item.setBackground(QBrush(QColor("#cce2cb")))  # light green
            self.versionsList.addItem(item)

        if self.versionsList.count() > 0:
            self.versionsList.setCurrentRow(0)

    def selected_tool_and_version(self) -> Optional[Tuple[str, str]]:
        tool = self.lblSelectedTool.text().strip()
        if not tool or tool == "-":
            return None
        cur = self.versionsList.currentItem()
        if not cur:
            return None
        ver = cur.data(Qt.ItemDataRole.UserRole)
        if not ver:
            return None
        return tool, str(ver)

    def on_install(self):
        pair = self.selected_tool_and_version()
        if not pair:
            QMessageBox.warning(self, "Install", "Select a tool and a version to install.")
            return
        tool, ver = pair
        self.run_cmd("install", [self.ghcup_path(), "install", tool, ver])

        if self.cbSetDefault.isChecked():
            from PyQt6.QtCore import QTimer
            def do_set():
                self.run_cmd("set", [self.ghcup_path(), "set", tool, ver])
            QTimer.singleShot(1200, do_set)


# -------------------------------
# Entrypoint
# -------------------------------

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

