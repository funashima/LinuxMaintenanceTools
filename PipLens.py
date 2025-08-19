#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQt6 GUI for reviewing and upgrading Python packages using `pip-review`.

Features
- Fetch upgradable packages via `pip-review --format json`
     (with robust fallbacks)
- Header "Select All" checkbox (column 0), per-row checkboxes
- Unchecked rows are dimmed in light gray
- Upgrade selected packages with live logs and a progress bar
- Auto-refresh toggle after upgrades
- Colorize version bumps (major/minor/patch/same/downgrade/unknown)
- Show data source (pip-review / pip list)

Requirements
    pip install PyQt6 pip-review packaging

Run
    python pip_review_gui.py

Tested with Python 3.10+.
"""
from __future__ import annotations

import json
import os
import re
import sys
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

from packaging.version import Version, InvalidVersion

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRect
from PyQt6.QtGui import QColor, QBrush
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QAbstractItemView,
    QHeaderView,
    QCheckBox,
    QLabel,
    QProgressBar,
    QPlainTextEdit,
    QMessageBox,
)

# ----------------------------- Data Model -----------------------------


@dataclass
class Upgradable:
    name: str
    current: str
    latest: str
    source: str = "pip-review"  # or "pip list"

# -------------------------- Utility functions -------------------------


def _run(cmd: List[str], cwd: Optional[str] = None) -> Tuple[int, str, str]:
    """
       Run a command
       and return (returncode, stdout, stderr) with UTF-8 text.
    """
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError as e:
        return 127, "", str(e)


def using_venv() -> bool:
    """Detect if Python is running inside a virtual environment."""
    return getattr(sys, "base_prefix", sys.prefix) != sys.prefix or \
        os.environ.get("VIRTUAL_ENV") is not None


def python_exe() -> str:
    """Return the current Python executable path."""
    return sys.executable or "python"


# Regex parser for textual pip-review output
#_PIP_REVIEW_LINE_RE = re.compile(
#    r"^(?P<name>[A-Za-z0-9_.\-]+)\s*\(current\s*(?P<current>[^)]+)\)\s*-\>\s*(?P<latest>\S+)",
#    re.IGNORECASE,
#)
_PIP_REVIEW_LINE_RE = re.compile(
    r"""
    ^(?P<name>[A-Za-z0-9_.\-]+)      # package name
    \s*\(current\s*(?P<current>[^)]+)\)  # current version
    \s*->\s*(?P<latest>\S+)          # latest version
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_pip_review_text(text: str) -> List[Upgradable]:
    """Parse textual output of pip-review into Upgradable objects."""
    pkgs: List[Upgradable] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _PIP_REVIEW_LINE_RE.match(line)
        if m:
            pkgs.append(Upgradable(m.group("name"),
                                   m.group("current"),
                                   m.group("latest")))
            continue
        # Alternate format: "package 1.2.3 -> 1.2.4"
        if "->" in line:
            try:
                left, latest = [s.strip() for s in line.split("->", 1)]
                parts = left.split()
                if len(parts) >= 2:
                    name = parts[0]
                    current = parts[1]
                    pkgs.append(Upgradable(name, current, latest))
                    continue
            except Exception:
                pass
    return pkgs


# ---------------------- Fetch upgradable packages ----------------------
def fetch_upgradable() -> List[Upgradable]:
    """Try multiple strategies to list upgradable packages.

    1) pip-review --format json (preferred)
    2) pip-review (parse text)
    3) pip list --outdated --format=json (fallback)
    """
    # Strategy 1: pip-review JSON
    rc, out, err = _run(["pip-review", "--format", "json", "--no-pre"])
    if rc == 0 and out.strip():
        try:
            data = json.loads(out)
            return [Upgradable(d["name"],
                               d["current"],
                               d["latest"],
                               source="pip-review") for d in data]
        except Exception:
            pass

    # Strategy 2: pip-review text
    rc, out, err = _run(["pip-review"])
    if rc == 0 and out.strip():
        pkgs = parse_pip_review_text(out)
        if pkgs:
            return pkgs

    # Strategy 3: pip list
    rc, out, err = _run([python_exe(),
                         "-m",
                         "pip",
                         "list",
                         "--outdated",
                         "--format",
                         "json"])
    if rc == 0 and out.strip():
        try:
            data = json.loads(out)
            return [Upgradable(d["name"],
                               d["version"],
                               d["latest_version"],
                               source="pip list") for d in data]
        except Exception:
            pass

    err_msg = "Failed to get upgradable packages. "
    err_msg += f"Last error: rc={rc} err={err.strip()}"
    raise RuntimeError(err_msg)


# ----------------------------- Workers -------------------------------
class FetchWorker(QThread):
    fetched = pyqtSignal(list)
    failed = pyqtSignal(str)

    def run(self):
        try:
            pkgs = fetch_upgradable()
            self.fetched.emit(pkgs)
        except Exception as e:
            self.failed.emit(str(e))


class UpgradeWorker(QThread):
    progress = pyqtSignal(int, int)  # current_index, total
    log = pyqtSignal(str)
    finished_all = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self,
                 packages: List[Upgradable],
                 exact_pin: bool = True,
                 user_install_if_needed: bool = True):
        super().__init__()
        self.packages = packages
        self.exact_pin = exact_pin
        self.user_install_if_needed = user_install_if_needed
        self._stop = False

    def stop(self):
        """Request to stop the upgrade process."""
        self._stop = True

    def run(self):
        total = len(self.packages)
        for i, pkg in enumerate(self.packages, start=1):
            if self._stop:
                self.log.emit("[INFO] Upgrade cancelled by user.\n")
                break

            self.progress.emit(i - 1, total)
            target = f"{pkg.name}=={pkg.latest}" \
                if self.exact_pin else pkg.name

            cmd = [python_exe(), "-m", "pip", "install", "--upgrade", target]
            if self.user_install_if_needed and not using_venv():
                cmd.append("--user")

            self.log.emit(f"\n[RUN] {' '.join(cmd)}\n")
            try:
                with subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    bufsize=1,
                ) as proc:
                    for line in proc.stdout:
                        self.log.emit(line)
                        if self._stop:
                            try:
                                proc.terminate()
                            except Exception:
                                pass
                            err_msg = "[Info]"
                            err_msg += "Terminating current pip process...\n"
                            self.log.emit(err_msg)
                            break
                    rc = proc.wait()
                if rc != 0:
                    err_msg = "[ERROR] pip exited with code "
                    err_msg += f"{rc} for {pkg.name}.\n"
                    self.log.emit(err_msg)
            except FileNotFoundError:
                self.failed.emit("Python executable not found.")
                return
            except Exception as e:
                self.log.emit(f"[EXCEPTION] {e}\n")

            self.progress.emit(i, total)

        self.finished_all.emit()


# ------------------------------- UI ---------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("pip-review GUI Upgrader")
        self.resize(1100, 720)

        # Central widget and layout
        central = QWidget(self)
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)

        info = QLabel(
            "Environment: <b>{}</b>  (venv: {})".
            format(python_exe(), "Yes" if using_venv() else "No"))
        info.setTextFormat(Qt.TextFormat.RichText)
        vbox.addWidget(info)

        # Table: columns = [Select, Package, Current, Latest, Source]
        self.table = QTableWidget(0, 5, self)
        '''
        Header label for column 0 is empty,
        header checkbox will be drawn over it
        '''
        self.table.setHorizontalHeaderLabels(["",
                                              "Package",
                                              "Current",
                                              "Latest",
                                              "Source"])
        self.table.setSelectionBehavior(QAbstractItemView.
                                        SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.
                                   EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)

        header = self.table.horizontalHeader()
        # Column 0 fixed width for checkbox column
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 120)
        # Other columns
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        vbox.addWidget(self.table)

        # Header "Select All" checkbox
        self.header_cb = QCheckBox("Select All", header)
        self.header_cb.stateChanged.connect(self.on_header_select_all)
        self._position_header_checkbox()
        header.sectionResized.connect(self._position_header_checkbox)
        header.sectionMoved.connect(self._position_header_checkbox)

        # Buttons row
        btn_row = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh")
        self.btn_select_all = QPushButton("Select All")
        self.btn_clear = QPushButton("Clear Selection")
        self.btn_upgrade = QPushButton("Upgrade Selected")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        # Auto-refresh toggle
        self.chk_auto_refresh = QCheckBox("Auto refresh after upgrade")
        self.chk_auto_refresh.setChecked(True)

        btn_row.addWidget(self.btn_refresh)
        btn_row.addWidget(self.btn_select_all)
        btn_row.addWidget(self.btn_clear)
        btn_row.addStretch(1)
        btn_row.addWidget(self.chk_auto_refresh)
        btn_row.addWidget(self.btn_upgrade)
        btn_row.addWidget(self.btn_stop)
        vbox.addLayout(btn_row)

        # Progress + Log
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        vbox.addWidget(self.progress)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        vbox.addWidget(self.log, 1)

        # Connections
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_select_all.clicked.connect(self.select_all)
        self.btn_clear.clicked.connect(self.clear_selection)
        self.btn_upgrade.clicked.connect(self.upgrade_selected)
        self.btn_stop.clicked.connect(self.stop_upgrades)

        # State
        self.current_packages: List[Upgradable] = []
        self.upgrade_worker: Optional[UpgradeWorker] = None

        # Initial fetch
        self.refresh()

    # ----- header checkbox positioning -----
    def _position_header_checkbox(self):
        """Place the 'Select All' checkbox over header section 0."""
        header = self.table.horizontalHeader()
        x = header.sectionPosition(0) + 6
        y = 2
        w = max(80, header.sectionSize(0) - 12)
        h = header.height() - 4
        self.header_cb.setGeometry(QRect(x, y, w, h))
        self.header_cb.show()

    # ----- bump-kind classification and styling -----
    def _bump_kind(self, old: str, new: str) -> str:
        """
        Return one of: 'major', 'minor', 'patch',
           'same', 'downgrade', or 'unknown'.
        Handles pre/dev/post/local versions as best-effort.
        """
        try:
            a, b = Version(old), Version(new)
        except InvalidVersion:
            return "unknown"
        if b == a:
            return "same"
        if b < a:
            return "downgrade"
        if b.major != a.major:
            return "major"
        if b.minor != a.minor:
            return "minor"
        if b.micro != a.micro:
            return "patch"
        # If only pre/dev/post/local differ, treat as patch
        return "patch"

    def _apply_kind_style(self, row: int, kind: str):
        """Apply visible colors for the 'Latest' cell based on bump kind."""
        latest_item = self.table.item(row, 3)
        if not latest_item:
            return

        # Reset first
        latest_item.setBackground(QBrush())
        latest_item.setForeground(QBrush())

        # Strong foreground + pastel background per kind
        FG = {
            "major": Qt.GlobalColor.red,
            "minor": Qt.GlobalColor.darkYellow,
            "patch": Qt.GlobalColor.darkBlue,
            "same": Qt.GlobalColor.darkGreen,
            "downgrade": Qt.GlobalColor.magenta,
            "unknown": Qt.GlobalColor.darkGray,
        }
        BG = {
            "major": QColor(255, 220, 220),
            "minor": QColor(255, 245, 210),
            "patch": QColor(220, 230, 255),
            "same": QColor(220, 245, 220),
            "downgrade": QColor(240, 220, 255),
            "unknown": QColor(235, 235, 235),
        }

        latest_item.setForeground(QBrush(FG.get(kind, Qt.GlobalColor.black)))
        latest_item.setBackground(QBrush(BG.get(kind, QColor(0, 0, 0, 0))))
        latest_item.setToolTip(f"Bump: {kind}")

    # ----- row dimming -----
    def _apply_row_dim(self, row: int, dim: bool):
        """
           Apply dimmed style (light gray background, dark gray text)
           when unchecked.
        """
        if dim:
            for col in range(1, self.table.columnCount()):
                item = self.table.item(row, col)
                if not item:
                    continue
                item.setBackground(QBrush(QColor(235, 235, 235)))
                item.setForeground(QBrush(Qt.GlobalColor.darkGray))
        else:
            # Clear row colors, then re-apply bump color to Latest cell
            for col in range(1, self.table.columnCount()):
                item = self.table.item(row, col)
                if not item:
                    continue
                item.setBackground(QBrush())
                item.setForeground(QBrush())
            current = self.table.item(row, 2).text() \
                if self.table.item(row, 2) else ""
            latest = self.table.item(row, 3).text() \
                if self.table.item(row, 3) else ""
            kind = self._bump_kind(current, latest)
            self._apply_kind_style(row, kind)

    def _on_row_checkbox_changed(self, state: int):
        """Handle row checkbox state change: dim row when unchecked."""
        cb = self.sender()
        for r in range(self.table.rowCount()):
            if self.table.cellWidget(r, 0) is cb:
                self._apply_row_dim(r, not cb.isChecked())
                break
        self._sync_header_checkbox_state()

    # ------------------ Table helpers ------------------
    def populate_table(self, packages: List[Upgradable]):
        """Fill the table with packages."""
        self.table.setRowCount(0)
        for pkg in packages:
            row = self.table.rowCount()
            self.table.insertRow(row)

            # Row checkbox
            cb = QCheckBox()
            cb.setChecked(True)
            cb.stateChanged.connect(self._on_row_checkbox_changed)
            self.table.setCellWidget(row, 0, cb)

            # Package name
            name_item = QTableWidgetItem(pkg.name)
            # Current version
            current_item = QTableWidgetItem(pkg.current)
            current_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            # Latest version
            latest_item = QTableWidgetItem(pkg.latest)
            latest_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            # Bump kind styling
            kind = self._bump_kind(pkg.current, pkg.latest)
            self._apply_kind_style(row, kind)
            # Source
            source_item = QTableWidgetItem(pkg.source)
            source_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            self.table.setItem(row, 1, name_item)
            self.table.setItem(row, 2, current_item)
            self.table.setItem(row, 3, latest_item)
            self.table.setItem(row, 4, source_item)

            # Initially not dimmed
            self._apply_row_dim(row, False)

        # Sort by package name
        self.table.sortItems(1, Qt.SortOrder.AscendingOrder)
        self._sync_header_checkbox_state()

    def selected_packages(self) -> List[Upgradable]:
        """Return list of selected packages (checked rows)."""
        selected: List[Upgradable] = []
        for row in range(self.table.rowCount()):
            widget = self.table.cellWidget(row, 0)
            if isinstance(widget, QCheckBox) and widget.isChecked():
                name = self.table.item(row, 1).text()
                current = self.table.item(row, 2).text()
                latest = self.table.item(row, 3).text()
                source = self.table.item(row, 4).text()
                selected.append(Upgradable(name, current, latest, source))
        return selected

    def select_all(self):
        """Check all rows and remove dimming."""
        for row in range(self.table.rowCount()):
            w = self.table.cellWidget(row, 0)
            if isinstance(w, QCheckBox):
                w.setChecked(True)
            self._apply_row_dim(row, False)
        self._sync_header_checkbox_state()

    def clear_selection(self):
        """Uncheck all rows and apply dimming."""
        for row in range(self.table.rowCount()):
            w = self.table.cellWidget(row, 0)
            if isinstance(w, QCheckBox):
                w.setChecked(False)
            self._apply_row_dim(row, True)
        self._sync_header_checkbox_state()

    # ----- header checkbox behavior -----
    def on_header_select_all(self, state: int):
        """Select/Deselect all rows from the header checkbox."""
        if state == Qt.CheckState.Checked.value:
            self.select_all()
        elif state == Qt.CheckState.Unchecked.value:
            self.clear_selection()
        # PartiallyChecked is driven by row checkbox changes

    def _sync_header_checkbox_state(self):
        """Update header checkbox state based on row selection state."""
        total = self.table.rowCount()
        if total == 0:
            self.header_cb.setTristate(False)
            self.header_cb.setCheckState(Qt.CheckState.Unchecked)
            return
        checked = 0
        for row in range(total):
            w = self.table.cellWidget(row, 0)
            if isinstance(w, QCheckBox) and w.isChecked():
                checked += 1
        if checked == 0:
            self.header_cb.setTristate(False)
            self.header_cb.setCheckState(Qt.CheckState.Unchecked)
        elif checked == total:
            self.header_cb.setTristate(False)
            self.header_cb.setCheckState(Qt.CheckState.Checked)
        else:
            self.header_cb.setTristate(True)
            self.header_cb.setCheckState(Qt.CheckState.PartiallyChecked)

    # ------------------ Actions ------------------
    def refresh(self):
        """Start fetching upgradable packages."""
        self.log.appendPlainText("[INFO] Fetching upgradable packages...\n")
        self.btn_refresh.setEnabled(False)
        self.btn_upgrade.setEnabled(False)
        self.progress.setRange(0, 0)  # busy indicator

        self.fetch_worker = FetchWorker()
        self.fetch_worker.fetched.connect(self.on_fetched)
        self.fetch_worker.failed.connect(self.on_fetch_failed)
        self.fetch_worker.start()

    def on_fetched(self, packages: List[Upgradable]):
        """Handle successful package fetch."""
        self.current_packages = packages
        self.populate_table(packages)
        self.log.appendPlainText(f"[OK] {len(packages)} packages found.\n")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.btn_refresh.setEnabled(True)
        self.btn_upgrade.setEnabled(True)

    def on_fetch_failed(self, msg: str):
        """Handle fetch failure."""
        self.log.appendPlainText(f"[ERROR] {msg}\n")
        QMessageBox.critical(self, "Fetch failed", msg)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.btn_refresh.setEnabled(True)
        self.btn_upgrade.setEnabled(False)

    def upgrade_selected(self):
        """Kick off upgrades for selected packages."""
        packages = self.selected_packages()
        if not packages:
            QMessageBox.information(self,
                                    "No selection",
                                    "No packages are selected.")
            return

        self.log.appendPlainText("[INFO] Starting upgrades...\n")
        self.btn_upgrade.setEnabled(False)
        self.btn_refresh.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.progress.setRange(0, len(packages))
        self.progress.setValue(0)

        self.upgrade_worker = UpgradeWorker(packages,
                                            exact_pin=True,
                                            user_install_if_needed=True)
        self.upgrade_worker.progress.connect(self.on_upgrade_progress)
        self.upgrade_worker.log.connect(self.log.appendPlainText)
        self.upgrade_worker.finished_all.connect(self.on_upgrade_finished)
        self.upgrade_worker.failed.connect(self.on_upgrade_failed)
        self.upgrade_worker.start()

    def on_upgrade_progress(self, done: int, total: int):
        """Update progress bar during upgrades."""
        self.progress.setMaximum(total)
        self.progress.setValue(done)

    def on_upgrade_finished(self):
        """Handle the end of the upgrade process."""
        self.log.appendPlainText("\n[INFO] Upgrades finished.\n")
        self.btn_upgrade.setEnabled(True)
        self.btn_refresh.setEnabled(True)
        self.btn_stop.setEnabled(False)
        # Conditional auto refresh
        if self.chk_auto_refresh.isChecked():
            self.refresh()

    def on_upgrade_failed(self, msg: str):
        """Handle an unrecoverable error during upgrade."""
        self.log.appendPlainText(f"[ERROR] {msg}\n")
        QMessageBox.critical(self, "Upgrade failed", msg)
        self.btn_upgrade.setEnabled(True)
        self.btn_refresh.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def stop_upgrades(self):
        """Request to stop ongoing upgrades."""
        if self.upgrade_worker is not None:
            self.upgrade_worker.stop()
            self.btn_stop.setEnabled(False)

    # ----- graceful stop on window close -----
    def closeEvent(self, event):
        """Try to gracefully stop workers before closing the window."""
        try:
            if self.upgrade_worker and self.upgrade_worker.isRunning():
                self.upgrade_worker.stop()
                self.upgrade_worker.wait(3000)
        except Exception:
            pass
        try:
            if hasattr(self, "fetch_worker") \
                    and self.fetch_worker and self.fetch_worker.isRunning():
                self.fetch_worker.requestInterruption()
                self.fetch_worker.wait(1000)
        except Exception:
            pass
        event.accept()


# ----------------------------- Entrypoint -----------------------------
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
