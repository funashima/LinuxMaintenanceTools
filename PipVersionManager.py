#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pip Version Manager (PyQt6) — clean rebuild
- Asynchronous pip via QProcess
- Versions via: `pip index versions` → PyPI JSON (Qt) → pip-install probe
- Robust `pip list` parsing with ANSI stripping + importlib.metadata fallback
- English-only comments as requested
"""
from __future__ import annotations

import json
import os
import re
import sys
import shlex
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import (
    QAbstractTableModel,
    QItemSelection,
    QProcess,
    QSortFilterProxyModel,
    QTimer,
    QUrl,
    Qt,
    pyqtSignal,
    QSize,
    QSettings,
)
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QKeySequence,
    QStandardItem,
    QStandardItemModel,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGroupBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QAbstractItemView,
)

# Optional Qt network for non-blocking HTTP
try:
    from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
    HAVE_QTNETWORK = True
except Exception:
    HAVE_QTNETWORK = False

# Optional packaging for strong version ordering
try:
    from packaging.version import Version as _PkgVersion
    HAVE_PACKAGING = True
except Exception:
    HAVE_PACKAGING = False


@dataclass
class Package:
    name: str
    version: str


def detect_environment() -> Dict[str, str]:
    """Detect environment flavor (system / virtualenv / conda)."""
    env = {
        "type": "system",
        "name": "",
        "prefix": sys.prefix,
        "base_prefix": getattr(sys, "base_prefix", sys.prefix),
        "executable": sys.executable,
    }
    if os.environ.get("CONDA_PREFIX") or os.environ.get("CONDA_DEFAULT_ENV"):
        env["type"] = "conda"
        env["name"] = os.environ.get("CONDA_DEFAULT_ENV", "")
    elif (hasattr(sys, "base_prefix") and sys.prefix != sys.base_prefix) or os.environ.get("VIRTUAL_ENV"):
        env["type"] = "virtualenv"
        env["name"] = os.path.basename(os.environ.get("VIRTUAL_ENV", ""))
    return env


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences (CSI/OSC)."""
    text = re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", text)         # CSI
    text = re.sub(r"\x1B\][^\a]*(?:\a|\x1B\\)", "", text)        # OSC
    return text


def extract_json_array(text: str) -> Optional[str]:
    """
    Extract a JSON array substring from noisy or colored text.
    Strategy:
      1) strip ANSI colors
      2) prefer arrays that start with "[{"
      3) bracket-balance scan with string-awareness
    """
    s = _strip_ansi(text or "")
    m = re.search(r"\[\s*\{", s, re.S)
    start = m.start() if m else -1
    if start == -1:
        m2 = re.search(r"\[\s*\]", s)
        if m2:
            return s[m2.start():m2.end()]
        start = s.find("[")
        if start == -1:
            return None
    depth = 0
    in_string = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_string = False
            continue
        else:
            if ch == '"':
                in_string = True
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    return None


def is_prerelease(v: str) -> bool:
    """Return True iff version is pre-release; 'post' is NOT pre-release."""
    if HAVE_PACKAGING:
        try:
            return _PkgVersion(v).is_prerelease
        except Exception:
            pass
    return bool(re.search(r"(?i)(?:a|alpha|b|beta|rc|dev)\d*", v))


def version_key(v: str):
    """Ordering key for versions; use packaging if available, else tuple-ish fallback."""
    if HAVE_PACKAGING:
        try:
            return _PkgVersion(v)
        except Exception:
            pass
    parts: List[Tuple[int, str]] = []
    for token in re.split(r"[\.\-\+_]", v):
        if token.isdigit():
            parts.append((0, str(int(token))))
        else:
            parts.append((1, token))
    return tuple(parts)


def parse_semver_triplet(v: Optional[str]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Return (major, minor, patch) if available, else (None, None, None)."""
    if not v:
        return (None, None, None)
    m = re.match(r"^\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?", v)
    if not m:
        return (None, None, None)
    return tuple(int(x) if x is not None else 0 for x in m.groups(default="0"))


def _collect_installed_importlib() -> List[Package]:
    """Enumerate installed distributions via importlib.metadata as fallback."""
    try:
        import importlib.metadata as _ilm
    except Exception:
        try:
            import importlib_metadata as _ilm  # backport
        except Exception:
            _ilm = None
    pkgs: List[Package] = []
    if _ilm is None:
        return pkgs
    try:
        for dist in _ilm.distributions():
            name = None
            version = None
            try:
                md = getattr(dist, "metadata", None)
                if md:
                    name = md.get("Name") or md.get("name") or getattr(dist, "name", None)
                else:
                    name = getattr(dist, "metadata", {}).get("Name", None)  # type: ignore[call-arg]
                version = getattr(dist, "version", None)
            except Exception:
                pass
            if name and version:
                pkgs.append(Package(name, str(version)))
    except Exception:
        pass
    uniq: Dict[str, Package] = {}
    for p in pkgs:
        k = p.name.lower()
        if k not in uniq:
            uniq[k] = p
    return sorted(uniq.values(), key=lambda x: x.name.lower())


class InstalledModel(QStandardItemModel):
    """Two-column model: name | installed version."""
    COL_NAME = 0
    COL_VER = 1

    def __init__(self) -> None:
        super().__init__(0, 2)
        self.setHorizontalHeaderLabels(["Package", "Installed Version"])

    def set_packages(self, pkgs: List[Package]) -> None:
        self.setRowCount(0)
        for p in pkgs:
            it_name = QStandardItem(p.name)
            it_ver = QStandardItem(p.version)
            it_name.setEditable(False)
            it_ver.setEditable(False)
            self.appendRow([it_name, it_ver])


class PipRunner(QProcess):
    """Wrapper for QProcess to run `python -m pip` and stream logs."""
    outputReady = pyqtSignal(str)
    finishedOk = pyqtSignal(int, str)
    finishedErr = pyqtSignal(int, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.readyReadStandardOutput.connect(self._on_read)
        self.finished.connect(self._on_finished)
        self._buf: List[str] = []

    def run(self, args: List[str]) -> None:
        self._buf.clear()
        self.start(sys.executable, ["-m", "pip"] + args)

    def _on_read(self) -> None:
        data = self.readAllStandardOutput().data().decode(errors="replace")
        if data:
            self._buf.append(data)
            self.outputReady.emit(data)

    def _on_finished(self, code: int, _status: QProcess.ExitStatus) -> None:
        text = "".join(self._buf)
        if code == 0:
            self.finishedOk.emit(code, text)
        else:
            self.finishedErr.emit(code, text)


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.env = detect_environment()
        self.setWindowTitle("Pip Version Manager (PyQt6)")
        self.resize(1160, 760)
        self._busy = False

        # Settings
        self.settings = QSettings("hfuna", "PipVersionManager")

        # Process + queue
        self.proc = PipRunner(self)
        self.proc.outputReady.connect(self._append_log)
        self.proc.finishedOk.connect(self._on_pip_ok)
        self.proc.finishedErr.connect(self._on_pip_err)
        self._pending: List[Tuple[str, List[str]]] = []

        # Network
        self.nam: Optional["QNetworkAccessManager"] = None
        if HAVE_QTNETWORK:
            self.nam = QNetworkAccessManager(self)

        # UI
        self._build_ui()
        self._wire()
        self._restore_settings()

        QTimer.singleShot(50, self.refresh_installed)

    def _build_ui(self) -> None:
        # Actions
        self.actRefresh = QAction("Refresh", self)
        self.actRefresh.setShortcut(QKeySequence.StandardKey.Refresh)
        self.actInstall = QAction("Install", self)
        self.actUninstall = QAction("Uninstall", self)

        # Left table
        self.model = InstalledModel()
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.proxy.setFilterKeyColumn(InstalledModel.COL_NAME)

        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.editSearch = QLineEdit()
        self.editSearch.setPlaceholderText("Filter packages (incremental)")

        leftBox = QWidget()
        llay = QVBoxLayout(leftBox)
        llay.addWidget(self.editSearch)
        llay.addWidget(self.table, 1)

        # Right pane
        self.lblTarget = QLabel("Versions: -")
        self.listVersions = QListWidget()
        self.btnInstall = QPushButton("Install")
        self.btnInstall.setEnabled(False)
        self.txtInfo = QTextEdit()
        self.txtInfo.setReadOnly(True)
        self.txtInfo.setPlaceholderText("Package info (summary / urls / requires-python)")

        rightBox = QWidget()
        rlay = QVBoxLayout(rightBox)
        rlay.addWidget(self.lblTarget)
        rlay.addWidget(self.listVersions, 2)
        rlay.addWidget(self.txtInfo, 1)
        rlay.addWidget(self.btnInstall)

        # Options
        self.lblEnv = QLabel(self._format_env_label())
        self.txtIndex = QLineEdit()
        self.txtIndex.setPlaceholderText("Index URL (-i), e.g. https://pypi.org/simple")
        self.txtExtra = QLineEdit()
        self.txtExtra.setPlaceholderText("Extra index URL (--extra-index-url)")
        self.chkPre = QCheckBox("--pre")
        self.chkUser = QCheckBox("--user")
        self.chkNoDeps = QCheckBox("--no-deps")

        if self.env["type"] in {"virtualenv", "conda"}:
            self.chkUser.setChecked(False)
            self.chkUser.setEnabled(False)
            self.chkUser.setToolTip("--user is ignored inside virtualenv/conda.")

        optBox = QGroupBox("Environment & Options")
        optLay = QHBoxLayout(optBox)
        optLay.addWidget(self.lblEnv, 1)
        optLay.addWidget(self.txtIndex, 2)
        optLay.addWidget(self.txtExtra, 2)
        optLay.addWidget(self.chkPre)
        optLay.addWidget(self.chkUser)
        optLay.addWidget(self.chkNoDeps)

        # Splitter
        split = QSplitter()
        split.addWidget(leftBox)
        split.addWidget(rightBox)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)

        # Log
        self.txtLog = QTextEdit()
        self.txtLog.setReadOnly(True)
        self.txtLog.setPlaceholderText("pip logs will appear here")

        # Central
        central = QWidget()
        lay = QVBoxLayout(central)
        lay.addWidget(optBox)
        lay.addWidget(split, 1)
        lay.addWidget(self.txtLog, 1)
        self.setCentralWidget(central)

        # Status bar
        self.setStatusBar(QStatusBar())

        self.addAction(self.actRefresh)
        self.addAction(self.actInstall)
        self.addAction(self.actUninstall)

    def _wire(self) -> None:
        self.actRefresh.triggered.connect(self.refresh_installed)
        self.actInstall.triggered.connect(self.install_selected)
        self.actUninstall.triggered.connect(self.uninstall_current)

        self.editSearch.textChanged.connect(self._on_filter)
        self.table.selectionModel().selectionChanged.connect(self._on_row_selected)
        self.table.customContextMenuRequested.connect(self._on_table_ctx)

        self.listVersions.itemSelectionChanged.connect(self._on_version_selected)
        self.btnInstall.clicked.connect(self.install_selected)

    def _restore_settings(self) -> None:
        size = self.settings.value("size", None)
        if isinstance(size, QSize):
            self.resize(size)
        self.txtIndex.setText(self.settings.value("index_url", ""))
        self.txtExtra.setText(self.settings.value("extra_index", ""))
        self.chkPre.setChecked(self.settings.value("pre", False, type=bool))
        self.chkNoDeps.setChecked(self.settings.value("no_deps", False, type=bool))

    def closeEvent(self, e) -> None:
        self.settings.setValue("size", self.size())
        self.settings.setValue("index_url", self.txtIndex.text())
        self.settings.setValue("extra_index", self.txtExtra.text())
        self.settings.setValue("pre", self.chkPre.isChecked())
        self.settings.setValue("no_deps", self.chkNoDeps.isChecked())
        super().closeEvent(e)

    # Helpers

    def _format_env_label(self) -> str:
        t = self.env["type"]
        name = self.env["name"]
        if t == "system":
            base = "System Python"
        elif t == "virtualenv":
            base = f"Virtualenv ({name or 'unnamed'})"
        else:
            base = f"Conda ({name or 'env'})"
        return f"{base} — {self.env['executable']}"

    def _append_log(self, text: str) -> None:
        self.txtLog.moveCursor(self.txtLog.textCursor().MoveOperation.End)
        self.txtLog.insertPlainText(text)
        self.txtLog.moveCursor(self.txtLog.textCursor().MoveOperation.End)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.table.setEnabled(not busy)  # keep explicit and simple
        self.listVersions.setEnabled(not busy)
        self.btnInstall.setEnabled(not busy and len(self.listVersions.selectedItems()) == 1)
        self.actInstall.setEnabled(not busy)
        self.actUninstall.setEnabled(not busy)
        self.actRefresh.setEnabled(not busy)

    def _current_row_pkg(self) -> Optional[str]:
        idxs = self.table.selectionModel().selectedRows(InstalledModel.COL_NAME)
        if not idxs:
            return None
        src = self.proxy.mapToSource(idxs[0])
        return self.model.item(src.row(), InstalledModel.COL_NAME).text()

    def _current_installed_version(self) -> Optional[str]:
        idxs = self.table.selectionModel().selectedRows(InstalledModel.COL_VER)
        if not idxs:
            return None
        src = self.proxy.mapToSource(idxs[0])
        return self.model.item(src.row(), InstalledModel.COL_VER).text()

    def _last_pip_cmd_text(self) -> Optional[str]:
        for line in reversed(self.txtLog.toPlainText().splitlines()):
            if line.startswith("$ ") and " -m pip " in line:
                return line.split(" -m pip ", 1)[1]
        return None

    # Actions

    def _extract_pkg_from_index_cmd(self, last: str) -> str:
        """Return the package name from a command like 'index versions <pkg> [flags...]'."""
        try:
            toks = shlex.split(last)
        except Exception:
            toks = last.split()
        if len(toks) >= 3 and toks[0] == "index" and toks[1] == "versions":
            return toks[2]
        return self._current_row_pkg() or ""
    def refresh_installed(self) -> None:
        if self._busy:
            self._pending.append(("list", []))
            return
        args = ["list", "--format=json", "--disable-pip-version-check", "--no-color"]
        self._log_cmd(args)
        self._set_busy(True)
        self.proc.run(args)

    def request_versions(self, pkg: str) -> None:
        if self._busy:
            self._pending.append(("versions", [pkg]))
            return
        args = ["index", "versions", pkg, "--disable-pip-version-check", "--no-color"]
        if self.txtIndex.text().strip():
            args += ["-i", self.txtIndex.text().strip()]
        if self.txtExtra.text().strip():
            args += ["--extra-index-url", self.txtExtra.text().strip()]
        self._log_cmd(args)
        self._set_busy(True)
        self.proc.run(args)

    def install_selected(self) -> None:
        pkg = self._current_row_pkg()
        if not pkg:
            return
        items = self.listVersions.selectedItems()
        if not items:
            return
        target = items[0].data(Qt.ItemDataRole.UserRole)
        if not target:
            return
        args = ["install", f"{pkg}=={target}", "--disable-pip-version-check", "--no-color"]
        if self.chkNoDeps.isChecked():
            args.append("--no-deps")
        if self.chkPre.isChecked():
            args.append("--pre")
        if self.chkUser.isEnabled() and self.chkUser.isChecked():
            args.append("--user")
        if self.txtIndex.text().strip():
            args += ["-i", self.txtIndex.text().strip()]
        if self.txtExtra.text().strip():
            args += ["--extra-index-url", self.txtExtra.text().strip()]
        self._log_cmd(args)
        self._set_busy(True)
        self.proc.run(args)

    def uninstall_current(self) -> None:
        pkg = self._current_row_pkg()
        if not pkg:
            return
        args = ["uninstall", "-y", pkg, "--disable-pip-version-check", "--no-color"]
        self._log_cmd(args)
        self._set_busy(True)
        self.proc.run(args)

    # Results

    def _on_pip_ok(self, _code: int, text: str) -> None:
        self._set_busy(False)
        last = self._last_pip_cmd_text() or ""
        if last.startswith("list "):
            try:
                data = json.loads(text) if text.strip() else []
            except json.JSONDecodeError:
                block = extract_json_array(text or "")
                if block:
                    try:
                        data = json.loads(block)
                    except Exception:
                        data = []
                else:
                    data = []
            pkgs = [Package(d.get("name", ""), d.get("version", "")) for d in data if isinstance(d, dict) and d.get("name")]
            if not pkgs:
                self._append_log("pip list produced no JSON data; using importlib.metadata fallback...\n")
                pkgs = _collect_installed_importlib()
            self.model.set_packages(pkgs)
            self.proxy.sort(InstalledModel.COL_NAME, Qt.SortOrder.AscendingOrder)
            self.statusBar().showMessage(f"Loaded {len(pkgs)} packages — {self._format_env_label()}", 5000)
            self._drain_queue()
            return
        if last.startswith("index versions "):
            pkg = self._extract_pkg_from_index_cmd(last)
            self._handle_versions_text(pkg, text)
            self._drain_queue()
            return
        if last.startswith("install "):
            self._append_log("\n✅ Install finished.\n")
            self.refresh_installed()
            self._drain_queue()
            return
        if last.startswith("uninstall "):
            self._append_log("\n✅ Uninstall finished.\n")
            self.refresh_installed()
            self._drain_queue()
            return
        self._drain_queue()

    def _on_pip_err(self, _code: int, text: str) -> None:
        self._set_busy(False)
        self._append_log("\n❌ pip exited with non-zero status\n")
        last = self._last_pip_cmd_text() or ""
        if last.startswith("index versions "):
            pkg = self._extract_pkg_from_index_cmd(last)
            self._append_log("Falling back to PyPI JSON API…\n")
            self._fetch_versions_json(pkg)
        elif last.startswith("list "):
            self._append_log("Falling back to importlib.metadata to enumerate packages…\n")
            pkgs = _collect_installed_importlib()
            if pkgs:
                self.model.set_packages(pkgs)
                self.proxy.sort(InstalledModel.COL_NAME, Qt.SortOrder.AscendingOrder)
                self.statusBar().showMessage(f"Loaded {len(pkgs)} packages via importlib.metadata — {self._format_env_label()}", 5000)
        elif last.startswith("install ") and "==999999999" in last:
            # Parse versions from pip's error ("from versions: ...")
            text_no_ansi = _strip_ansi(text)
            m1 = re.search(r"from versions:\s*([^\n]+)", text_no_ansi, re.IGNORECASE)
            if not m1:
                m1 = re.search(r"available\s+versions?\s*:\s*(.+)$", text_no_ansi, re.IGNORECASE | re.MULTILINE)
            versions = []
            if m1:
                tail = m1.group(1)
                versions = [t.strip() for t in re.split(r"[,\\s]+", tail) if t.strip()]
            if versions:
                try:
                    pkg = re.search(r"install\s+([^=\s]+)==999999999", last).group(1)
                except Exception:
                    pkg = self._current_row_pkg() or "(unknown)"
                self._append_log(f"Recovered {len(versions)} versions via probe.\n")
                self._populate_version_list(pkg, versions)
            else:
                self._append_log("Could not recover versions via probe.\n")
        self._drain_queue()

    # Versions pipeline

    def _handle_versions_text(self, pkg: str, text: str) -> None:
        versions = self._parse_versions_from_pip_index(text)
        if not versions:
            self._append_log("Could not parse `pip index versions` output; trying PyPI JSON…\n")
            self._fetch_versions_json(pkg)
            # Queue probe as last resort
            self._pending.append(("_probe", [pkg]))
            return
        self._populate_version_list(pkg, versions)
        self._fetch_info_json(pkg)

    @staticmethod
    def _parse_versions_from_pip_index(text: str) -> List[str]:
        versions: List[str] = []
        m = re.search(r"(?im)available\s+versions?\s*:\s*(.+)$", text)
        if m:
            tail = m.group(1)
            versions = [t.strip() for t in re.split(r"[,\s]+", tail) if t.strip()]
        if not versions:
            versions = re.findall(r"\b\d+(?:\.[A-Za-z0-9]+)*\b", text)
        return versions

    # Networking (PyPI JSON)

    def _fetch_versions_json(self, pkg: str) -> None:
        if HAVE_QTNETWORK and self.nam is not None:
            url = QUrl(f"https://pypi.org/pypi/{pkg}/json")
            req = QNetworkRequest(url)
            req.setHeader(QNetworkRequest.KnownHeaders.UserAgentHeader, "PipVersionManager/1.0")
            reply = self.nam.get(req)
            reply.finished.connect(lambda r=reply, p=pkg: self._on_versions_reply(p, r))
            return
        # No QtNetwork: do nothing here; probe will handle

    def _fetch_info_json(self, pkg: str) -> None:
        if HAVE_QTNETWORK and self.nam is not None:
            url = QUrl(f"https://pypi.org/pypi/{pkg}/json")
            req = QNetworkRequest(url)
            req.setHeader(QNetworkRequest.KnownHeaders.UserAgentHeader, "PipVersionManager/1.0")
            reply = self.nam.get(req)
            reply.finished.connect(lambda r=reply: self._on_info_reply(r))

    def _on_versions_reply(self, pkg: str, reply: "QNetworkReply") -> None:
        if reply.error() != QNetworkReply.NetworkError.NoError:
            self._append_log(f"Network error: {reply.errorString()}\n")
            self._probe_versions_via_pip_install(pkg)
            reply.deleteLater()
            return
        try:
            data = json.loads(bytes(reply.readAll()).decode())
            releases = list(data.get("releases", {}).keys())
            if not releases:
                self._append_log("PyPI JSON returned no releases; probing via pip…\n")
                self._probe_versions_via_pip_install(pkg)
                reply.deleteLater()
                return
            versions = sorted(releases, key=version_key, reverse=True)
            self._populate_version_list(pkg, versions)
            self._populate_info_from_json(data)
        except Exception as e:
            self._append_log(f"Failed to parse PyPI JSON: {e}\n")
        finally:
            reply.deleteLater()

    def _on_info_reply(self, reply: "QNetworkReply") -> None:
        if reply.error() != QNetworkReply.NetworkError.NoError:
            self._append_log(f"Network error: {reply.errorString()}\n")
            reply.deleteLater()
            return
        try:
            data = json.loads(bytes(reply.readAll()).decode())
            self._populate_info_from_json(data)
        except Exception as e:
            self._append_log(f"Failed to parse PyPI JSON: {e}\n")
        finally:
            reply.deleteLater()

    def _populate_info_from_json(self, data: dict) -> None:
        info = data.get("info", {}) if isinstance(data, dict) else {}
        name = info.get("name") or ""
        summary = info.get("summary") or ""
        home = info.get("home_page") or ""
        rp = info.get("requires_python") or ""
        license_ = info.get("license") or ""
        proj_urls = info.get("project_urls") or {}

        lines = []
        if name:
            lines.append(f"**{name}**")
        if summary:
            lines.append(summary)
        if rp:
            lines.append(f"Requires-Python: {rp}")
        if license_:
            lines.append(f"License: {license_}")
        if home:
            lines.append(f"Home: {home}")
        if isinstance(proj_urls, dict) and proj_urls:
            lines.append("Project URLs: " + ", ".join(f"{k}: {v}" for k, v in proj_urls.items()))
        self.txtInfo.setPlainText("\n".join(lines) if lines else "(no metadata)")

    # Probe fallback

    def _probe_versions_via_pip_install(self, pkg: str) -> None:
        if self._busy:
            self._pending.append(("_probe", [pkg]))
            return
        args = ["install", f"{pkg}==999999999", "--disable-pip-version-check", "--no-color", "--no-deps"]
        if self.chkPre.isChecked():
            args.append("--pre")
        if self.txtIndex.text().strip():
            args += ["-i", self.txtIndex.text().strip()]
        if self.txtExtra.text().strip():
            args += ["--extra-index-url", self.txtExtra.text().strip()]
        self._append_log("Probing versions via pip install failure trick...\n")
        self._log_cmd(args)
        self._set_busy(True)
        self.proc.run(args)

    # Rendering

    def _populate_version_list(self, pkg: str, versions: List[str]) -> None:
        self.lblTarget.setText(f"Versions: {pkg}")
        if not self.chkPre.isChecked():
            versions = [v for v in versions if not is_prerelease(v)]
        versions = sorted(set(versions), key=version_key, reverse=True)

        cur = self._current_installed_version()
        cur_triplet = parse_semver_triplet(cur)

        self.listVersions.clear()
        for v in versions:
            item = QListWidgetItem(v)
            item.setData(Qt.ItemDataRole.UserRole, v)
            if cur and v != cur:
                v_triplet = parse_semver_triplet(v)
                color: Optional[QColor] = None
                if v_triplet[0] is not None and cur_triplet[0] is not None and v_triplet[0] != cur_triplet[0]:
                    color = QColor(Qt.GlobalColor.red)
                elif v_triplet[1] is not None and cur_triplet[1] is not None and v_triplet[1] != cur_triplet[1]:
                    color = QColor(Qt.GlobalColor.darkYellow)
                elif v_triplet[2] is not None and cur_triplet[2] is not None and v_triplet[2] != cur_triplet[2]:
                    color = QColor(Qt.GlobalColor.blue)
                if color is not None:
                    item.setForeground(QBrush(color))
            else:
                item.setText(f"{v}  ✓ current")
                f = item.font()
                f.setBold(True)
                item.setFont(f)
            self.listVersions.addItem(item)

        # Preselect current version if present
        if cur:
            matches = self.listVersions.findItems(cur, Qt.MatchFlag.MatchStartsWith)
            if matches:
                self.listVersions.setCurrentItem(matches[0])
        self._update_install_button_label()

    def _update_install_button_label(self) -> None:
        items = self.listVersions.selectedItems()
        cur = self._current_installed_version()
        if not items:
            self.btnInstall.setText("Install")
            self.btnInstall.setEnabled(False)
            return
        v = items[0].data(Qt.ItemDataRole.UserRole) or ""
        if cur and v == cur:
            self.btnInstall.setText("Already installed")
            self.btnInstall.setEnabled(False)
        else:
            if cur:
                verb = "Upgrade" if version_key(v) > version_key(cur) else "Downgrade"
            else:
                verb = "Install"
            self.btnInstall.setText(f"{verb} to {v}")
            self.btnInstall.setEnabled(not self._busy)

    # UI slots

    def _on_filter(self, text: str) -> None:
        self.proxy.setFilterFixedString(text)

    def _on_row_selected(self, _sel: QItemSelection, _desel: QItemSelection) -> None:
        pkg = self._current_row_pkg()
        if not pkg:
            self.lblTarget.setText("Versions: -")
            self.listVersions.clear()
            self.txtInfo.clear()
            return
        self.request_versions(pkg)
        self._fetch_info_json(pkg)

    def _on_version_selected(self) -> None:
        self._update_install_button_label()

    def _on_table_ctx(self, pos) -> None:
        idx = self.table.indexAt(pos)
        if not idx.isValid():
            return
        src = self.proxy.mapToSource(idx)
        pkg = self.model.item(src.row(), InstalledModel.COL_NAME).text()
        menu = QMenu(self)
        actCopy = QAction("Copy name", self)
        actShow = QAction("Show versions", self)
        actCopy.triggered.connect(lambda: QApplication.clipboard().setText(pkg))
        actShow.triggered.connect(lambda: self.request_versions(pkg))
        menu.addAction(actCopy)
        menu.addAction(actShow)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    # Logging

    def _log_cmd(self, args: List[str]) -> None:
        quoted = " ".join(repr(a) if " " in a else a for a in args)
        self._append_log(f"$ {sys.executable} -m pip {quoted}\n")

    # Queue

    def _drain_queue(self) -> None:
        if self._busy or not self._pending:
            return
        label, args = self._pending.pop(0)
        if label == "list":
            self.refresh_installed()
        elif label == "versions":
            self.request_versions(args[0])
        elif label == "_probe":
            self._probe_versions_via_pip_install(args[0])


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()