#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple APT Manager (PyQt6) — sudo prompt, version color highlighting, and progress bars

Features:
- Buttons: "apt update", "apt list --upgradable", "Upgrade Selected", "Select All", "Clear All"
- Table columns: Select / Package / Current Version / Candidate Version / Arch / Origin
- Always prompts for sudo password before privileged ops (update, upgrade selected)
- Optional in-session password remember
- Highlights Current/Candidate cells by version difference level:
    Major -> light red (#ffe5e5)
    Minor -> light orange (#fff0e0)
    Patch -> light blue (#e7f0ff)
- Progress bar:
    * During `apt update`: indeterminate (busy) indicator
    * During upgrades: coarse % based on "Unpacking" + "Setting up" log lines per package

Run:
  python3 apt_manager.py
"""

import sys
import re
from typing import List, Tuple, Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QFontDatabase

# --- Parsing of `apt list --upgradable` lines ---
APT_LIST_PATTERN = re.compile(
    r"""^
    (?P<pkg>[^/]+)/(?P<origin>\S+)\s+
    (?P<candidate>[^\s]+)\s+
    (?P<arch>\S+)\s+
    \[upgradable\s+from:\s+(?P<current>[^\]]+)\]
    """,
    re.VERBOSE,
)

# --- Version comparison helpers (SemVer-like extraction from Debian-ish versions) ---
SEMVER_NUM = re.compile(r"(\d+)")

def _semver_parts(ver: str):
    """Extract up to three integer parts (major, minor, patch) from a Debian-like version.
    - Drop epoch (X:...)
    - Drop debian revision (-...)
    - From the upstream part, take first three numeric groups; pad with zeros.
    """
    if ":" in ver:
        ver = ver.split(":", 1)[1]
    if "-" in ver:
        ver = ver.split("-", 1)[0]
    nums = [int(m.group(1)) for m in SEMVER_NUM.finditer(ver)]
    while len(nums) < 3:
        nums.append(0)
    return nums[:3]

def version_diff_level(current: str, candidate: str) -> str:
    """Classify diff as 'major' / 'minor' / 'patch' / 'same' based on first differing part."""
    a = _semver_parts(current)
    b = _semver_parts(candidate)
    if a == b:
        return "same"
    if a[0] != b[0]:
        return "major"
    if a[1] != b[1]:
        return "minor"
    return "patch"

# --------------------------
# Sudo password dialog
# --------------------------
class PasswordDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, remember_default=True):
        super().__init__(parent)
        self.setWindowTitle("sudo password required")
        self.setModal(True)
        self.setMinimumWidth(420)

        lbl = QtWidgets.QLabel("Enter your sudo password to continue:")
        self.edit = QtWidgets.QLineEdit()
        self.edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.edit.setPlaceholderText("Password")
        self.edit.returnPressed.connect(self.accept)

        self.chk_show = QtWidgets.QCheckBox("Show password")
        self.chk_show.toggled.connect(self._toggle_echo)

        self.chk_remember = QtWidgets.QCheckBox("Remember for this session")
        self.chk_remember.setChecked(remember_default)

        self.lbl_error = QtWidgets.QLabel("")
        self.lbl_error.setStyleSheet("color: #c62828;")
        self.lbl_error.setVisible(False)

        btn_ok = QtWidgets.QPushButton("OK")
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(lbl, 0, 0, 1, 2)
        grid.addWidget(self.edit, 1, 0, 1, 2)
        grid.addWidget(self.chk_show, 2, 0, 1, 1)
        grid.addWidget(self.chk_remember, 2, 1, 1, 1)
        grid.addWidget(self.lbl_error, 3, 0, 1, 2)

        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)

        v = QtWidgets.QVBoxLayout(self)
        v.addLayout(grid)
        v.addStretch(1)
        v.addLayout(btns)

    def _toggle_echo(self, checked: bool):
        self.edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Normal if checked else QtWidgets.QLineEdit.EchoMode.Password)

    def get_password(self):
        return self.edit.text().strip()

    def set_error(self, msg: str):
        self.lbl_error.setText(msg)
        self.lbl_error.setVisible(True)

# --------------------------
# Main window
# --------------------------
class AptManager(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple APT Manager")
        self.resize(1000, 760)

        # Buttons
        self.btn_update = QtWidgets.QPushButton("apt update")
        self.btn_list = QtWidgets.QPushButton("apt list --upgradable")
        self.btn_upgrade_selected = QtWidgets.QPushButton("Upgrade Selected")
        self.btn_select_all = QtWidgets.QPushButton("Select All")
        self.btn_clear_all = QtWidgets.QPushButton("Clear All")

        # Table
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            "Select", "Package", "Current Version", "Candidate Version", "Arch", "Origin"
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        for col in range(1, 6):
            self.table.horizontalHeader().setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)

        # Progress bar (top of the log)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.reset()

        # Log
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Logs will appear here...")

        # Layout
        top_bar = QtWidgets.QHBoxLayout()
        top_bar.addWidget(self.btn_update)
        top_bar.addWidget(self.btn_list)
        top_bar.addSpacing(16)
        top_bar.addWidget(self.btn_upgrade_selected)
        top_bar.addStretch(1)
        top_bar.addWidget(self.btn_select_all)
        top_bar.addWidget(self.btn_clear_all)

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addLayout(top_bar)
        vbox.addWidget(self.table, stretch=1)
        vbox.addWidget(self.progress, stretch=0)
        vbox.addWidget(self.log, stretch=1)

        # QProcess used for apt/dpkg commands
        self.proc = QtCore.QProcess(self)
        self.proc.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_proc_output)
        self.proc.finished.connect(self._on_proc_finished)

        # Password cache (in-memory for this session only)
        self._session_pw_cache: Optional[bytes] = None
        self._sudo_password_bytes: Optional[bytes] = None

        # Callback after a non-list command finishes
        self._pending_on_finish = None

        # Progress state
        self._progress_mode = "idle"            # "idle" | "busy" | "upgrade"
        self._progress_total = 0
        self._progress_done = 0
        self._seen_unpack = set()
        self._seen_setup = set()
        # Row shading color for unchecked items (slightly dark gray)
        self._unchecked_bg = QtGui.QColor("#e0e0e0")

        # Wire buttons
        self.btn_update.clicked.connect(self.run_apt_update)
        self.btn_list.clicked.connect(self.run_apt_list_upgradable)
        self.btn_upgrade_selected.clicked.connect(self.upgrade_selected)
        self.btn_select_all.clicked.connect(self.select_all)
        self.btn_clear_all.clicked.connect(self.clear_all)

    # --------------------------
    # Button slots
    # --------------------------
    def run_apt_update(self):
        if self.proc.state() != QtCore.QProcess.ProcessState.NotRunning:
            return
        self.log.clear()
        self.append_log(">> Running: sudo apt-get update")
        self._progress_busy("Fetching package indexes...")
        self._sudo_run(["apt-get", "update"])

    def run_apt_list_upgradable(self):
        if self.proc.state() != QtCore.QProcess.ProcessState.NotRunning:
            return
        self.log.clear()
        self.append_log(">> Running: apt list --upgradable")
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("LC_ALL", "C")
        self.proc.setProcessEnvironment(env)
        self._start_process(["bash", "-lc", "apt list --upgradable 2>/dev/null"],
                            on_finish=self._populate_table_from_list)

    def upgrade_selected(self):
        if self.proc.state() != QtCore.QProcess.ProcessState.NotRunning:
            return
        packages = self._gather_selected_packages()
        if not packages:
            QtWidgets.QMessageBox.information(self, "No Selection", "Please select at least one package.")
            return
        self.log.clear()
        self.append_log(">> Running: sudo apt-get install --only-upgrade -y <selected>")
        self._progress_start_upgrade(packages)
        self._sudo_run(["apt-get", "install", "--only-upgrade", "-y"] + packages)

    def select_all(self):
        for row in range(self.table.rowCount()):
            chk = self.table.cellWidget(row, 0)
            if isinstance(chk, QtWidgets.QWidget):
                cb = chk.findChild(QtWidgets.QCheckBox)
                if cb:
                    cb.setChecked(True)
                    self._shade_row(row, True)

    def clear_all(self):
        for row in range(self.table.rowCount()):
            chk = self.table.cellWidget(row, 0)
            if isinstance(chk, QtWidgets.QWidget):
                cb = chk.findChild(QtWidgets.QCheckBox)
                if cb:
                    cb.setChecked(False)
                    self._shade_row(row, False)

    # --------------------------
    # Row shading helper
    # --------------------------
    def _shade_row(self, row: int, checked: bool):
        """Shade the entire row gray when unchecked; restore colors when checked."""
        # Column 0 is a QWidget (checkbox); 1..5 are QTableWidgetItems
        # Shade checkbox cell background via stylesheet
        w = self.table.cellWidget(row, 0)
        if isinstance(w, QtWidgets.QWidget):
            w.setStyleSheet("") if checked else w.setStyleSheet(f"background: {self._unchecked_bg.name()};")

        # Package name
        it_pkg = self.table.item(row, 1)
        it_cur = self.table.item(row, 2)
        it_cand = self.table.item(row, 3)
        it_arch = self.table.item(row, 4)
        it_origin = self.table.item(row, 5)
        items = [it_pkg, it_cur, it_cand, it_arch, it_origin]

        if not all(items):
            return

        if not checked:
            for it in items:
                it.setBackground(self._unchecked_bg)
            return

        # Restore original coloring for checked rows
        # Clear non-version cells
        it_pkg.setBackground(QtGui.QBrush())
        it_arch.setBackground(QtGui.QBrush())
        it_origin.setBackground(QtGui.QBrush())
        # Re-apply version diff coloring
        cur = it_cur.text()
        cand = it_cand.text()
        lvl = version_diff_level(cur, cand)
        if lvl == "major":
            color = QtGui.QColor("#ffe5e5")
            it_cur.setBackground(color)
            it_cand.setBackground(color)
        elif lvl == "minor":
            color = QtGui.QColor("#fff0e0")
            it_cur.setBackground(color)
            it_cand.setBackground(color)
        elif lvl == "patch":
            color = QtGui.QColor("#e7f0ff")
            it_cur.setBackground(color)
            it_cand.setBackground(color)
        else:
            it_cur.setBackground(QtGui.QBrush())
            it_cand.setBackground(QtGui.QBrush())

    # --------------------------
    # Process helpers
    # --------------------------
    def _start_process(self, args: List[str], on_finish=None):
        """Start process with given args; connect finish callback."""
        self._pending_on_finish = on_finish
        self._set_controls_enabled(False)
        self.proc.setProgram(args[0])
        self.proc.setArguments(args[1:])
        self.proc.start()

    def _set_controls_enabled(self, ok: bool):
        self.btn_update.setEnabled(ok)
        self.btn_list.setEnabled(ok)
        self.btn_upgrade_selected.setEnabled(ok)
        self.btn_select_all.setEnabled(ok)
        self.btn_clear_all.setEnabled(ok)

    def _sudo_run(self, cmd_parts: List[str]):
        """Prepare `sudo -S` command and prompt for password if needed."""
        if self._session_pw_cache:
            self._sudo_password_bytes = self._session_pw_cache
            self._start_sudo_with_password(cmd_parts)
            return
        self._prompt_password_then(lambda pw_bytes, remember: self._maybe_run_sudo(cmd_parts, pw_bytes, remember))

    def _maybe_run_sudo(self, cmd_parts: List[str], pw_bytes: Optional[bytes], remember: bool):
        if not pw_bytes:
            return
        self._sudo_password_bytes = pw_bytes
        if remember:
            self._session_pw_cache = pw_bytes
        self._start_sudo_with_password(cmd_parts)

    def _start_sudo_with_password(self, cmd_parts: List[str]):
        """Run sudo directly via QProcess and feed the password on stdin."""
        program = "sudo"
        args = ["-S", "--"] + cmd_parts
        self._start_process([program] + args, on_finish=lambda: self._check_sudo_result(cmd_parts))
        if self._sudo_password_bytes:
            self.proc.write(self._sudo_password_bytes + b"\n")  # sudo expects newline-terminated password

    def _prompt_password_then(self, cont):
        dlg = PasswordDialog(self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            pw = dlg.get_password()
            if pw:
                cont(pw.encode("utf-8"), dlg.chk_remember.isChecked())
            else:
                dlg.set_error("Password cannot be empty.")

    def _check_sudo_result(self, cmd_parts: List[str]):
        """Inspect output to detect wrong password; if so, re-prompt."""
        text = self.log.toPlainText()
        wrong_pw = any(s in text for s in [
            "Sorry, try again.",
            "sudo: 1 incorrect password attempt",
            "sudo: 2 incorrect password attempts",
            "sudo: 3 incorrect password attempts",
        ])
        not_in_sudoers = "is not in the sudoers file" in text

        if not_in_sudoers:
            QtWidgets.QMessageBox.critical(self, "sudo error",
                                           "Your user is not in the sudoers file. Cannot continue.")
            return

        if wrong_pw:
            # Clear any cached wrong password
            self._session_pw_cache = None
            self._sudo_password_bytes = None
            # Re-prompt
            self.append_log(">> Incorrect password. Please try again.")
            self._prompt_password_then(lambda pw_bytes, remember: self._retry_sudo(cmd_parts, pw_bytes, remember))

    def _retry_sudo(self, cmd_parts: List[str], pw_bytes: Optional[bytes], remember: bool):
        if not pw_bytes:
            return
        self._sudo_password_bytes = pw_bytes
        if remember:
            self._session_pw_cache = pw_bytes
        self._start_sudo_with_password(cmd_parts)

    # --------------------------
    # Progress helpers
    # --------------------------
    def _progress_reset(self):
        self._progress_mode = "idle"
        self._progress_total = 0
        self._progress_done = 0
        self._seen_unpack.clear()
        self._seen_setup.clear()
        self.progress.reset()

    def _progress_busy(self, msg: str):
        self._progress_mode = "busy"
        self.progress.setRange(0, 0)  # 0,0 shows indeterminate busy indicator
        self.progress.setFormat(msg)

    def _progress_start_upgrade(self, pkgs: List[str]):
        self._progress_mode = "upgrade"
        self._progress_total = max(1, 2 * len(pkgs))  # Unpack + Setup per package
        self._progress_done = 0
        self._seen_unpack.clear()
        self._seen_setup.clear()
        self.progress.setRange(0, self._progress_total)
        self.progress.setValue(0)
        self.progress.setFormat("Upgrading... %p%")

    def _progress_tick(self, kind: str, pkg: str):
        if self._progress_mode != "upgrade":
            return
        if kind == "unpack":
            if pkg and pkg not in self._seen_unpack:
                self._seen_unpack.add(pkg)
                self._progress_done += 1
        elif kind == "setup":
            if pkg and pkg not in self._seen_setup:
                self._seen_setup.add(pkg)
                self._progress_done += 1
        self.progress.setValue(min(self._progress_done, self._progress_total))

    # --------------------------
    # Process output/finish
    # --------------------------
    def _on_proc_output(self):
        data = self.proc.readAllStandardOutput().data().decode(errors="replace")
        if data:
            self.append_log(data)

            # Optional: if sudo prompts again, re-feed the password once
            if "sudo" in data.lower() and "password" in data.lower():
                if self._sudo_password_bytes:
                    try:
                        self.proc.write(self._sudo_password_bytes + b"\n")
                    except Exception:
                        pass

            # Progress detection while upgrading (coarse but user-friendly)
            if self._progress_mode == "upgrade":
                for line in data.splitlines():
                    s = line.strip()
                    low = s.lower()
                    # Typical dpkg/apt lines:
                    #   "Unpacking <pkg> ..." or "Unpacking <pkg> (ver) over ..."
                    #   "Setting up <pkg> (ver) ..."
                    #   "Processing triggers for <pkg> ..."
                    if low.startswith("unpacking "):
                        parts = s.split()
                        if len(parts) >= 2:
                            self._progress_tick("unpack", parts[1])
                    elif low.startswith("setting up "):
                        parts = s.split()
                        # parts[2] is the package when the line is exactly "Setting up <pkg> ..."
                        if len(parts) >= 3:
                            self._progress_tick("setup", parts[2] if parts[1] == "up" else parts[1])
                    elif low.startswith("processing triggers for "):
                        # Nudge forward a little; not strictly tied to a package phase
                        self._progress_done = min(self._progress_done + 1, self._progress_total)
                        self.progress.setValue(self._progress_done)

    def _on_proc_finished(self, exit_code: int, exit_status: QtCore.QProcess.ExitStatus):
        self._set_controls_enabled(True)
        # Make the progress bar show completion, then reset
        if self._progress_mode in ("busy", "upgrade"):
            self.progress.setRange(0, 1)
            self.progress.setValue(1)
            self.progress.setFormat("Done")
        if exit_status == QtCore.QProcess.ExitStatus.CrashExit:
            self.append_log("!! Process crashed.")
        else:
            self.append_log(f">> Process finished with exit code {exit_code}.")
        if self._pending_on_finish:
            cb = self._pending_on_finish
            self._pending_on_finish = None
            try:
                cb()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Post-process Error", str(e))
        self._progress_reset()

    # --------------------------
    # Table helpers
    # --------------------------
    def _populate_table_from_list(self):
        text = self.log.toPlainText()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        entries: List[Tuple[str, str, str, str, str]] = []
        for ln in lines:
            m = APT_LIST_PATTERN.match(ln)
            if m:
                pkg = m.group("pkg")
                origin = m.group("origin")
                candidate = m.group("candidate")
                arch = m.group("arch")
                current = m.group("current")
                entries.append((pkg, current, candidate, arch, origin))

        self.table.setRowCount(0)
        if not entries:
            self.append_log(">> No parsable upgradable entries found. System may be up to date.")
            return

        self.table.setRowCount(len(entries))
        for row, (pkg, current, candidate, arch, origin) in enumerate(entries):
            # Checkbox in column 0
            chk = QtWidgets.QCheckBox()
            chk.setChecked(True)
            chk_widget = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(chk_widget)
            layout.setContentsMargins(8, 0, 0, 0)
            layout.addWidget(chk)
            layout.addStretch(1)
            self.table.setCellWidget(row, 0, chk_widget)
            # Shade on toggle
            cb = chk
            cb.toggled.connect(lambda checked, r=row: self._shade_row(r, checked))

            # Package name
            it_pkg = QtWidgets.QTableWidgetItem(pkg)
            self.table.setItem(row, 1, it_pkg)

            # Versions
            it_cur = QtWidgets.QTableWidgetItem(current)
            it_cand = QtWidgets.QTableWidgetItem(candidate)

            # Color by diff level
            lvl = version_diff_level(current, candidate)
            if lvl == "major":
                it_cur.setBackground(QtGui.QColor("#ffe5e5"))
                it_cand.setBackground(QtGui.QColor("#ffe5e5"))
            elif lvl == "minor":
                it_cur.setBackground(QtGui.QColor("#fff0e0"))
                it_cand.setBackground(QtGui.QColor("#fff0e0"))
            elif lvl == "patch":
                it_cur.setBackground(QtGui.QColor("#e7f0ff"))
                it_cand.setBackground(QtGui.QColor("#e7f0ff"))

            self.table.setItem(row, 2, it_cur)
            self.table.setItem(row, 3, it_cand)

            # Arch, Origin
            self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(arch))
            self.table.setItem(row, 5, QtWidgets.QTableWidgetItem(origin))

        self.table.resizeRowsToContents()
        # Ensure current visual state reflects the initial checkbox (checked by default)
        for r in range(self.table.rowCount()):
            self._shade_row(r, True)

    def _gather_selected_packages(self) -> List[str]:
        pkgs = []
        for row in range(self.table.rowCount()):
            w = self.table.cellWidget(row, 0)
            if not w:
                continue
            chk = w.findChild(QtWidgets.QCheckBox)
            if chk and chk.isChecked():
                item = self.table.item(row, 1)
                if item:
                    pkgs.append(item.text())
        return pkgs

    def append_log(self, text: str):
        """Append text to the log area (preserving existing content)."""
        self.log.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        self.log.insertPlainText(text if text.endswith("\n") else text + "\n")
        self.log.moveCursor(QtGui.QTextCursor.MoveOperation.End)

# --------------------------
# App bootstrap
# --------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)

    # Try to use a monospaced font for better alignment of logs/table
    preferred = [
        "Optima", "Futura",
        "Fira Code", "JetBrains Mono", "Cascadia Code", "Source Code Pro",
        "Noto Sans Mono CJK JP", "Noto Sans Mono", "DejaVu Sans Mono",
        "Monaco", "Consolas", "Liberation Mono", "Menlo", "Courier New",
        "Monospace",
    ]
    fams = set(QFontDatabase.families())
    fontsize = 11
    chosen = next((f for f in preferred if f in fams), "Monospace")
    font = QtGui.QFont(chosen, fontsize)
    font.setStyleHint(QtGui.QFont.StyleHint.TypeWriter)
    app.setFont(font)

    w = AptManager()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

