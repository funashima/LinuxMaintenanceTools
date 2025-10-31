# Lab Maintenance & Package Tools

A small collection of PyQt6 desktop utilities to help maintain Linux lab machines and Python environments. All tools are GUI-first, designed to be safe for students/TA-level users, and to make “what will this command actually do?” visible before changes are applied.

Author: **Hiroki Funashima**
License: **MIT**

---

## Tools in this repo

### 1. DepAtlas — Python dependency graph visualizer

A PyQt6 app that scans installed Python packages and lets you **build & visualize** dependency or *reverse* dependency graphs. It supports multiple layouts (CPU/GPU), clustering, zoom/pan, and export. Useful for teaching, auditing large envs, or debugging messy dependency chains. 

**Key features**

* List installed packages and pick “roots”
* Build *dependencies* (what this needs) or *dependents* (who needs this)
* Layout on CPU (spring / Kamada–Kawai) or GPU (cuGraph ForceAtlas2)
* Spectral / k-means / DBSCAN clustering, with auto re-layout
* Zoom (wheel) & pan (drag)
* Export figure to PNG/SVG (for reports/slides)

---

### 2. Pacifica — Simple APT Manager

A PyQt6 wrapper around common `apt` operations for Ubuntu/Debian labs. It does `sudo apt-get update` → `apt list --upgradable` automatically on startup, shows upgradable packages in a table with color-coded version bumps, lets you upgrade selected ones, and (new) lets you **check and run `apt autoremove`** from the GUI. All apt commands are run with `LC_ALL=C` / `LANG=C` so parsing is stable across locale settings. 

**Key features**

* Auto-refresh upgradable list at startup
* Upgrade selected packages with progress
* “Refresh list after upgrade” toggle (default ON)
* **NEW:** “Check autoremove” → runs `sudo apt-get autoremove --dry-run`, shows candidates, and runs `sudo apt-get autoremove -y` on confirmation
* Sudo password dialog with “remember for this session”
* One `QProcess` at a time → safer for lab machines

---

### 3. PipLens — pip-review GUI upgrader

A GUI on top of `pip-review` / `pip list --outdated`. It fetches upgradable Python packages, shows them in a table with per-row checkboxes, colorizes version bumps, and upgrades only what you select. After upgrading, it can auto-refresh. Good for venvs, Jupyter machines, or instructor laptops. 

**Key features**

* Multiple fetch strategies (`pip-review --format json`, text mode, `pip list --outdated`)
* Select all / clear, per-row dimming
* Upgrade selected with live log + progress bar
* Auto-refresh after upgrade (toggle)
* Shows whether you are inside a venv

---

### 4. Pip Version Manager

A PyQt6 tool to browse installed packages, query PyPI for all available versions, and install a selected version. Includes a small dependency-graph view and several safety rails (e.g. hiding yanked versions by default). Useful when you have to **pin a specific version for an experiment or for a student’s environment.** 

**Key features**

* List installed packages (importlib.metadata)
* Fetch versions from PyPI
* “Include yanked” checkbox (off by default)
* Install selected version with confirmation + live log
* Simple dependency graph (NetworkX + Matplotlib) with zoom/pan

---

## Requirements

* Python **3.10+** (3.11+ recommended)
* **PyQt6**
* For graph features (DepAtlas, Pip Version Manager):

  * `networkx`
  * `matplotlib`
  * Optionally `cugraph` / `cudf` (if you want GPU layouts)
* For PipLens:

  * `pip-review`
  * `packaging`

Example:

```bash
python -m pip install PyQt6 networkx matplotlib packaging pip-review
```

(Install `cudf` / `cugraph` only if your environment supports it.)

---

## Running

Each tool is a standalone script:

```bash
python DepAtlas.py
python Pacifica.py
python PipLens.py
python PipVersionManager.py
```

On Ubuntu/Debian, **Pacifica** will ask for your sudo password in a dialog before running `apt-get ...`. Keep in mind that upgrading or autoremove can actually change the system.

---

## Repository structure

* `DepAtlas.py` — dependency graph visualizer (Python/pip)
* `Pacifica.py` — apt GUI for lab machines
* `PipLens.py` — pip-review GUI
* `PipVersionManager.py` — per-package, per-version installer/visualizer

You can keep them all in one public repo and run only what you need.

---

## License

MIT License

Copyright (c) 2025 Hiroki Funashima

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

---

## Copyright

**© Hiroki Funashima**
You may use, modify, and redistribute under the terms of the MIT License. If you use these tools in a paper, lecture, or course material, please credit the author.

