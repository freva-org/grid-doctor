"""
Fetch, vendor and embed the Freva Data Browser into the Waterpark (data) docs.

The host loader (``docs/assets/databrowser-loader.js``) calls ``mountDataBrowser``
with the settings in ``docs/assets/databrowser.config.js``.

IMPORTANT: Runtime settings live in ``docs/assets/databrowser.config.js``.
Env overrides:
  ``DATABROWSER_VERSION``            npm version/dist-tag (default ``latest``)
  ``DATABROWSER_PACKAGE``            package name (default ``@freva-org/databrowser``)
  ``DATABROWSER_INSPECTOR_VERSION``  inspector version/tag (default ``latest``)
  ``DATABROWSER_INSPECTOR_PACKAGE``  inspector pkg (default ``@freva-org/data-inspector``)
  ``WATERPARK_BASE_PATH``            URL base, e.g. ``/`` or ``/grid-doctor``
  ``GRID_DOCTOR_ROOT``              repo root override
"""
from __future__ import annotations

import html
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Locations
SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR.parent / "assets"
REPO_ROOT = Path(os.environ.get("GRID_DOCTOR_ROOT", SCRIPT_DIR.parents[1])).resolve()

CONFIG_FILE = ASSETS_DIR / "databrowser.config.js"

# Isolated install dir
INSTALL_DIR = REPO_ROOT / ".build" / "databrowser-src"
BUILD_DATA = REPO_ROOT / ".build" / "data"

PACKAGE = os.environ.get("DATABROWSER_PACKAGE", "@freva-org/databrowser").strip()
# The lazy file-inspector web component
INSPECTOR_PACKAGE = os.environ.get(
    "DATABROWSER_INSPECTOR_PACKAGE", "@freva-org/data-inspector").strip()
# latest always
VERSION = os.environ.get("DATABROWSER_VERSION", "latest").strip() or "latest"
INSPECTOR_VERSION = os.environ.get(
    "DATABROWSER_INSPECTOR_VERSION", "latest").strip() or "latest"


# URL layout
def normalize_base_path(value: str | None) -> str:
    """'' for root, else a single leading slash and no trailing slash."""
    value = (value or "/").strip()
    return "" if value in {"", "/"} else "/" + value.strip("/")


BASE_PATH = normalize_base_path(os.environ.get("WATERPARK_BASE_PATH", "/"))
ASSET_URL = f"{BASE_PATH}/databrowser"
INSPECTOR_URL = f"{BASE_PATH}/databrowser-inspector"
CONFIG_URL = f"{BASE_PATH}/assets/databrowser.config.js"


def _run(cmd: list[str], cwd: Path | None = None, env: dict | None = None,
         capture: bool = False) -> str:
    print(f"[databrowser] $ {' '.join(cmd)}")
    if capture:
        out = subprocess.run(cmd, cwd=cwd, env=env, check=True,
                             capture_output=True, text=True).stdout.strip()
        print(f"[databrowser]   -> {out}")
        return out
    subprocess.run(cmd, cwd=cwd, env=env, check=True)
    return ""


def install_packages() -> dict[str, Path]:
    """
    Install ``PACKAGE`` and ``INSPECTOR_PACKAGE`` into
    an isolated dir
    """
    npm = shutil.which("npm")
    if not npm:
        sys.exit("[databrowser] npm not found; the docs build needs Node.js.")

    if INSTALL_DIR.exists():
        shutil.rmtree(INSTALL_DIR)
    INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    # A minimal private package.json keeps npm from walking up to a parent
    # manifest and avoids the "no lockfile in CI" refusal path.
    (INSTALL_DIR / "package.json").write_text(json.dumps(
        {"name": "waterpark-databrowser-vendor", "private": True,
         "version": "0.0.0"}, indent=2))

    specs = [f"{PACKAGE}@{VERSION}", f"{INSPECTOR_PACKAGE}@{INSPECTOR_VERSION}"]
    print(f"[databrowser] installing {', '.join(specs)} "
          f"(latest-always unless pinned)")
    # esbuild bundles the browser's multi-file ESM tree into ONE self-contained
    # file 
    _run([npm, "install", *specs, "esbuild@0.24.2",
          "--no-audit", "--no-fund", "--silent"], cwd=INSTALL_DIR)

    dists: dict[str, Path] = {}
    for name in (PACKAGE, INSPECTOR_PACKAGE):
        pkg_dir = INSTALL_DIR / "node_modules" / name
        dist = pkg_dir / "dist"
        if not dist.exists():
            sys.exit(f"[databrowser] installed package has no dist/: {dist}")
        resolved = json.loads((pkg_dir / "package.json").read_text()).get("version")
        print(f"[databrowser] resolved {name} -> {resolved}")
        dists[name] = dist
    return dists


def entry_file(dist: Path) -> str:
    """
    The ESM entry, taken from the package's own ``exports``/``module``/``main``.
    Both packages publish a browser-ready ESM entry
    """
    manifest = json.loads((dist.parent / "package.json").read_text())
    raw = (
        (manifest.get("exports", {}).get(".", {}) or {}).get("import")
        or manifest.get("module")
        or manifest.get("main")
        or "./dist/index.js"
    )
    return Path(raw).name


BROWSER_BUNDLE = "index.js" 


def bundle_browser(dist: Path) -> str:
    """
    Bundle the browser's multi-file ESM tree into ONE self-contained file at
    .build/data/databrowser/index.js
    """
    npm = shutil.which("npm")
    esbuild = INSTALL_DIR / "node_modules" / ".bin" / "esbuild"
    if not esbuild.exists():
        sys.exit(f"[databrowser] esbuild not found at {esbuild}")

    target_dir = BUILD_DATA / "databrowser"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / BROWSER_BUNDLE

    _run([str(esbuild), str(dist / "index.js"),
          "--bundle", "--format=esm", "--minify",
          "--legal-comments=none", f"--outfile={out}"])
    size_kb = out.stat().st_size / 1024
    print(f"[databrowser] bundled browser -> {out} ({size_kb:.0f} KB, 1 request)")
    return BROWSER_BUNDLE


def copy_single(dist: Path, entry: str, subdir: str) -> None:
    """Copy a single already-bundled entry file into .build/data/<subdir>/."""
    target = BUILD_DATA / subdir
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dist / entry, target / entry)
    print(f"[databrowser] copied {entry} -> {target}")


def render_page(entry_js: str, inspector_mjs: str) -> str:
    """Emit the mkdocs page: a sized mount div; the host loader does the rest."""
    entry_url = html.escape(f"{ASSET_URL}/{entry_js}", quote=True)
    inspector_url = html.escape(f"{INSPECTOR_URL}/{inspector_mjs}", quote=True)
    config_url = html.escape(CONFIG_URL, quote=True)
    return f"""---
title: Data Browser
hide:
  - navigation
  - toc
---

<!-- AUTO-GENERATED by docs/scripts/build_databrowser_docs.py. Do not edit;
     settings live in docs/assets/databrowser.config.js. -->

<div class="waterpark-databrowser-wrapper">
  <div id="databrowser-mount"
       data-databrowser-entry="{entry_url}"
       data-databrowser-inspector="{inspector_url}"
       data-databrowser-config="{config_url}"></div>
</div>
"""


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "tech"
    if target != "data":
        print(f"[databrowser] target={target!r}: nothing to do.")
        return

    if not BUILD_DATA.exists():
        sys.exit(
            f"[databrowser] .build/data missing ({BUILD_DATA}). Run via tox "
            f"after docs-prep, or set GRID_DOCTOR_ROOT. Repo root: {REPO_ROOT}."
        )

    if not CONFIG_FILE.exists():
        print(f"[databrowser] ::warning:: {CONFIG_FILE.name} not found; "
              f"the loader will fall back to built-in defaults.")

    dists = install_packages()
    entry_js = bundle_browser(dists[PACKAGE])
    inspector_mjs = entry_file(dists[INSPECTOR_PACKAGE])
    copy_single(dists[INSPECTOR_PACKAGE], inspector_mjs, "databrowser-inspector")
    (BUILD_DATA / "databrowser.md").write_text(
        render_page(entry_js, inspector_mjs))
    print(f"[databrowser] embedded OK (entry={entry_js}, "
          f"inspector={inspector_mjs})")


if __name__ == "__main__":
    main()
