"""
Vendor, patch, build and embed STAC Browser into the Waterpark (data) docs.

IMPORTANT: Settings live in ``docs/assets/stac-browser.config.js``.
Env overrides: ``STAC_CATALOG_URL`` (catalog URL only),
``WATERPARK_BASE_PATH`` (URL base, e.g. ``/`` or ``/grid-doctor``),
``STAC_BROWSER_REPO`` / ``STAC_BROWSER_REF`` (source), ``GRID_DOCTOR_ROOT``
"""
from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Locations
SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR.parent / "assets"
REPO_ROOT = Path(os.environ.get("GRID_DOCTOR_ROOT", SCRIPT_DIR.parents[1])).resolve()

PATCH_DIR = ASSETS_DIR / "stac-browser-patches"
PIN_FILE = PATCH_DIR / "stac-browser.commit"
CONFIG_FILE = ASSETS_DIR / "stac-browser.config.js"

SRC_DIR = REPO_ROOT / ".build" / "stac-browser-src"
BUILD_DATA = REPO_ROOT / ".build" / "data"

# URL layout
def normalize_base_path(value: str | None) -> str:
    """'' for root, else a single leading slash and no trailing slash."""
    value = (value or "/").strip()
    return "" if value in {"", "/"} else "/" + value.strip("/")

BASE_PATH = normalize_base_path(os.environ.get("WATERPARK_BASE_PATH", "/"))
PATH_PREFIX = f"{BASE_PATH}/stac-browser/"
ASSET_URL = f"{BASE_PATH}/stac-browser"

STAC_REPO = os.environ.get("STAC_BROWSER_REPO",
                           "https://github.com/radiantearth/stac-browser.git")
CATALOG_URL_ENV = os.environ.get("STAC_CATALOG_URL", "").strip()

# Prefixes one compiled CSS file in place so STAC's globally-imported Bootstrap
# can't bleed onto the Material chrome. Global element rules are scoped under the
# always-present #stac-browser-mount host; STAC's own #stac-browser rules and
# @keyframes steps are left intact.
PREFIX_MJS = r"""
import postcss from 'postcss';
import prefixer from 'postcss-prefix-selector';
import { readFileSync, writeFileSync } from 'node:fs';

const ROOT = '#stac-browser-mount';
const ALREADY = /(^|[\s>+~])#stac-browser(?![\w-])/;

function transform(prefix, selector, prefixed, file, rule) {
  if (rule && rule.parent && rule.parent.type === 'atrule'
      && /keyframes$/i.test(rule.parent.name)) {
    return selector;
  }
  const s = selector.trim();
  if (ALREADY.test(s)) return s;
  if (/^\[data-bs-theme/.test(s)) return prefix + s;
  if (/^(:root|html|body)(?![\w-])/.test(s))
    return s.replace(/^(:root|html|body)/, prefix);
  if (s === '*' || s === '*::before' || s === '*::after') return prefix + ' ' + s;
  return prefixed;
}

const file = process.argv[2];
const out = postcss([prefixer({ prefix: ROOT, transform })])
  .process(readFileSync(file, 'utf8'), { from: undefined }).css;
writeFileSync(file, out);
"""


def _run(cmd: list[str], cwd: Path | None = None, env: dict | None = None,
         capture: bool = False) -> str:
    print(f"[stac-browser] $ {' '.join(cmd)}")
    if capture:
        out = subprocess.run(cmd, cwd=cwd, env=env, check=True,
                             capture_output=True, text=True).stdout.strip()
        print(f"[stac-browser]   -> {out}")
        return out
    subprocess.run(cmd, cwd=cwd, env=env, check=True)
    return ""


def _git(args: list[str], **kw) -> str:
    return _run(["git", *args], cwd=SRC_DIR, **kw)


def _read_ref() -> str:
    ref = os.environ.get("STAC_BROWSER_REF", "").strip()
    if not ref and PIN_FILE.exists():
        ref = PIN_FILE.read_text().strip()
    if not ref:
        print("[stac-browser] ::warning:: no ref/pin found; using origin/main")
        ref = "origin/main"
    return ref


def clone_and_patch() -> None:
    """
    Shallow-fetch the pinned commit's tree and apply the patches.
    """
    if not shutil.which("git"):
        sys.exit("[stac-browser] git not found on PATH.")

    ref = _read_ref()
    print(f"[stac-browser] fetching ref (shallow): {ref}")

    if not (SRC_DIR / ".git").exists():
        if SRC_DIR.exists():
            shutil.rmtree(SRC_DIR)
        SRC_DIR.mkdir(parents=True, exist_ok=True)
        _git(["init", "-q"])
        _git(["remote", "add", "origin", STAC_REPO])

    target = ref.split("/", 1)[1] if ref.startswith("origin/") else ref
    is_sha = re.fullmatch(r"[0-9a-fA-F]{7,40}", target) is not None
    try:
        _git(["fetch", "--depth", "1", "--filter=blob:none", "origin", target])
        _git(["checkout", "-q", "--detach", "FETCH_HEAD"])
    except subprocess.CalledProcessError:
        if is_sha:
            raise
        _git(["fetch", "--depth", "1", "--filter=blob:none", "origin"])
        _git(["checkout", "-q", "--detach", "FETCH_HEAD"])

    _git(["reset", "--hard", "-q", "HEAD"])
    _git(["clean", "-fdq"])
    _git(["rev-parse", "HEAD"], capture=True)

    patches = sorted(PATCH_DIR.glob("*.patch"))
    print(f"[stac-browser] applying {len(patches)} patch(es)")
    for patch in patches:
        print(f"[stac-browser] applying {patch.name}")
        data = patch.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        result = subprocess.run(
            ["git", "apply", "--ignore-whitespace", "--whitespace=nowarn"],
            cwd=SRC_DIR, input=data, capture_output=True,
        )
        if result.returncode != 0:
            sys.exit(
                f"[stac-browser] patch failed: {patch.name}\n"
                f"  {result.stderr.decode(errors='replace').strip()}\n"
                f"  Patches are generated against the pinned commit; a failure "
                f"usually means a stale or edited patch in {PATCH_DIR.name}/."
            )


def run_build() -> None:
    npm = shutil.which("npm")
    if not npm:
        sys.exit("[stac-browser] npm not found; the docs build needs Node.js.")

    env = os.environ.copy()
    # Deployment invariants: force these so they win over the config file.
    env.update({"SB_historyMode": "hash", "SB_pathPrefix": PATH_PREFIX})
    if CONFIG_FILE.exists():
        ext = SRC_DIR / "waterpark-external-config.js"
        shutil.copy2(CONFIG_FILE, ext)
        env["SB_CONFIG"] = str(ext)
    else:
        print(f"[stac-browser] ::warning:: {CONFIG_FILE.name} not found; using STAC defaults.")
    if CATALOG_URL_ENV:
        env["SB_catalogUrl"] = CATALOG_URL_ENV

    has_lock = (SRC_DIR / "package-lock.json").exists()
    in_ci = bool(os.environ.get("CI"))
    if in_ci and not has_lock:
        sys.exit("[stac-browser] package-lock.json missing; refusing npm install in CI.")
    if has_lock:
        try:
            _run([npm, "ci"], cwd=SRC_DIR, env=env)
        except subprocess.CalledProcessError:
            if in_ci or os.environ.get("STAC_BROWSER_ALLOW_NPM_INSTALL") != "1":
                raise
            _run([npm, "install"], cwd=SRC_DIR, env=env)
    else:
        _run([npm, "install"], cwd=SRC_DIR, env=env)

    _run([npm, "run", "build"], cwd=SRC_DIR, env=env)
    prefix_css(npm, env)


def prefix_css(npm: str, env: dict) -> None:
    """Scope STAC's compiled CSS under #stac-browser-mount (see PREFIX_MJS)."""
    css_files = sorted((SRC_DIR / "dist" / "assets").glob("*.css"))
    if not css_files:
        print("[stac-browser] ::warning:: no CSS files to prefix.")
        return
    # Pinned tooling in an ISOLATED dir so we don't touch STAC's lockfiled
    # install or float versions per build.
    tools = REPO_ROOT / ".build" / "stac-css-tools"
    tools.mkdir(parents=True, exist_ok=True)
    _run([npm, "install", "--prefix", str(tools), "--no-save", "--no-package-lock",
          "postcss@8.4.49", "postcss-prefix-selector@2.1.1"], env=env)
    script = tools / "prefix-stac-css.mjs"
    script.write_text(PREFIX_MJS)
    run_env = {**env, "NODE_PATH": str(tools / "node_modules")}
    for css in css_files:
        _run(["node", str(script), str(css)], env=run_env)
    validate_prefixed_css(css_files)
    print(f"[stac-browser] prefixed {len(css_files)} CSS file(s)")


def validate_prefixed_css(css_files: list[Path]) -> None:
    """Fail if prefixing left a bleeding or double-scoped selector."""
    leaking = re.compile(r"(^|[},])\s*(:root|html|body)\s*[,{]")
    for css in css_files:
        text = css.read_text(errors="ignore")
        if leaking.search(text):
            raise RuntimeError(f"[stac-browser] global :root/html/body left in {css.name}.")
        if "#stac-browser #stac-browser" in text:
            raise RuntimeError(f"[stac-browser] double-scoped selector in {css.name}.")


def copy_runtime(dist: Path) -> None:
    """Copy dist/* (except index.html and .vite/) into .build/data/stac-browser/."""
    target = BUILD_DATA / "stac-browser"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    for item in dist.iterdir():
        if item.name in {"index.html", ".vite"}:
            continue
        dest = target / item.name
        shutil.copytree(item, dest) if item.is_dir() else shutil.copy2(item, dest)
    print(f"[stac-browser] copied runtime -> {target}")


def resolve_entry(dist: Path) -> tuple[str, list[str]]:
    """Return (entry_js, [entry_css...]) from the Vite manifest."""
    manifest = json.loads((dist / ".vite" / "manifest.json").read_text())
    entry = next((v for v in manifest.values() if v.get("isEntry")), None)
    if not entry:
        raise RuntimeError("No isEntry chunk in Vite manifest.")
    return entry["file"], list(entry.get("css", []))


def render_page(entry_js: str, entry_css: list[str]) -> str:
    """Emit the mkdocs page: just a mount div; the host loader does the rest."""
    entry_url = html.escape(f"{ASSET_URL}/{entry_js}", quote=True)
    css_attr = html.escape(json.dumps([f"{ASSET_URL}/{h}" for h in entry_css]), quote=True)
    return f"""---
title: STAC Browser
hide:
  - navigation
  - toc
---

<!-- AUTO-GENERATED by docs/scripts/build_stac_browser_docs.py. Do not edit;
     settings live in docs/assets/stac-browser.config.js. -->

<div class="waterpark-stac-wrapper">
  <div id="stac-browser-mount"
       data-stac-entry="{entry_url}"
       data-stac-css="{css_attr}"></div>
</div>
"""


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "tech"
    if target != "data":
        print(f"[stac-browser] target={target!r}: nothing to do.")
        return

    if not BUILD_DATA.exists():
        sys.exit(
            f"[stac-browser] .build/data missing ({BUILD_DATA}). Run via tox after "
            f"docs-prep, or set GRID_DOCTOR_ROOT. Resolved repo root: {REPO_ROOT}."
        )

    clone_and_patch()
    run_build()

    dist = SRC_DIR / "dist"
    if not dist.exists():
        sys.exit("[stac-browser] dist/ not produced by the build.")

    copy_runtime(dist)
    entry_js, entry_css = resolve_entry(dist)
    (BUILD_DATA / "stac-browser.md").write_text(render_page(entry_js, entry_css))
    print(f"[stac-browser] embedded OK (entry={entry_js}, css={entry_css})")


if __name__ == "__main__":
    main()
