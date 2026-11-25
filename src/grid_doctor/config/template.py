"""Write a commented grid-doctor config file, or set entries in one.

grid-doctor has no command-line tool, so this module is run with Python:

```console
python -m grid_doctor.config.template                  # your config file
python -m grid_doctor.config.template -o "$CONDA_PREFIX/etc/grid-doctor"
python -m grid_doctor.config.template --stdout
python -m grid_doctor.config.template --set weights_store=/work/ab1234/weights
```

Without ``-o`` the file is the user config file
([`user_config_file()`][grid_doctor.config.user_config_file]).  Every
setting is in the file, commented out at its default, so a new file changes
nothing until a line is uncommented or set with ``--set``.  With ``--set``,
an existing file is updated in place: only the named settings change.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from . import SETTINGS, _check, _clean, _read_file, config_files, env_var, user_config_file

FILE_NAME = "config.toml"
"""Name of the file written into a directory given to ``-o``."""

_HELP: dict[str, tuple[str, str]] = {
    "ellipsoid": (
        'Ellipsoid on which new HEALPix grids are built: "sphere" or "WGS84".\n'
        "Recorded in the output; existing data keeps its recorded ellipsoid.",
        '"sphere"',
    ),
    "weights_store": (
        "Where weight files are looked up and stored: a local path or a\n"
        "file:// or memory:// URL.  Unset: grid-doctor/weights in your cache\n"
        "directory (~/.cache/grid-doctor/weights on Linux).",
        '"/path/to/weights"',
    ),
    "esmf_regrid_weightgen": (
        "Name or path of the ESMF_RegridWeightGen executable for offline\n"
        "weight generation.  Unset: ESMF_RegridWeightGen from PATH.",
        '"/path/to/bin/ESMF_RegridWeightGen"',
    ),
    "scratch_dir": (
        "Directory for the temporary files of offline weight generation,\n"
        "which get large for fine grids.  Unset: the system's temporary\n"
        "directory ($TMPDIR, usually /tmp).",
        '"/path/to/scratch"',
    ),
}
"""Description and example value of each setting, for the template."""


def render(values: dict[str, str] | None = None) -> str:
    """Return the text of a config file.

    Settings in *values* are written as active lines with those values; all
    others are commented out at an example value.
    """
    values = values or {}
    environment_file, *user_file = config_files()
    user = user_file[0] if user_file else user_config_file()
    lines = [
        "# grid-doctor configuration (TOML).",
        "#",
        "# grid-doctor reads, later ones winning:",
        f"#   {environment_file}  (everyone using this environment)",
        f"#   {user}  (you; $GRID_DOCTOR_CONFIG names another file)",
        "# Environment variables GRID_DOCTOR_<SETTING> override both files.",
        "#",
        "# Every setting is optional: uncomment a line to change it.  An empty",
        '# string ("") unsets a setting given in a file read before.',
    ]
    for name in SETTINGS:
        description, example = _HELP[name]
        lines.append("")
        lines.extend(f"# {line}" if line else "#" for line in description.splitlines())
        lines.append(f"# Environment variable: {env_var(name)}")
        lines.append(_line(name, values[name]) if name in values else f"# {name} = {example}")
    return "\n".join(lines) + "\n"


def _line(name: str, value: str) -> str:
    """Return the TOML line setting *name* to the string *value*."""
    return f"{name} = {json.dumps(value)}"  # a JSON string is a valid TOML basic string


def update(text: str, values: dict[str, str]) -> str:
    """Return config file *text* with the settings in *values* set.

    The first line setting each one, active or commented out, is replaced;
    settings without such a line are appended.  Everything else is kept.
    """
    lines = text.splitlines()
    for name, value in values.items():
        pattern = re.compile(rf"^\s*#?\s*{re.escape(name)}\s*=")
        for index, line in enumerate(lines):
            if pattern.match(line):
                lines[index] = _line(name, value)
                break
        else:
            lines.append(_line(name, value))
    return "\n".join(lines) + "\n"


def target_path(output: Path | None) -> Path:
    """Return the file to write for ``-o OUTPUT``, the user config file if ``None``.

    A path ending in ``.toml`` is the file itself; anything else is a
    directory, which gets a ``config.toml``.
    """
    if output is None:
        return user_config_file()
    output = output.expanduser()
    return output if output.suffix == ".toml" else output / FILE_NAME


def _parse_set(items: Sequence[str]) -> dict[str, str]:
    """Return ``--set KEY=VALUE`` items as a dict, each value checked.

    Raises
    ------
    ValueError
        If an item has no ``=``, names an unknown setting or holds an
        invalid value.
    """
    values: dict[str, str] = {}
    for item in items:
        name, sep, value = item.partition("=")
        name = name.strip()
        if not sep:
            raise ValueError(f"--set takes KEY=VALUE, got {item!r}")
        if name not in SETTINGS:
            raise ValueError(f"Unknown setting {name!r}; settings are {SETTINGS}")
        _check(name, _clean(name, value))
        values[name] = value.strip()
    return values


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command; return the exit status."""
    parser = argparse.ArgumentParser(
        prog="python -m grid_doctor.config.template",
        description=(
            "Write a commented grid-doctor config file, every setting at its default, or set entries in one with --set."
        ),
    )
    where = parser.add_mutually_exclusive_group()
    where.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="PATH",
        help=(
            f"directory to write {FILE_NAME} into (created if needed), or a file ending in .toml; "
            f"default: {user_config_file()}"
        ),
    )
    where.add_argument("--stdout", action="store_true", help="print the file instead of writing it")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="set a setting (repeatable); an existing file is updated, keeping everything else",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="replace an existing file with a new template",
    )
    args = parser.parse_args(argv)

    try:
        values = _parse_set(args.set)
    except ValueError as error:
        parser.error(str(error))

    if args.stdout:
        sys.stdout.write(render(values))
        return 0
    path = target_path(args.output)
    if path.exists() and not args.force:
        if not values:
            print(f"{path} exists; use --set KEY=VALUE to change it, or --force to replace it.", file=sys.stderr)
            return 1
        text = update(path.read_text(encoding="utf-8"), values)
        action = "Updated"
    else:
        text = render(values)
        action = "Wrote"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    try:
        _read_file(path)
    except ValueError as error:  # an edited file that was broken before stays broken
        print(f"{action} {path}, but it is not valid: {error}", file=sys.stderr)
        return 1
    print(f"{action} {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
