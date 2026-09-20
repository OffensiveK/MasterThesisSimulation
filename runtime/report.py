
import json
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

# Hacky but works as long as experiment script location follows convention
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo_relative(script):
    path = Path(script).resolve()
    try:
        return path.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return path.name

def _format_entry(key, value, indent):
    pad = "  " * indent
    if is_dataclass(value) and not isinstance(value, type):
        lines = [f"{pad}{key}: {type(value).__name__}"]
        for f in fields(value):
            lines.append(_format_entry(f.name, getattr(value, f.name), indent + 1))
        return "\n".join(lines)
    return f"{pad}{key}: {value}"


def format_params(params):
    return "\n".join(_format_entry(key, value, 0) for key, value in params.items())


def columnar(rows):
    if not rows:
        return {}
    return {key: [row[key] for row in rows] for key in rows[0]}


def rows_from_columnar(block):
    keys = list(block)
    return [dict(zip(keys, values)) for values in zip(*block.values())]


def grid_key(n):
    return f"n={n}"


def n_from_grid_key(key):
    return int(key.split("=")[1])


def write_report(path, title, params, doc=None, script=None, data=None, generated_at=None):
    generated_at = generated_at or datetime.now(timezone.utc)
    if script is not None:
        params = {"script": _repo_relative(script), **params}
    sections = [f"# {title}", ""]
    if doc and doc.strip():
        sections += [doc.strip(), ""]
    sections += [
        f"Generated: {generated_at.isoformat(timespec='seconds')}",
        "",
        "## Parameters",
        "",
        format_params(params),
        "",
    ]
    if data:
        data_path = path.with_suffix(".json")
        data_path.write_text(json.dumps(data, indent=2), encoding="utf-8", newline="")
        sections += ["## Data", "", f"Raw output: [{data_path.name}]({data_path.name})", ""]
    path.write_text("\n".join(sections), encoding="utf-8", newline="")
