#!/usr/bin/env python3
"""Reject private source identifiers, TIFF files, and unapproved raster images."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_RASTER_FILES = {"docs/assets/figure5_external_transfer.png"}
RASTER_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
TIFF_SUFFIXES = {".tif", ".tiff"}
TIFF_SIGNATURES = {b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"}
INLINE_IMAGE_PREFIX = "data:" + "image/"
SENSITIVE_SOURCE_MARKERS = ("b3" + "gt", "b3" + "tp")


def tracked_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / value.decode("utf-8") for value in result.stdout.split(b"\x00") if value]


def notebook_findings(path: Path) -> list[str]:
    findings = []
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook.get("cells", [])):
        for attachment, payload in cell.get("attachments", {}).items():
            image_types = sorted(mime for mime in payload if mime.startswith("image/"))
            if image_types:
                findings.append(
                    f"{path.relative_to(ROOT)} cell {index} embeds attachment {attachment}: {', '.join(image_types)}"
                )
        for output in cell.get("outputs", []):
            image_types = sorted(mime for mime in output.get("data", {}) if mime.startswith("image/"))
            if image_types:
                findings.append(
                    f"{path.relative_to(ROOT)} cell {index} embeds output: {', '.join(image_types)}"
                )
    return findings


def main() -> None:
    findings = []
    for path in tracked_paths():
        relative = path.relative_to(ROOT).as_posix()
        relative_lower = relative.lower()
        suffix = path.suffix.lower()
        for marker in SENSITIVE_SOURCE_MARKERS:
            if marker in relative_lower:
                findings.append(f"Sensitive source identifier in path: {relative}")
        if suffix in TIFF_SUFFIXES:
            findings.append(f"Tracked TIFF file: {relative}")
        elif suffix in RASTER_SUFFIXES and relative not in ALLOWED_RASTER_FILES:
            findings.append(f"Unapproved tracked raster image: {relative}")

        with path.open("rb") as stream:
            if stream.read(4) in TIFF_SIGNATURES:
                findings.append(f"File has a TIFF binary signature: {relative}")

        if suffix == ".ipynb":
            findings.extend(notebook_findings(path))
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        text_lower = text.lower()
        if INLINE_IMAGE_PREFIX in text_lower:
            findings.append(f"Inline data-image payload: {relative}")
        for marker in SENSITIVE_SOURCE_MARKERS:
            if marker in text_lower:
                findings.append(f"Sensitive source identifier in file: {relative}")

    if findings:
        raise SystemExit("Repository image audit failed:\n- " + "\n- ".join(findings))
    print(
        "Repository privacy audit passed: no sensitive source identifiers, TIFF files, "
        "or embedded notebook images found."
    )


if __name__ == "__main__":
    main()
