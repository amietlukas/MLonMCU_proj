#!/usr/bin/env python3
"""Re-apply project fixes that a CubeMX regenerate keeps dropping.

The STM32N6 multi-context project links the shared top-level Drivers/ and
Middlewares/ source into the FSBL Eclipse project via per-file <link> resources
in FSBL/.project. CubeMX's regen does NOT reliably regenerate those links, so
after every regen the build fails at link time with a flood of
"undefined reference ... Unknown destination type (ARM/Thumb)" errors for HAL /
LL_ATON symbols. This script makes FSBL/.project link every shared .c the build
needs again. It is idempotent — safe to run any time.

Run after each CubeMX regenerate:
    python scripts/fix_cubemx_project.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]            # firmware/BallDetector_N6
DOTPROJECT = PROJ / "FSBL" / ".project"
DOTCPROJECT = PROJ / "FSBL" / ".cproject"

# Source folders that must be registered in the .cproject (CDT only compiles
# files under a sourcePath entry). CubeMX's N6 regen drops "Middlewares".
SOURCE_FOLDERS = ["Core", "Drivers", "Middlewares", "X-CUBE-AI"]

# Shared source dirs (relative to PROJ) whose every .c must be compiled into
# the FSBL build. These live one level above the FSBL project, hence the
# PARENT-1-PROJECT_LOC locationURI Eclipse uses for them.
SHARED_DIRS = [
    "Drivers/STM32N6xx_HAL_Driver/Src",
    "Drivers/BSP/STM32N6xx_Nucleo",
    "Drivers/BSP/Components/mx25um51245g",   # OctoFlash driver (memory-mapped weights)
    "Middlewares/ST/AI/Npu/ll_aton",
    "Middlewares/ST/AI/Npu/Devices/STM32N6XX",
]


def fix_cproject() -> int:
    """Ensure every SOURCE_FOLDERS entry is a sourcePath in each <sourceEntries>
    block of the .cproject. Returns count of entries added. Preserves the
    file's CRLF line endings + tab indentation."""
    with open(DOTCPROJECT, "r", newline="") as f:    # keep \r\n as-is
        raw = f.read()
    eol = "\r\n" if "\r\n" in raw else "\n"
    added = 0
    out_blocks = []
    # Split on the closing tag so we can patch each <sourceEntries> block.
    parts = raw.split("</sourceEntries>")
    for i, part in enumerate(parts[:-1]):
        # indentation of the entries = indentation of <sourceEntries> + one tab
        m = re.search(r"([ \t]*)<sourceEntries>", part)
        entry_indent = (m.group(1) + "\t") if m else "\t\t\t\t\t\t"
        inserts = ""
        for name in SOURCE_FOLDERS:
            if f'kind="sourcePath" name="{name}"' not in part:
                inserts += (f'{entry_indent}<entry flags="VALUE_WORKSPACE_PATH|RESOLVED" '
                            f'kind="sourcePath" name="{name}"/>{eol}')
                added += 1
        # close tag indentation
        cm = re.search(r"([ \t]*)$", part)
        close_indent = cm.group(1) if cm else "\t\t\t\t\t"
        if inserts:
            # part ends right before </sourceEntries>; its tail is the close indent
            part = part[: -len(close_indent)] + inserts + close_indent if close_indent else part + inserts
        out_blocks.append(part)
    out_blocks.append(parts[-1])
    new = "</sourceEntries>".join(out_blocks)
    if added:
        with open(DOTCPROJECT, "w", newline="") as f:
            f.write(new)
    return added


def fix_project_links() -> int:
    """Ensure every shared .c is a <link> in FSBL/.project. Returns count added."""
    text = DOTPROJECT.read_text()
    if "</linkedResources>" not in text:
        print(f"no <linkedResources> in {DOTPROJECT}", file=sys.stderr)
        return -1

    wanted: list[str] = []
    for d in SHARED_DIRS:
        srcdir = PROJ / d
        if not srcdir.is_dir():
            continue
        for c in sorted(srcdir.glob("*.c")):
            wanted.append(f"{d}/{c.name}")

    missing = [rel for rel in wanted if f"<name>{rel}</name>" not in text]
    if not missing:
        return 0

    # Derive the indentation of </linkedResources> so inserted blocks match the
    # file's style (CubeMX regen flips between tabs/spaces).
    m = re.search(r"\n([ \t]*)</linkedResources>", text)
    close_indent = m.group(1) if m else "\t"
    link_i = close_indent + "\t"
    field_i = close_indent + "\t\t"
    block = "".join(
        f"{link_i}<link>\n"
        f"{field_i}<name>{rel}</name>\n"
        f"{field_i}<type>1</type>\n"
        f"{field_i}<locationURI>PARENT-1-PROJECT_LOC/{rel}</locationURI>\n"
        f"{link_i}</link>\n"
        for rel in missing
    )
    text = text.replace(f"{close_indent}</linkedResources>",
                        block + f"{close_indent}</linkedResources>", 1)
    DOTPROJECT.write_text(text)
    for rel in missing:
        print(f"  + link {rel}")
    return len(missing)


def main() -> int:
    n_links = fix_project_links()
    if n_links < 0:
        return 1
    n_src = fix_cproject()
    if n_src:
        print(f"  + {n_src} sourcePath entr(ies) (Middlewares) into .cproject")

    if n_links or n_src:
        print(f"fixed: {n_links} link(s) + {n_src} source-folder entr(ies). "
              f"Do a clean rebuild (scripts/build.sh).")
    else:
        print("ok — .project links and .cproject source folders already correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
