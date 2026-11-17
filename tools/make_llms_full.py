#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
Write llms-full.txt: the README and every page under docs/ in one file, for a coding assistant or a crawler that
reads one file and follows no links.

    python3 tools/make_llms_full.py            # writes llms-full.txt at the repository's root
    python3 tools/make_llms_full.py --check    # exits 1, changing nothing, where the file is not what would be written

In one sentence: the head of llms.txt (its title and its summary) opens the file → README.md follows, then
docs/quickstart.md, then the other pages of docs/ by name → in each, a link to a file of this repository becomes that
file's address on GitHub, since a relative link means nothing in one flat file, and the README's centered blocks (its
banner, its badges, its row of links) are left out, since they are layout.

WHY generated and checked: a version written by hand goes stale the first time a page changes. tests/test_llms_full.py
runs `--check`, so a page changed without this file fails the tests.

Standard library only.
"""
import pathlib
import posixpath
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "llms-full.txt"
# a markdown link's target that is neither an address, an anchor on the same page, nor a mail address
LINK = re.compile(r"\]\((?!https?://|#|mailto:)([^)\s]+)\)")
# a centered block of HTML: a banner, a row of badges, a row of links
CENTERED = re.compile(r'<p align="center">.*?</p>\n*', re.S)


def repository() -> str:
    """The repository's address on GitHub, as pyproject.toml states it under [project.urls]."""
    text = (ROOT / "pyproject.toml").read_text()
    found = re.search(r'^Repository\s*=\s*"([^"]+)"', text, re.M)
    if not found:
        raise SystemExit("pyproject.toml has no Repository under [project.urls]: llms-full.txt's links need it")
    return found.group(1).rstrip("/")


def pages() -> "list[pathlib.Path]":
    """README.md, the quickstart, then the other pages of docs/ by name; docs/README.md is only a list of them."""
    docs = sorted(p for p in (ROOT / "docs").glob("*.md") if p.name not in ("README.md", "quickstart.md"))
    first = [ROOT / "README.md", ROOT / "docs" / "quickstart.md"]
    return [p for p in first if p.exists()] + docs


def absolute(text: str, page: pathlib.Path, base: str) -> str:
    """`text` with each link to a file of this repository made that file's address on GitHub."""
    folder = page.parent.relative_to(ROOT).as_posix()

    def address(match: "re.Match[str]") -> str:
        target, _, anchor = match.group(1).partition("#")
        path = posixpath.normpath(posixpath.join(folder, target)) if target else page.relative_to(ROOT).as_posix()
        if path.startswith(".."):
            return match.group(0)                              # points outside the repository: left unchanged
        kind = "tree" if (ROOT / path).is_dir() else "blob"
        return f"]({base}/{kind}/main/{path}{'#' + anchor if anchor else ''})"

    return LINK.sub(address, text)


def build() -> str:
    """The whole file's text."""
    base = repository()
    head = (ROOT / "llms.txt").read_text().split("\n\n")
    parts = ["\n\n".join(head[:2]).rstrip() + "\n",              # the title and the summary of llms.txt
             f"This file is {base}'s README and documentation in one piece, written by tools/make_llms_full.py.\n"
             f"The map of the repository, file by file, is {base}/blob/main/llms.txt.\n"]
    for page in pages():
        name = page.relative_to(ROOT).as_posix()
        body = absolute(CENTERED.sub("", page.read_text()), page, base).strip()
        parts.append(f"---\n\nSource: {base}/blob/main/{name}\n\n{body}\n")
    return "\n".join(parts)


def main(argv: "list[str]") -> int:
    """Write the file, or with --check report whether it is current."""
    text = build()
    if "--check" in argv:
        if not OUT.exists() or OUT.read_text() != text:
            print("llms-full.txt is not what tools/make_llms_full.py writes now: run it, and commit the result",
                  file=sys.stderr)
            return 1
        return 0
    OUT.write_text(text)
    print(f"llms-full.txt: {len(pages())} pages, {len(text):,} characters")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
