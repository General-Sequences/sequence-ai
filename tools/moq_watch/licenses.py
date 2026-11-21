"""LICENSES.txt for the console page's MoQ bundle: each package it took code from (meta.json), its version and
licence, and the licence's text with its copyright notice - from the package, or, for moq-dev's packages, which ship
none, from their repository (MIT chosen of "MIT OR Apache-2.0")."""
import json
import pathlib
import re
import sys

UPSTREAM = {"moq": pathlib.Path(sys.argv[1]), "web-transport": pathlib.Path(sys.argv[2])}   # moq-dev's LICENSE-MIT
meta = json.load(open("meta.json"))
took = sorted({m.group(1) for p in meta["inputs"] for m in [re.search(r"node_modules/((?:@[^/]+/)?[^/]+)/", p)] if m})
out = ["moq-watch.js, the page seq teleop serve serves, watching a robot through a MoQ relay, bundles code from these",
       "packages (esbuild 0.25.10, minified). Each is named with its version and licence, "
       "and its licence's text follows.",
       "@moq/hang's audio polyfill (libav.js and libavjs-webcodecs-polyfill) is left out of it.", ""]
for name in took:
    pj = json.load(open(f"node_modules/{name}/package.json"))
    lic = pj.get("license")
    files = sorted(pathlib.Path(f"node_modules/{name}").glob("LICEN[CS]E*"))
    if files:
        text = "\n\n".join(f.read_text().strip() for f in files)
    elif name.startswith("@moq/"):
        repo = "web-transport" if "web-transport" in json.dumps(pj.get("repository")) else "moq"
        text = UPSTREAM[repo].read_text().strip()
        lic = f"{lic}, used under MIT"
    else:
        raise SystemExit(f"{name}: no licence file")
    out += ["=" * 100, f"{name} {pj['version']} ({lic})", "=" * 100, "", text, ""]
pathlib.Path("LICENSES.txt").write_text("\n".join(out))
print(len(took), "packages", sum(1 for _ in open("LICENSES.txt")), "lines")
