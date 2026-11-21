#!/bin/bash
# The teleop console page's MoQ bundle, sequence_ai/_core/teleop_console/vendor/moq-watch.js, and its LICENSES.txt:
# @moq/watch and @moq/net at the versions package.json pins, the ones the page is tested with, from package-lock.json,
# bundled by esbuild into one minified ES module that registers <moq-watch> and exports @moq/net as Net. @moq/hang's Opus polyfill (libav.js, LGPL-2.1, and its WebCodecs wrapper) is left out: a robot's broadcasts carry
# no audio (no-audio-polyfill.js). Needs node and npm; writes nothing outside this folder but the two files.
#
#   bash tools/moq_watch/build.sh
set -eu
cd "$(dirname "$0")"
OUT=../../sequence_ai/_core/teleop_console/vendor
npm ci --no-audit --no-fund --silent
./node_modules/.bin/esbuild entry.js --bundle --format=esm --minify --target=es2022 --legal-comments=eof \
  --alias:@libav.js/variant-opus-af=./no-audio-polyfill.js \
  --alias:@kixelated/libavjs-webcodecs-polyfill=./no-audio-polyfill.js \
  --metafile=meta.json --outfile="$OUT/moq-watch.js"
python3 licenses.py moq-LICENSE-MIT web-transport-LICENSE-MIT
mv LICENSES.txt "$OUT/LICENSES.txt"
rm -f meta.json
ls -la "$OUT"
