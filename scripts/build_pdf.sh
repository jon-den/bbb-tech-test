#!/usr/bin/env bash
# Build docs/CASE_STUDY.pdf from docs/CASE_STUDY.md.
#
# Pipeline: pandoc (markdown → styled HTML, images embedded, citations rendered
# from docs/references.bib) → Chrome headless (HTML → PDF).
#
# Prerequisites:
#   - pandoc ≥ 3.0            (brew install pandoc)
#   - Google Chrome            (macOS default install path)
#   - docs/references.bib      (BibTeX for --citeproc)
#   - docs/CASE_STUDY.md YAML  (bibliography: references.bib)
#
# Run from anywhere; paths are resolved relative to this script's location.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

INPUT="docs/CASE_STUDY.md"
OUTPUT="docs/CASE_STUDY.pdf"
# mktemp -t X.html produces X.html.<random>; Chrome then treats the file as
# unknown extension and stops parsing <style> tags correctly (renders CSS as
# body text). Suffix must be exactly .html for Chrome to parse it as HTML.
TMP_HTML="$(mktemp -t case_study_XXXXXX).html"
trap 'rm -f "$TMP_HTML"' EXIT

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

if [[ ! -x "$CHROME" ]]; then
  echo "error: Google Chrome not found at $CHROME" >&2
  exit 1
fi
if ! command -v pandoc >/dev/null 2>&1; then
  echo "error: pandoc not on PATH (brew install pandoc)" >&2
  exit 1
fi
if [[ ! -f "$INPUT" ]]; then
  echo "error: $INPUT not found" >&2
  exit 1
fi

# CSS is passed as a single line — pandoc's `header-includes` variable is
# inserted into <head> verbatim, and multi-line content breaks the placement
# (Chrome then renders the <style> block as body text on page 1).
CSS='<style>body{font-family:Georgia,serif;max-width:820px;margin:20px auto;padding:0 18px;font-size:9.5pt;line-height:1.38}h1{font-size:15pt;border-bottom:2px solid #333;padding-bottom:5px;margin-bottom:0.2em}h2{font-size:12pt;margin-top:1.2em;margin-bottom:0.3em;color:#1a1a2e}h3{font-size:10pt;color:#333;margin-top:0.9em;margin-bottom:0.2em}h4{font-size:9.5pt;color:#444;margin-top:0.7em;margin-bottom:0.15em;font-style:italic}p{margin:0.3em 0}table{border-collapse:collapse;width:100%;margin:0.4em 0;font-size:8.5pt}th,td{border:1px solid #ccc;padding:3px 6px;text-align:left}th{background:#f0f0f0}tr:nth-child(even){background:#f9f9f9}code{background:#f4f4f4;padding:1px 2px;border-radius:2px;font-size:8pt}hr{border:none;border-top:1px solid #ccc;margin:1em 0}strong{color:#1a1a2e}blockquote{background:#f8f7f4;border-left:3px solid #ccc;padding:5px 8px;margin:0.4em 0;font-size:8pt;line-height:1.3}img{max-width:100%;height:auto;margin:0.2em 0}ul,ol{margin:0.2em 0;padding-left:1.3em}li{margin:0.08em 0}header{display:none}.csl-entry{margin-bottom:0.4em;font-size:8.5pt}@media print{body{margin:0;padding:10px}}</style>'

echo "[1/2] pandoc: $INPUT → $TMP_HTML"
pandoc "$INPUT" \
  -o "$TMP_HTML" \
  --standalone \
  --embed-resources \
  --citeproc \
  --resource-path=docs:. \
  --metadata title="Camzyos Case Study" \
  --metadata lang="en" \
  -V "header-includes=$CSS"

echo "[2/2] chrome: $TMP_HTML → $OUTPUT"
"$CHROME" \
  --headless \
  --disable-gpu \
  --no-sandbox \
  --print-to-pdf="$OUTPUT" \
  --print-to-pdf-no-header \
  --no-pdf-header-footer \
  "file://$TMP_HTML" 2>/dev/null

echo "done: $OUTPUT ($(du -h "$OUTPUT" | cut -f1))"
