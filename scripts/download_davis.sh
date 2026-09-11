#!/usr/bin/env bash
# DAVIS 2017 trainval, 480p (~800 MB). Licence: see https://davischallenge.org
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
URL="https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
ZIP="data/DAVIS-2017-trainval-480p.zip"
if [ ! -d data/DAVIS/JPEGImages ]; then
  [ -f "$ZIP" ] && [ "$(wc -c < "$ZIP")" -gt 100000000 ] || curl -fL --retry 3 -o "$ZIP" "$URL"
  unzip -q "$ZIP" -d data && rm -f "$ZIP"
fi
echo "DAVIS at data/DAVIS: $(ls data/DAVIS/JPEGImages/480p | wc -l) sequences"
