#!/usr/bin/env bash
# Public colonoscopy data for the polyp study. Sizes and licences are logged; nothing here is
# redistributed. SUN-SEG is not downloaded: it needs a personal request to the SUN database.
#   LDPolypVideo   boxes, 160 videos      Ma et al., MICCAI 2021       terms: see repo (research use)
#   PolypGen       masks, video sequences Ali et al., Sci Data 2023    open access (Sci Data)
#   Kvasir-Instrument  instrument masks   Jha et al., MMM 2021        CC BY 4.0 (Simula)
#   Kvasir-SEG     polyp masks, stills    Jha et al., MMM 2020        CC BY 4.0 (Simula)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
D=data/polyp
mkdir -p "$D"
LOG=$D/download.log
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

fetch() {  # fetch <name> <url> <zipname>
  local name=$1 url=$2 zip=$D/$3
  if [ -d "$D/$name" ]; then log "$name exists, skip"; return; fi
  log "$name: downloading $url"
  curl -fL --retry 3 -o "$zip" "$url" >>"$LOG" 2>&1 || { log "ERROR $name download"; return; }
  mkdir -p "$D/$name" && unzip -q "$zip" -d "$D/$name" && rm -f "$zip" && log "$name: ok ($(du -sh "$D/$name" | cut -f1))"
}
fetch kvasir-instrument https://datasets.simula.no/downloads/kvasir-instrument.zip kvasir-instrument.zip
fetch kvasir-seg        https://datasets.simula.no/downloads/kvasir-seg.zip        kvasir-seg.zip

gd() {  # gd <name> <kind: folder|file> <id-or-url>
  local name=$1 kind=$2 ref=$3
  if [ -d "$D/$name" ] && [ -n "$(ls -A "$D/$name" 2>/dev/null)" ]; then log "$name exists, skip"; return; fi
  mkdir -p "$D/$name"
  log "$name: gdown $kind $ref"
  if [ "$kind" = folder ]; then
    .venv/bin/gdown --folder -O "$D/$name" "$ref" >>"$LOG" 2>&1 || log "ERROR $name gdown folder"
  else
    .venv/bin/gdown -O "$D/$name/" "$ref" >>"$LOG" 2>&1 || log "ERROR $name gdown file"
  fi
  log "$name: $(du -sh "$D/$name" | cut -f1), $(find "$D/$name" -type f | wc -l | tr -d ' ') files"
}
gd ldpolypvideo-part1 folder "https://drive.google.com/drive/folders/13KwU_uZcxsl6dL-mqcs39Yb0gjU9vn3G"
gd ldpolypvideo-part2 file   "https://drive.google.com/file/d/1pxFYO-nRd5uqdYsjs7NRkwXQMMurbWsZ/view"
gd polypgen-sequences folder "https://drive.google.com/drive/folders/16uL9n84SrMt7IiQFzTUQNaJ9TbHJ8DhW"
log "DONE-DOWNLOAD free=$(df -h / | tail -1 | awk '{print $4}')"
