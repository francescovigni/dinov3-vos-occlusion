#!/usr/bin/env bash
# Google Drive parts of the polyp data (gdown >= 6). Sequential; sizes logged. Needs 7z for RAR.
set -uo pipefail
cd "$(dirname "$0")/.."
D=data/polyp; LOG=$D/download.log; mkdir -p "$D"
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
get() {  # get <dir> <file id> <filename>
  local dir=$D/$1 id=$2 name=$3
  if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then log "$1 exists, skip"; return; fi
  mkdir -p "$dir"
  log "$1: downloading $name"
  .venv/bin/gdown -q -O "$D/$name" "https://drive.google.com/uc?id=$id" >>"$LOG" 2>&1 || { log "ERROR $1 download"; return; }
  log "$1: got $(du -h "$D/$name" | cut -f1), extracting"
  case "$name" in
    *.rar) 7z x -y -o"$dir" "$D/$name" >>"$LOG" 2>&1 || { log "ERROR $1 extract"; return; } ;;
    *.zip) unzip -q "$D/$name" -d "$dir" >>"$LOG" 2>&1 || { log "ERROR $1 extract"; return; } ;;
  esac
  rm -f "$D/$name"
  log "$1: ok ($(du -sh "$dir" | cut -f1), $(find "$dir" -type f | wc -l | tr -d ' ') files)"
}
get ldpolypvideo/TrainValid 1Y-fS9vMGvrEQZS1rKDnCxSkvrQz1lyXX TrainValid.rar
get ldpolypvideo/Test       1gSJiUhZJnpkELXoCUgNsPtYnUCS7qqyP Test.rar
get polypgen-sequences      1wBgGO9c9aeb211GhaSmOA0lSgFpfBeXN sequence_data_positive_cropped.zip
log "DONE-DRIVE free=$(df -h / | tail -1 | awk '{print $4}')"
