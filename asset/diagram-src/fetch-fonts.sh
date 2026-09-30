#!/bin/sh
# Download the hand-written fonts used by the diagrams (Xiaolai, SIL OFL 1.1).
set -e
cd "$(dirname "$0")"
mkdir -p fonts
base=https://github.com/lxgw/kose-font/releases/download/v3.126
for f in Xiaolai-Regular.ttf XiaolaiMono-Regular.ttf; do
  [ -s "fonts/$f" ] || curl -fL -o "fonts/$f" "$base/$f"
done
ls -l fonts
