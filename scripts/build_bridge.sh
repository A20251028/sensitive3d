#!/usr/bin/env bash
# Build the native OSGB bridge (requires OpenSceneGraph development files).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
cmake -S "$here/native/osgb_bridge" -B "$here/native/osgb_bridge/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$here/native/osgb_bridge/build" -j
echo "built: $here/native/osgb_bridge/build/osgb_bridge"
