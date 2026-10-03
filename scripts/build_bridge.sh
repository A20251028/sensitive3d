#!/usr/bin/env bash
# Build the native OSGB bridge (requires OpenSceneGraph development files).
#
#   ./scripts/build_bridge.sh                          # -> native/osgb_bridge/build/osgb_bridge
#   S3D_BRIDGE_BUILD_DIR=/tmp/b ./scripts/build_bridge.sh
#   CMAKE_PREFIX_PATH=/opt/osg ./scripts/build_bridge.sh   # custom OpenSceneGraph prefix
#
# When Homebrew is present (macOS) its prefix is passed to CMake automatically.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
src="$here/native/osgb_bridge"
build="${S3D_BRIDGE_BUILD_DIR:-$src/build}"

install_hint() {
  if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "  macOS:          brew install open-scene-graph cmake" >&2
  else
    echo "  Debian/Ubuntu:  sudo apt-get install -y cmake g++ libopenscenegraph-dev openscenegraph" >&2
    echo "  Fedora:         sudo dnf install -y cmake gcc-c++ OpenSceneGraph-devel" >&2
  fi
}

if ! command -v cmake >/dev/null 2>&1; then
  echo "error: cmake not found. Install it first:" >&2
  install_hint
  exit 1
fi

cmake_args=(-DCMAKE_BUILD_TYPE=Release)
prefix="${CMAKE_PREFIX_PATH:-}"
if command -v brew >/dev/null 2>&1; then
  brew_prefix="$(brew --prefix)"
  echo "Homebrew detected: adding $brew_prefix to CMAKE_PREFIX_PATH"
  prefix="${prefix:+$prefix;}$brew_prefix"
fi
if [[ -n "$prefix" ]]; then
  cmake_args+=("-DCMAKE_PREFIX_PATH=$prefix")
fi

if ! cmake -S "$src" -B "$build" "${cmake_args[@]}"; then
  echo "" >&2
  echo "error: CMake configuration failed - is OpenSceneGraph (>= 3.4, development files) installed?" >&2
  install_hint
  echo "For a custom install prefix run: CMAKE_PREFIX_PATH=<prefix> $0" >&2
  exit 1
fi
cmake --build "$build" -j
exe="$build/osgb_bridge"
echo "built: $exe"
if [[ -x "$exe" ]]; then
  if ! "$exe" version; then
    echo "warning: '$exe version' failed; run 'sensitive3d doctor' for details" >&2
  fi
fi
