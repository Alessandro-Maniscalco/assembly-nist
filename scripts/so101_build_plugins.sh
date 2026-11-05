#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TASK_PYTHON=${PYTHON:-"$PWD/.venv/bin/python"}
PYTHON="$TASK_PYTHON" bash plugin/build.sh
TASK_MJ=$("$TASK_PYTHON" -c 'import mujoco, pathlib; print(pathlib.Path(mujoco.__file__).parent)')
TASK_LIB=$(find "$TASK_MJ" -maxdepth 1 -name 'libmujoco.so*' -print -quit)
g++ -O3 -std=c++17 -shared -fPIC -I"$TASK_MJ/include" plugin/so101_driver.cc \
    -o plugin/build/libso101_driver.so "$TASK_LIB" -Wl,-rpath,"$TASK_MJ"
