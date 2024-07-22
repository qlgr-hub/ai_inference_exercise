#!/usr/bin/env bash

set -e

ORG_PATH=$(pwd)
SCRIPT_BASE_DIR=$(cd "$(dirname $0)"; pwd)
cd "${SCRIPT_BASE_DIR}"

if [ ! -d build ]; then
    mkdir build
fi

cd build

BuildType="Debug"
if [ -n "$1" ]; then
    BuildType="$1"
fi

cmake -DCMAKE_BUILD_TYPE="${BuildType}" ..

make -j8

cd "${ORG_PATH}"
