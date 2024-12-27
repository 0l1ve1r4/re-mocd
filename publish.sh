#!/bin/bash

set -e

TARGETS=("x86_64-unknown-linux-gnu" "x86_64-pc-windows-gnu")

if ! command -v maturin &> /dev/null
then
    echo "Maturin not found! Please install maturin before running this script."
    exit 1
fi

build_for_target() {
    local TARGET=$1
    echo "Building for $TARGET..."
    
    if [[ "$TARGET" == *"windows"* ]]; then
        if ! command -v x86_64-w64-mingw32-gcc &> /dev/null; then
            echo "Cross-compilation toolchain for Windows not found! Install x86_64-w64-mingw32-gcc."
            exit 1
        fi
    fi

    # Run maturin build
    maturin build --release --target "$TARGET"
}

for TARGET in "${TARGETS[@]}"; do
    build_for_target "$TARGET"
done

echo "Builds completed successfully!"
