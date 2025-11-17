#!/bin/bash
export FINN_XILINX_PATH=/opt/vivado
export FINN_XILINX_VERSION=2024.2
this_dir="$(dirname "$0")"
export FINN_ROOT="$(realpath "$this_dir/../..")"
echo $FINN_ROOT
export FINN_BUILD_DIR="$FINN_ROOT/tutorials/fpga_flow/build"
export PATH="$FINN_XILINX_PATH/Vitis_HLS/$FINN_XILINX_VERSION/bin:$PATH"
export PATH="$FINN_XILINX_PATH/Vivado/$FINN_XILINX_VERSION/bin:$PATH"
export HLS_PATH="$FINN_XILINX_PATH/Vitis/$FINN_XILINX_VERSION"
export VIVADO_PATH="$FINN_XILINX_PATH/Vivado/$FINN_XILINX_VERSION"
uv run build.py
