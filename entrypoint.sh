#!/bin/bash
set -e

# Needed to point the system towards pytorch CUDA
export LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/torch/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH


# Main command
if [ "$XPASSTHROUGH" = true ]
then
    echo "Passing through local X server."
    $@
elif nvidia-smi > /dev/null 2>&1 ; then
    echo "Using Docker virtual X server (with GPU)."
    export VGL_DISPLAY=$DISPLAY
    xvfb-run -a --server-num=$((99 + $RANDOM % 10000)) \
	     --server-args='-screen 0 640x480x24 +extension GLX +render -noreset' vglrun $@
else
    echo "Using Docker virtual X server (no GPU)."
    xvfb-run -a --server-num=$((99 + $RANDOM % 10000)) \
	     --server-args='-screen 0 640x480x24' $@
fi
