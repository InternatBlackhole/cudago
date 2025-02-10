#!/bin/bash

function log {
    echo "$@" 1>&2
}

howMany=${1?:'Missing "howMany" arg'}
source=${2?:'Missing "source" file output location'}

(
cat <<EOF
#include <vector>
#include "common.hpp"
#include "callbacks.cuh"

std::vector<generatorParam> get(dim3 grid, dim3 block) {
    return std::vector<generatorParam> {
    
EOF

for ((i=1; i<=$howMany; i++)); do
cat <<EOF
        {
            "Arg$i",
            10,
            [=]() {
                auto res = runArg$i(grid, block);
                report("Arg$i", res);
            }
        },
EOF
done

cat <<EOF
    };
}
EOF
) >$source