#!/bin/bash

function log {
    echo "$@" 1>&2
}

howMany=${1?:'Missing "howMany" arg'}

cat <<EOF
package main

import (
    "github.com/InternatBlackhole/cudago/cuda"
)

func tests(grid, block cuda.Dim3) []generatorParam {
    return []generatorParam{
EOF

for ((i=1; i<=$howMany; i++)); do
    cat <<EOF
        {
            name: "Arg$i",
            times: 10,
            fun: func() {
                end, err := runArg$i(grid, block)
                panicErr(err)
                report("Arg$i", end)
            },
        },
EOF
done

echo -en "    }\n}\n"