#!/bin/bash

function log {
    echo "$@" 1>&2
}

function gen_bench_call_func {
    local argsNum=$1
cat <<EOF
func runArg$argsNum(grid, block cuda.Dim3) (time.Duration, error) {
    start := time.Now()
EOF
    echo -en "    err := cu.Arg$argsNum(grid, block, "
    local i="1"
    for ((; i<=$argsNum-1; i++)); do
        echo -en "int32($i), "
    done
    echo -e "int32($i))"
cat <<EOF
    if err != nil {
        return time.Since(start), err
    }
    return time.Since(start), nil
}

EOF
}

howMany=${1?:'Missing "howMany" arg'}
pkgName=${2?:'Missing "pkgName" arg'}

cat <<EOF
package main

import (
    "time"
    "$pkgName"
    "github.com/InternatBlackhole/cudago/cuda"
)

EOF


for ((i=1; i<=$howMany; i++)); do
    gen_bench_call_func $i
done

echo "func callAllOnce(grid, block cuda.Dim3) {"

for ((i=1; i<=$howMany; i++)); do
    echo "    runArg$i(grid, block)" 
done

echo "}"
