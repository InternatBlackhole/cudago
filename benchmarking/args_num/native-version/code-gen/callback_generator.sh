#!/bin/bash

function log {
    echo "$@" 1>&2
}

function gen_bench_call_func {
    #log "Generating benchmarking call with $1 args"
    local argsNum=$1
    cat <<EOF
int64_t runArg$argsNum(dim3 grid, dim3 block) {
    chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
EOF
    echo -n "    arg$argsNum<<<grid, block>>>(" #leave like this to avoid newline from cat
    local i="1"
    for ((; i<=$argsNum-1; i++)); do
        echo -en "$i, "
    done
    echo -e "$i);"
cat <<EOF
    chrono::high_resolution_clock::time_point end = chrono::high_resolution_clock::now();
    getLastCudaError("Kernel arg5 execution failed");
    return chrono::duration_cast<chrono::nanoseconds>(end - start).count();
}

EOF
    #log "Done generating benchmarking call with $1 args"
}

howMany=${1?:'Missing "howMany" arg'}
header=${2?:'Missing "header" file output location'}
source=${3?:'Missing "source" file output location'}

(
cat <<EOF
#ifndef CALLBACKS_H_
#define CALLBACKS_H_

EOF

for ((i=1; i<=$howMany; i++)); do
    echo "int64_t runArg$i(dim3 grid, dim3 block);"
done

cat <<EOF
void runOnce(dim3 grid, dim3 block);

#endif
EOF
) >$header &

(
cat <<EOF
#include "${header##*/}"
#include <chrono>
#include "helper_cuda.h"
#include "kernels.cuh"

using namespace std;

EOF
for ((i=1; i<=$howMany; i++)); do
    gen_bench_call_func $i
done

cat <<EOF
void runOnce(dim3 grid, dim3 block) {
EOF

for ((i=1; i<=$howMany; i++)); do
    echo "    runArg$i(grid, block);"
done

echo "}"
) >$source &

wait