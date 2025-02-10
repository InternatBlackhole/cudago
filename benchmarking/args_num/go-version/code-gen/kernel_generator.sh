#!/bin/bash

# Print all args to stderr
function log {
    echo "$@" 1>&2
}

function gen {
    local argsNum=$1
    echo -en "\n__global__ void arg$argsNum("
    local i="1"
    for ((; i<=$argsNum-1; i++)); do
        echo -en "int a$i, "
    done
    echo -en "int a$i"
    echo -en ") {\n\n}\n"
}

howMany=${1?:'Missing "howMany" arg'}

echo -en "#ifdef __cplusplus\n extern \"C\" {\n#endif\n"

for ((i=1; i<=$howMany; i++)); do
    gen $i
done

echo -en "\n#ifdef __cplusplus\n}\n#endif\n"