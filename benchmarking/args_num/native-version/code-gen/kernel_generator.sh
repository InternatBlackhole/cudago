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

    echo -en "int a$i)"

    echo -e "$2"
    #log "Done generating kernel with $1 args"
}

howMany=${1?:'Missing "howMany" arg'}
header=${2?:'Missing "header" file output location'}
source=${3?:'Missing "source" file output location'}

(
cat <<EOF
#ifndef KERNELS_CUH_H_
#define KERNELS_CUH_H_

#ifdef __cplusplus
extern "C" {
#endif

EOF

for ((i=1; i<=$howMany; i++)); do
    gen $i ";" #generates header file
done

cat <<EOF
#ifdef __cplusplus
}
#endif

#endif
EOF
) >$header &

(
cat <<EOF
#include "${header##*/}"

#ifdef __cplusplus
extern "C" {
#endif

EOF

for ((i=1; i<=$howMany; i++)); do
    gen $i " {\n\n}"  #generates source file
done

cat <<EOF
#ifdef __cplusplus
}
#endif
EOF

) >$source &

wait

#echo -en "#ifdef __cplusplus\n extern \"C\" {\n#endif\n" | tee $header $source >/dev/null
#for ((i=1; i<=$howMany; i++)); do
#    gen $i ";" >>kernels.cuh #generates header file
#    gen $i " {\n\n}"  >>kernels.cu #generates source file
#done
#echo -en "\n#ifdef __cplusplus\n}\n#endif\n" | tee -a $header $source >/dev/null