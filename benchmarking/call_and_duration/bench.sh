#!/bin/bash -e

BENCH_NUM=${1?Error: no bench number given as first argument}
echo Running "$BENCH_NUM" benchmarks

IMGS_DIR=${IMGS_DIR:-testImgs}
echo Using images from "\"$IMGS_DIR\""

IMGS=$(find $IMGS_DIR -type f | sort | tr '\n' ' ')

#echo "Images: $IMGS"
#exit 0

IMG_OUT_DIR=${IMG_OUT_DIR?Error: no image output directory given}
echo Image output will be stored in "\"$IMG_OUT_DIR\""

BENCH_DIR=${BENCH_DIR?Error: no benchmark result directory given}
echo Benchmark results will be stored in "\"$BENCH_DIR\""

BENCHING_PROGRAM_DIR=${2?Error: No benchching program directory given as second argument}
echo Using benching program from "\"$BENCHING_PROGRAM_DIR\""
(
    cd $BENCHING_PROGRAM_DIR
    echo Running make in "\"$BENCHING_PROGRAM_DIR\""
    make || exit $?
) || exit $?

#echo $BENCH_DIR/{*.log,*.csv}

mkdir -p $BENCH_DIR
rm -rf $BENCH_DIR/{*.log,*.csv}

for i in $(seq 1 $BENCH_NUM); do
    echo "Running bench $i"
    if [ $IMG_OUT_DIR == "/dev/null" ]; then
        OUT="/dev/null"
    else
        OUT="$IMG_OUT_DIR/bench_out_$i"
        mkdir -p "$OUT"
    fi
    $BENCHING_PROGRAM_DIR/edge_recognition $OUT $IMGS 2>"$BENCH_DIR/bench_$i.log" >"$BENCH_DIR/measurement_$i.csv"
done

echo "Waiting for all benchmarks to finish..."
wait