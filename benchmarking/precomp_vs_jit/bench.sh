#!/bin/bash -e

BENCH_NUM=${1?Error: no bench number given as first argument}
echo Running "$BENCH_NUM" benchmarks

SLEEP=${SLEEP:-0}
if [ $SLEEP -gt 0 ]; then
    echo "Will sleep between benchmarks for $SLEEP seconds"
fi

BENCH_DIR=${BENCH_DIR?Error: no benchmark result directory given}
echo Benchmark results will be stored in "\"$BENCH_DIR\""

BENCHING_PROGRAM_DIR=${2?Error: No benchching program directory given as second argument}
echo Using benching program from "\"$BENCHING_PROGRAM_DIR\""
(
    cd $BENCHING_PROGRAM_DIR
    echo Running make in "\"$BENCHING_PROGRAM_DIR\""
    make -j$(nproc) || exit $?
) || exit $?

#echo $BENCH_DIR/{*.log,*.csv}

mkdir -p $BENCH_DIR
rm -rf $BENCH_DIR/{*.log,*.csv}

for i in $(seq 1 $BENCH_NUM); do
    echo "Running bench $i"
    $BENCHING_PROGRAM_DIR/test 2>"$BENCH_DIR/bench_$i.log" >"$BENCH_DIR/measurement_$i.csv"
    if [ $SLEEP -gt 0 ]; then
        echo "Sleeping for $SLEEP seconds..."
        sleep $SLEEP
    fi
done

echo "Waiting for all benchmarks to finish..." #if using jobs
wait
