package main

import (
	"cudaTest/ints"
	"fmt"
	"math"
	"time"
	"unsafe"

	"github.com/InternatBlackhole/cudago/cuda"
)

// format: <operation>;"<image_name>";<time in ms>
const reportFormat = "%s;\"%d\";%f\n"
const reportHeader = "Operation;ProblemSize;Time\n"

func main() {

	var err error

	fmt.Print(reportHeader)

	// inicializiramo napravo
	dev, err := cuda.Init(0)
	if err != nil {
		panic(err)
	}
	defer dev.Close()

	maxLen := uint64(1 << 28)
	elemSize := uint64(unsafe.Sizeof(int32(0)))

	arr, err := cuda.HostMemAlloc[int32](maxLen, elemSize)
	panicErr(err)
	defer arr.Free()

	dArr, err := cuda.DeviceMemAlloc(maxLen)
	panicErr(err)
	defer dArr.Free()

	//routines := runtime.NumCPU() - 1

	// 1<<14 = 16384 = 2^14, 1<<28 = 268_435_456 = 2^28
	for i := uint64(1 << 14); i <= maxLen; i <<= 1 {
		for j := uint64(0); j < i; j++ {
			arr.Arr[j] = int32(j)
		}

		err = dArr.MemcpyToDevice(unsafe.Pointer(arr.Ptr), i)
		panicErr(err)

		callDur, dur := increase(dArr, uint32(i), elemSize, 128, 11)

		fmt.Printf(reportFormat, "CallDur", i, callDur)
		fmt.Printf(reportFormat, "Dur", i, dur)
	}
}

func increase(d_array *cuda.DeviceMemory, arrLen uint32, elemSize uint64, numThreads uint32, by int) (kernCallDur float64, kernDur float32) {
	var err error
	//Create start event
	start, err := cuda.NewEvent()
	panicErr(err)
	defer start.Destroy()

	//Create end event
	end, err := cuda.NewEvent()
	panicErr(err)
	defer end.Destroy()

	calc := uint32(math.Ceil(float64(arrLen) / float64(numThreads)))

	grid, block := cuda.Dim3{X: calc, Y: 1, Z: 1}, cuda.Dim3{X: uint32(numThreads), Y: 1, Z: 1}

	err = start.Record(nil)
	if err != nil {
		panic(err)
	}

	startT := time.Now()
	err = ints.AddToAll(grid, block, d_array.Ptr, int32(by), int32(arrLen))
	kernCallDur = float64(time.Since(startT).Nanoseconds()) / 1000
	panicErr(err)

	err = end.Record(nil)
	panicErr(err)

	err = end.Synchronize()
	panicErr(err)

	kernDur, err = cuda.EventElapsedTime(start, end)
	panicErr(err)

	return
}

func panicErr(err error) {
	if err != nil {
		panic(err)
	}
}
