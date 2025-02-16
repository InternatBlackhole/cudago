package main

import (
	"cudaTest/ints"
	"math"
	"time"

	"github.com/InternatBlackhole/cudago/cuda"
)

func increase(d_array *cuda.DeviceMemory, arrLen uint32, elemSize uint64, numThreads uint32, by int) (callDurNs int64, kernelDurMs float32) {
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
	callDurNs = time.Since(startT).Nanoseconds()
	panicErr(err)

	err = end.Record(nil)
	panicErr(err)

	err = end.Synchronize()
	panicErr(err)

	kernelDurMs, err = cuda.EventElapsedTime(start, end)
	//elapsedTime, err := cuda.EventElapsedTime(start, end)
	panicErr(err)

	//fmt.Printf(reportFormat, "AddToAll_GoStartToEndKernelCall", "", float64(callDurNs)/1000.0)
	//fmt.Printf(reportFormat, "AddToAll_KernelCall", "", kernelDurMs)
	return
}
