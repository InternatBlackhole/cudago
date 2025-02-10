package main

import (
	"cudaTest/ints"
	"fmt"
	"math"
	"time"

	"github.com/InternatBlackhole/cudago/cuda"
)

func increase(d_array *cuda.DeviceMemory, arrLen uint32, elemSize uint64, numThreads uint32, by int) {
	var err error
	//size := uint64(len(array))
	//elemSize := uint64(unsafe.Sizeof(array[0]))
	//numThreads := uint64(64)

	//Create start event
	start, err := cuda.NewEvent()
	panicErr(err)
	defer start.Destroy()

	//Create end event
	end, err := cuda.NewEvent()
	panicErr(err)
	defer end.Destroy()

	//register memory
	//arr, err := cuda.RegisterAllocationHost(array, elemSize, cuda.CU_MEMHOSTREGISTER_DEVICEMAP)
	//panicErr(err)
	//defer arr.Free()

	// allocate device memory
	//d_array, err := cuda.DeviceMemAlloc(uint64(size * elemSize))
	//panicErr(err)
	//defer d_array.Free()

	// copy data to device
	//err = d_array.MemcpyToDevice(unsafe.Pointer(arr.Ptr), arr.ActualSize)
	//panicErr(err)

	calc := uint32(math.Ceil(float64(arrLen) / float64(numThreads)))

	grid, block := cuda.Dim3{X: calc, Y: 1, Z: 1}, cuda.Dim3{X: uint32(numThreads), Y: 1, Z: 1}

	err = start.Record(nil)
	if err != nil {
		panic(err)
	}

	startT := time.Now()
	err = ints.AddToAll(grid, block, d_array.Ptr, int32(by), int32(arrLen))
	took := time.Since(startT)
	panicErr(err)

	err = end.Record(nil)
	panicErr(err)

	err = end.Synchronize()
	panicErr(err)

	//err = d_array.MemcpyFromDevice(unsafe.Pointer(arr.Ptr), arr.ActualSize)
	//panicErr(err)

	elapsedTime, err := cuda.EventElapsedTime(start, end)
	panicErr(err)

	fmt.Printf(reportFormat, "AddToAll_GoStartToEndKernelCall", "", float64(took.Nanoseconds())/1000)
	fmt.Printf(reportFormat, "AddToAll_KernelCall", "", elapsedTime)
}
