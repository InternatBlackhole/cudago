package main

import (
	sorting "cudaTest/sort"
	"fmt"
	"time"

	"github.com/InternatBlackhole/cudago/cuda"
)

func sort(d_array *cuda.DeviceMemory, arrLen uint32, elemSize uint64, numThreads uint32) {
	var err error
	// blocks = pairs of elements to be compared
	numBlocks := (arrLen/2-1)/numThreads + 1

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

	// sort
	gridSize, blockSize := cuda.Dim3{X: uint32(numBlocks), Y: 1, Z: 1}, cuda.Dim3{X: uint32(numThreads), Y: 1, Z: 1}
	bytesLocalMemory := uint64(2 * blockSize.X * uint32(elemSize))

	err = start.Record(nil)
	panicErr(err)

	startT := time.Now()
	sorting.BitonicSortStartEx(gridSize, blockSize, bytesLocalMemory, nil, d_array.Ptr, int32(arrLen)) // k = 2 ... 2 * blockSize.x
	took := time.Since(startT)
	fmt.Printf(reportFormat, "Sort_GoStartKernelCall", "", float64(took.Nanoseconds())/1000)
	for k := 4 * int32(blockSize.X); k <= int32(arrLen); k <<= 1 { // k = 4 * blockSize ... tableLength
		for j := k / 2; j >= 2*int32(blockSize.X); j >>= 1 { //   j = k/2 ... 2 * blockSize.x
			startT = time.Now()
			err = sorting.BitonicSortMiddleEx(gridSize, blockSize, bytesLocalMemory, nil, d_array.Ptr, int32(arrLen), k, j)
			took = time.Since(startT)
			fmt.Printf(reportFormat, "Sort_GoMiddleKernelCall", "", float64(took.Nanoseconds())/1000)
		}
		startT = time.Now()
		sorting.BitonicSortFinishEx(gridSize, blockSize, bytesLocalMemory, nil, d_array.Ptr, int32(arrLen), k) //   j = 2 * blockSize.x ... 1
		took = time.Since(startT)
		fmt.Printf(reportFormat, "Sort_GoFinishKernelCall", "", float64(took.Nanoseconds())/1000)
	}

	err = end.Record(nil)
	panicErr(err)

	err = end.Synchronize()
	panicErr(err)

	//err = d_array.MemcpyFromDevice(unsafe.Pointer(arr.Ptr), arr.ActualSize)
	//panicErr(err)

	elapsedTime, err := cuda.EventElapsedTime(start, end)
	panicErr(err)

	//fmt.Printf(reportFormat, "Sort_GoStartToEndKernelCall", "", float64(took.Microseconds())/1000)
	fmt.Printf(reportFormat, "Sort_KernelCall", "", elapsedTime)

}
