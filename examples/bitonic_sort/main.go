package main

import (
	"bitonic_sort/cuda_stuff"
	"fmt"
	"math"
	"math/rand"
	"unsafe"

	"github.com/InternatBlackhole/cudago/cuda"
)

func main() {
	size := int32(1 << 12)
	dev, err := cuda.Init(0)
	if err != nil {
		panic(err)
	}
	defer dev.Close()
	multiKernel(int32(math.Min(1024, float64(size))), size)
}

func multiKernel(numThreads, tableLength int32) {
	var err error
	numBlocks := (tableLength/2-1)/numThreads + 1

	intSize := uint64(4) //uint64(unsafe.Sizeof(int32(0)))
	memSize := uint64(tableLength) * intSize

	//Allocate host memory
	ha := make([]int32, tableLength)

	//Optimization for faster data transfer between host and device, not needed to run
	has, err := cuda.RegisterAllocationHost(ha, intSize, cuda.CU_MEMHOSTREGISTER_DEVICEMAP)
	panicErr(err)
	defer has.Free()

	//Allocate memory on device
	da, err := cuda.DeviceMemAlloc(memSize)
	panicErr(err)
	defer da.Free()

	fmt.Println("Generating random numbers")

	for i := int32(0); i < tableLength; i++ {
		ha[i] = rand.Int31()
		//fmt.Println(ha[i])
	}

	fmt.Println("Random numbers generated")

	//Create start event
	start, err := cuda.NewEvent()
	panicErr(err)
	defer start.Destroy()

	//Create end event
	end, err := cuda.NewEvent()
	panicErr(err)
	defer end.Destroy()

	fmt.Println("Starting multi kernel on device")

	//Record start
	err = start.Record(nil)
	panicErr(err)

	//Start copy to device
	err = da.MemcpyToDevice(uintptr(unsafe.Pointer(&ha[0])), memSize)
	panicErr(err)

	gridSize, blockSize := cuda.Dim3{X: uint32(numBlocks), Y: 1, Z: 1}, cuda.Dim3{X: uint32(numThreads), Y: 1, Z: 1}
	bytesLocalMemory := uint64(2 * blockSize.X * uint32(intSize))

	cuda_stuff.BitonicSortStartEx(gridSize, blockSize, bytesLocalMemory, nil, da.Ptr, int32(tableLength)) // k = 2 ... 2 * blockSize.x
	for k := 4 * int32(blockSize.X); k <= int32(tableLength); k <<= 1 {                                   // k = 4 * blockSize ... tableLength
		for j := k / 2; j >= 2*int32(blockSize.X); j >>= 1 { //   j = k/2 ... 2 * blockSize.x
			err = cuda_stuff.BitonicSortMiddleEx(gridSize, blockSize, bytesLocalMemory, nil, da.Ptr, int32(tableLength), k, j)
			if err != nil {
				panic(err)
			}
		}
		cuda_stuff.BitonicSortFinishEx(gridSize, blockSize, bytesLocalMemory, nil, da.Ptr, int32(tableLength), k) //   j = 2 * blockSize.x ... 1
	}

	//Copy results from device to host
	err = da.MemcpyFromDevice(uintptr(unsafe.Pointer(&ha[0])), memSize)
	panicErr(err)

	//Trigger end event
	err = end.Record(nil)
	panicErr(err)

	//Wait for end event to be triggered
	err = end.Synchronize()
	panicErr(err)

	//Calculate time between start and end events
	elapsedTimeDevice, err := cuda.EventElapsedTime(start, end)
	panicErr(err)
	fmt.Printf("Multi kernel on device finished. Elapsed time: %f ms\n", elapsedTimeDevice)

	fmt.Println("Printing results")
	fmt.Println("ha size:", len(ha))
	prev := ha[0]
	fmt.Println(prev)
	ok := true
	i := int32(1)
	for ; i < tableLength; i++ {
		fmt.Println(ha[i])
		if ha[i] < prev {
			ok = false
			//break
		}
	}

	if ok {
		fmt.Println("Device sort is correct")
	} else {
		fmt.Println("Device sort is incorrect")
	}
}

func panicErr(err error) {
	if err != nil {
		panic(err)
	}
}
