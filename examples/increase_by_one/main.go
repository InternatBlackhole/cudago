package main

import (
	"fmt"
	"increase_by_one/cu"
	"math"

	"github.com/InternatBlackhole/cudago/cuda"
)

func main() {
	var err error
	dev, err := cuda.Init(0)
	if err != nil {
		panic(err)
	}
	defer dev.Close()

	//printGPUCapabilites(dev.Device)
	//return

	fmt.Println("Init arr")

	//ints := make([]int, 1<<9)
	ints, err := cuda.ManagedMemAlloc[int32](1<<9, 4, cuda.CU_MEM_ATTACH_HOST)
	if err != nil {
		panic(err)
	}

	for i := range ints.Arr {
		ints.Arr[i] = int32(i)
		fmt.Printf("ints[%d] = %d (%x)\n", i, i, i)
	}

	start, err := cuda.NewEvent()
	if err != nil {
		panic(err)
	}

	stop, err := cuda.NewEvent()
	if err != nil {
		panic(err)
	}

	toAdd := 1
	calc := uint32(math.Ceil(float64(len(ints.Arr)) / float64(32)))

	grid, block := cuda.Dim3{X: calc, Y: 1, Z: 1}, cuda.Dim3{X: 32, Y: 1, Z: 1}

	fmt.Println("Grid: ", grid, "Block: ", block)
	fmt.Println("Working on device...")

	err = start.Record(nil)
	if err != nil {
		panic(err)
	}

	err = cu.AddToAll(grid, block, ints.Ptr, toAdd, len(ints.Arr))
	if err != nil {
		panic(err)
	}

	err = stop.Record(nil)
	if err != nil {
		panic(err)
	}

	err = stop.Synchronize()
	if err != nil {
		panic(err)
	}

	fmt.Println("Device finished")

	ms, err := cuda.EventElapsedTime(start, stop)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Took: %f ms\n", ms)
	fmt.Println("checking...")
	for i := range ints.Arr {
		if ints.Arr[i] != int32(i)+int32(toAdd) {
			fmt.Printf("Error at: %d Expected: %d Got: %d (%x)\n", i, (i)+toAdd, ints.Arr[i], ints.Arr[i])
		}
	}
}

func printGPUCapabilites(gpu *cuda.Device) {
	name, err := gpu.Name()
	if err != nil {
		panic(err)
	}
	fmt.Println("Device: ", name)
	warp_size, err := gpu.GetAttribute(cuda.CU_DEVICE_ATTRIBUTE_WARP_SIZE)
	if err != nil {
		panic(err)
	}
	fmt.Println("Warp size: ", warp_size)
	max_threads_per_block, err := gpu.GetAttribute(cuda.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
	if err != nil {
		panic(err)
	}
	fmt.Println("Max threads per block: ", max_threads_per_block)
	max_block_dim_x, err := gpu.GetAttribute(cuda.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
	if err != nil {
		panic(err)
	}
	fmt.Println("Max block dim x: ", max_block_dim_x)
	max_block_dim_y, err := gpu.GetAttribute(cuda.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)
	if err != nil {
		panic(err)
	}
	fmt.Println("Max block dim y: ", max_block_dim_y)
	max_block_dim_z, err := gpu.GetAttribute(cuda.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)
	if err != nil {
		panic(err)
	}
	fmt.Println("Max block dim z: ", max_block_dim_z)
	max_grid_dim_x, err := gpu.GetAttribute(cuda.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)
	if err != nil {
		panic(err)
	}
	fmt.Println("Max grid dim x: ", max_grid_dim_x)
	max_grid_dim_y, err := gpu.GetAttribute(cuda.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
	if err != nil {
		panic(err)
	}
	fmt.Println("Max grid dim y: ", max_grid_dim_y)
	max_grid_dim_z, err := gpu.GetAttribute(cuda.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)
	if err != nil {
		panic(err)
	}
	fmt.Println("Max grid dim z: ", max_grid_dim_z)
	mp_count, err := gpu.GetAttribute(cuda.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
	if err != nil {
		panic(err)
	}
	fmt.Println("Multiprocessor count: ", mp_count)
}
