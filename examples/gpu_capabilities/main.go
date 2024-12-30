package main

import (
	"fmt"

	"github.com/InternatBlackhole/cudago/cuda"
)

func main() {
	//Initialize CUDA API for this OS thread
	var err error
	dev, err := cuda.Init(0)
	if err != nil {
		panic(err)
	}
	defer dev.Close()

	printGPUCapabilites(dev.Device)
}

// Prints some GPU capabilities
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
