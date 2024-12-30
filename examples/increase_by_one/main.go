package main

import (
	"fmt"
	"increase_by_one/cu"
	"math"

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

	fmt.Println("Init arr")

	//Allocate managed memory
	ints, err := cuda.ManagedMemAlloc[int32](1<<9, 4)
	if err != nil {
		panic(err)
	}

	//Fill array with numbers
	for i := range ints.Arr {
		ints.Arr[i] = int32(i)
		fmt.Printf("ints[%d] = %d (%x)\n", i, i, i)
	}

	//Create event to signal GPU operation start time
	start, err := cuda.NewEvent()
	if err != nil {
		panic(err)
	}

	//Create event to signal GPU operation stop time
	stop, err := cuda.NewEvent()
	if err != nil {
		panic(err)
	}

	//Add 3 to every number
	toAdd := 3
	//Repeat adding 3 times
	repeat := 3

	//Calculate grid size. In this case one cell will operate on at most 32 ints
	calc := uint32(math.Ceil(float64(len(ints.Arr)) / float64(32)))

	//Specify grid and block size
	grid, block := cuda.Dim3{X: calc, Y: 1, Z: 1}, cuda.Dim3{X: 32, Y: 1, Z: 1}

	fmt.Println("Grid: ", grid, "Block: ", block)
	fmt.Println("Working on device...")

	//Signal start of GPU opration
	err = start.Record(nil)
	if err != nil {
		panic(err)
	}

	//Run CUDA kernel <repeat> times
	for range repeat {
		//Calls kernel "addToAll" from file "ints.cu"
		err = cu.AddToAll(grid, block, ints.Ptr, int32(toAdd), int32(len(ints.Arr)))
		if err != nil {
			panic(err)
		}
	}

	//Signal stop of GPU operation
	err = stop.Record(nil)
	if err != nil {
		panic(err)
	}

	//Wait for all CUDA operations to complete
	err = stop.Synchronize()
	if err != nil {
		panic(err)
	}

	fmt.Println("Device finished")

	//Calculate time between start and stop events
	ms, err := cuda.EventElapsedTime(start, stop)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Took: %f ms\n", ms)

	//Check if calculation was correct
	fmt.Println("checking...")
	for i := range ints.Arr {
		if ints.Arr[i] != int32(i)+int32(repeat)*int32(toAdd) {
			fmt.Printf("Error at: %d Expected: %d Got: %d (%x)\n", i, (i)+toAdd*repeat, ints.Arr[i], ints.Arr[i])
		} else {
			fmt.Printf("ints[%d] = %d\n", i, ints.Arr[i])
		}
	}
}
