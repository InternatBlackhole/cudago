package main

import (
	//"cudaTest/edge"
	//"cudaTest/ints"
	//sorting "cudaTest/sort"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unsafe"

	"github.com/InternatBlackhole/cudago/cuda"
)

// format: <operation>;"<image_name>";<time in ms>
const reportFormat = "%s;\"%s\";%f\n"
const reportHeader = "Operation;Image;Time\n"

func main() {

	var err error

	// inicializiramo napravo
	dev, err := cuda.Init(0)
	if err != nil {
		panic(err)
	}
	defer dev.Close()

	outputDir := os.Args[1]
	null := outputDir == os.DevNull
	if null {
		fmt.Fprintln(os.Stderr, "VOIDING IMAGES!!!!\n")
	}

	//If output directory doesn't exist, create it
	if _, err := os.Stat(outputDir); os.IsNotExist(err) {
		err = os.Mkdir(outputDir, 0755)
		panicErr(err)
	}

	pics := os.Args[2:]
	if len(pics) == 0 {
		fmt.Fprintln(os.Stderr, "No pictures provided")
		os.Exit(1)
	}

	//Print header
	fmt.Printf(reportHeader)

	/*fmt.Fprintln(os.Stderr, "Calling all kernels once to avoid first call overhead...")
	{
		grid, block := cuda.Dim3{X: 1, Y: 1, Z: 1}, cuda.Dim3{X: 1, Y: 1, Z: 1}
		d, _ := cuda.DeviceMemAlloc(1)
		edge.Borders(grid, block, d.Ptr, 0, 0, d.Ptr, 0)
		sorting.BitonicSortStart(grid, block, d.Ptr, 0)
		sorting.BitonicSortMiddle(grid, block, d.Ptr, 0, 0, 0)
		sorting.BitonicSortFinish(grid, block, d.Ptr, 0, 0)
		ints.AddToAll(grid, block, d.Ptr, 0, 0)
		cuda.CurrentContextSynchronize()
		d.Free()
	}
	fmt.Fprintln(os.Stderr, "Finished calling all kernels once")*/

	fmt.Fprintln(os.Stderr, "Starting test: edge recognition...")

	for _, picPath := range pics {
		picBaseName := filepath.Base(picPath)
		picNoExt := strings.SplitN(picBaseName, ".", 2)[0]

		reader, err := os.Open(picPath)
		panicErr(err)

		var writer *os.File
		if null {
			writer, err = os.Open(os.DevNull)
		} else {
			writer, err = os.Create(filepath.Join(outputDir, picNoExt+".jpg"))
		}
		panicErr(err)

		borders(reader, writer, picBaseName)

		reader.Close()
		writer.Close()
		fmt.Fprintln(os.Stderr)
	}

	fmt.Fprintln(os.Stderr, "Ended test: edge recognition")
	fmt.Fprintln(os.Stderr, "Starting test: bitonic sort with increase...")

	numThreads := uint32(128)

	// 8192 int32s = 32KB, 2^25 ints = 128MB
	for size := int32(1 << 13); size <= 1<<25; size <<= 1 {
		fmt.Fprintf(os.Stderr, "Size: %d\n", size)

		elemSize := uint64(unsafe.Sizeof(int32(0)))
		arr, err := cuda.HostMemAlloc[int32](uint64(size), elemSize) //make([]int, size)
		for i := range size {
			arr.Arr[i] = size - i
		}

		//arr, err := cuda.RegisterAllocationHost(harr, elemSize, cuda.CU_MEMHOSTREGISTER_DEVICEMAP)
		//panicErr(err)

		d_array, err := cuda.DeviceMemAlloc(uint64(size) * elemSize)
		panicErr(err)

		err = d_array.MemcpyToDevice(unsafe.Pointer(arr.Ptr), arr.ActualSize)
		panicErr(err)

		sort(d_array, uint32(size), elemSize, numThreads)
		increase(d_array, uint32(size), elemSize, numThreads, 10)

		err = d_array.MemcpyFromDevice(unsafe.Pointer(arr.Ptr), arr.ActualSize)
		panicErr(err)

		arr.Free()
		d_array.Free()
	}
	fmt.Fprintln(os.Stderr, "Ended test: bitonic sort with increase")
	fmt.Println()
}

func panicErr(err error) {
	if err != nil {
		panic(err)
	}
}
