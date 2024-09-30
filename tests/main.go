package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"math"
	"os"
	"time"
	"unsafe"

	"github.com/InternatBlackhole/cudago/cuda"
	"github.com/InternatBlackhole/cudago/tests/cuda_stuff"
)

func main() {
	var err error
	dev, err := cuda.Init(0)
	panicErr(err)
	defer dev.Close()

	borders()
	//size := 1 << 15
	//multiKernel(int(math.Min(1024, float64(size))), size)
}

func borders() {
	reader, err := os.Open("test10.jpg")
	panicErr(err)
	defer reader.Close()

	origImage, err := jpeg.Decode(reader)
	panicErr(err)

	img := rgbaToGray(origImage)
	arr, err := cuda.RegisterAllocationHost(img.Pix, 1, cuda.CU_MEMHOSTREGISTER_READ_ONLY)
	panicErr(err)
	defer arr.Free()

	imgSize := img.Bounds().Size()
	size := uint64(imgSize.X * imgSize.Y)

	grayImg, err := cuda.DeviceMemAlloc(size)
	panicErr(err)
	defer grayImg.Free()

	grad, err := cuda.DeviceMemAlloc(size)
	panicErr(err)
	defer grad.Free()

	blockSize := uint32(32)

	dimBlock := cuda.Dim3{X: blockSize, Y: blockSize, Z: 1}
	dimGrid := cuda.Dim3{
		X: uint32(math.Ceil(float64(imgSize.X) / float64(blockSize))),
		Y: uint32(math.Ceil(float64(imgSize.Y) / float64(blockSize))),
		Z: 1,
	}
	fmt.Printf("CUDA config: block_size: %d, grid_size: (x: %d, y: %d, z: %d)\n", blockSize, dimGrid.X, dimGrid.Y, dimGrid.Z)

	start, err := cuda.NewEvent()
	panicErr(err)
	defer start.Destroy()

	end, err := cuda.NewEvent()
	panicErr(err)
	defer end.Destroy()

	//finalImg := make([]byte, size)
	finalImg, err := cuda.HostMemAlloc[byte](size, 1)
	panicErr(err)
	defer finalImg.Free()

	err = start.Record(nil)
	panicErr(err)

	err = grayImg.MemcpyToDevice(unsafe.Pointer(&img.Pix[0]), uint64(len(img.Pix)))
	panicErr(err)

	err = cuda_stuff.Borders(dimGrid, dimBlock, grayImg.Ptr, imgSize.X, imgSize.Y, grad.Ptr, int(size))
	panicErr(err)
	defer cuda_stuff.CloseLibrary(cuda_stuff.KeyEdges)

	//err = grad.MemcpyFromDevice(finalImg)
	err = grad.MemcpyFromDevice(unsafe.Pointer(&finalImg.Arr[0]), uint64(len(finalImg.Arr)))
	panicErr(err)

	err = end.Record(nil)
	panicErr(err)

	err = end.Synchronize()
	panicErr(err)

	elapsedTime, err := cuda.EventElapsedTime(start, end)
	panicErr(err)

	fmt.Printf("Elapsed time: %f ms\n", elapsedTime)

	outFile, err := os.Create("output.jpg")
	panicErr(err)
	defer outFile.Close()

	final := image.NewGray(img.Bounds())
	final.Pix = finalImg.Arr
	//final.Pix = finalImg

	err = jpeg.Encode(outFile, final, nil)
	panicErr(err)

	fmt.Println("Image saved to output.jpg")
}

func multiKernel(numThreads, tableLength int) {
	var err error
	numBlocks := (tableLength/2-1)/numThreads + 1

	intSize := uint64(unsafe.Sizeof(int32(0)))
	memSize := uint64(tableLength) * intSize

	a := make([]int, tableLength)
	ha := make([]int, tableLength)

	has, err := cuda.RegisterAllocationHost(ha, intSize, cuda.CU_MEMHOSTREGISTER_DEVICEMAP)
	panicErr(err)
	defer has.Free()

	da, err := cuda.DeviceMemAlloc(memSize)
	panicErr(err)
	defer da.Free()

	for i := 0; i < tableLength; i++ {
		a[i] = tableLength - i //rand.Int()
		ha[i] = a[i]
	}

	start, err := cuda.NewEvent()
	panicErr(err)
	defer start.Destroy()

	end, err := cuda.NewEvent()
	panicErr(err)
	defer end.Destroy()

	fmt.Println("Starting multi kernel test on device")

	err = start.Record(nil)
	panicErr(err)

	err = da.MemcpyToDevice(unsafe.Pointer(&ha[0]), memSize)
	panicErr(err)

	gridSize, blockSize := cuda.Dim3{X: uint32(numBlocks), Y: 1, Z: 1}, cuda.Dim3{X: uint32(numThreads), Y: 1, Z: 1}
	//err = cuda_stuff.BitonicSortStartEx(gridSize, blockSize, uint64(2*blockSize.X*uint32(intSize)), nil, da.Ptr, tableLength)
	err = cuda_stuff.BitonicSortStart(gridSize, blockSize, da.Ptr, tableLength)
	panicErr(err)

	for k := int(4 * blockSize.X); k <= tableLength; k <<= 2 {
		for j := k / 2; j > int(2*blockSize.X); j >>= 1 {
			//err = cuda_stuff.BitonicSortMiddleEx(gridSize, blockSize, uint64(2*blockSize.X*uint32(intSize)), nil, da.Ptr, tableLength, k, j)
			err = cuda_stuff.BitonicSortMiddle(gridSize, blockSize, da.Ptr, tableLength, k, j)
			panicErr(err)
		}
		//err = cuda_stuff.BitonicSortFinishEx(gridSize, blockSize, uint64(2*blockSize.X*uint32(intSize)), nil, da.Ptr, tableLength, k)
		err = cuda_stuff.BitonicSortFinish(gridSize, blockSize, da.Ptr, tableLength, k)
		panicErr(err)
	}

	err = cuda.CurrentContextSynchronize()
	panicErr(err)

	err = da.MemcpyFromDevice(unsafe.Pointer(&ha[0]), memSize)
	panicErr(err)

	err = end.Record(nil)
	panicErr(err)

	err = end.Synchronize()
	panicErr(err)

	elapsedTimeDevice, err := cuda.EventElapsedTime(start, end)
	panicErr(err)
	fmt.Printf("Multi kernel test on device finished. Elapsed time: %f ms\n", elapsedTimeDevice)

	fmt.Println("Starting multi kernel test on host")

	timeStart := time.Now()

	var i2, dec, temp int
	for k := 2; k <= tableLength; k <<= 1 {
		for j := k / 2; j > 0; j >>= 1 {
			for i1 := 0; i1 < tableLength; i1++ {
				i2 = i1 ^ j
				dec = i1 & k
				if i2 > i1 {
					if (dec == 0 && a[i1] > a[i2]) || (dec != 0 && a[i1] < a[i2]) {
						temp = a[i1]
						a[i1] = a[i2]
						a[i2] = temp
					}
				}
			}
		}
	}

	timeEnd := time.Now()
	elapsedTimeHost := timeEnd.Sub(timeStart)
	fmt.Printf("Multi kernel test on host finished. Elapsed time: %v ms\n", elapsedTimeHost.Nanoseconds()/1e6)

	//okDevice, okHost, prevDev, prevHost := true, true, ha[0], a[0]
	i := 0
	for ; i < tableLength; i++ {
		//okDevice = okDevice && (prevDev <= ha[i])
		//okHost = okHost && (prevHost <= a[i])
		if ha[i] != a[i] {
			break
		}
	}

	if i < tableLength {
		fmt.Printf("Host sorting and device sorting are different at index %d\n", i)
	} else {
		fmt.Println("Host sorting and device sorting are the same")
	}

	//fmt.Println("Device sort is correct:", okDevice)
	//fmt.Println("Host sort is correct:", okHost)
}

func rgbaToGray(img image.Image) *image.Gray {
	var (
		bounds = img.Bounds()
		gray   = image.NewGray(bounds)
	)
	for x := 0; x < bounds.Max.X; x++ {
		for y := 0; y < bounds.Max.Y; y++ {
			var rgba = img.At(x, y)
			gray.Set(x, y, rgba)
		}
	}
	return gray
}

func panicErr(err error) {
	if err != nil {
		panic(err)
	}
}
