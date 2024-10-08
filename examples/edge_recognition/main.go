package main

import (
	"edge_recognition/cuda_stuff"
	"fmt"
	"image"
	"image/jpeg"
	"math"
	"os"
	"unsafe"

	"github.com/InternatBlackhole/cudago/cuda"
)

func main() {
	var err error
	dev, err := cuda.Init(0)
	panicErr(err)
	defer dev.Close()

	borders()
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

	err = grayImg.MemcpyToDevice(uintptr(unsafe.Pointer(&img.Pix[0])), uint64(len(img.Pix)))
	panicErr(err)

	err = cuda_stuff.Borders(dimGrid, dimBlock, grayImg.Ptr, int32(imgSize.X), int32(imgSize.Y), grad.Ptr, int32(size))
	panicErr(err)
	defer cuda_stuff.CloseLibrary(cuda_stuff.KeyEdges)

	//err = grad.MemcpyFromDevice(finalImg)
	err = grad.MemcpyFromDevice(uintptr(unsafe.Pointer(&finalImg.Arr[0])), uint64(len(finalImg.Arr)))
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
