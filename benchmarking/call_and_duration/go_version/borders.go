package main

import (
	"cudaTest/edge"
	"fmt"
	"image"
	"image/jpeg"
	"math"
	"os"
	"time"
	"unsafe"

	"github.com/InternatBlackhole/cudago/cuda"
)

func borders(reader, writer *os.File, picBaseName string) {
	origImage, _, err := image.Decode(reader)
	//panicErr(err)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error decoding image %s: %s\n", picBaseName, err)
		return
	}

	img := rgbaToGray(origImage)
	//Optimization for faster memory transfers
	arr, err := cuda.RegisterAllocationHost(img.Pix, 1, cuda.CU_MEMHOSTREGISTER_DEVICEMAP)
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
	fmt.Fprintf(os.Stderr, "CUDA config for %s: block_size: %d, grid_size: (x: %d, y: %d, z: %d)\n", picBaseName, blockSize, dimGrid.X, dimGrid.Y, dimGrid.Z)

	start, err := cuda.NewEvent()
	panicErr(err)
	defer start.Destroy()

	end, err := cuda.NewEvent()
	panicErr(err)
	defer end.Destroy()

	edgesKernelStart, err := cuda.NewEvent()
	panicErr(err)
	defer edgesKernelStart.Destroy()

	edgesKernelEnd, err := cuda.NewEvent()
	panicErr(err)
	defer edgesKernelEnd.Destroy()

	finalImg, err := cuda.HostMemAlloc[byte](size, 1)
	panicErr(err)
	defer finalImg.Free()

	err = start.Record(nil)
	panicErr(err)

	fmt.Fprintf(os.Stderr, "Copy start %s to device...\n", picBaseName)
	err = grayImg.MemcpyToDevice(unsafe.Pointer(&img.Pix[0]), uint64(len(img.Pix)))
	panicErr(err)
	fmt.Fprintf(os.Stderr, "Copy ended %s to device\n", picBaseName)

	err = edgesKernelStart.Record(nil)
	panicErr(err)

	fmt.Fprintf(os.Stderr, "Starting kernel for %s...\n", picBaseName)
	goStart := time.Now()
	err = edge.Borders(dimGrid, dimBlock, grayImg.Ptr, int32(imgSize.X), int32(imgSize.Y), grad.Ptr, int32(size))
	took := time.Since(goStart)
	panicErr(err)
	fmt.Fprintf(os.Stderr, "Kernel ended for %s\n", picBaseName)

	err = edgesKernelEnd.Record(nil)
	panicErr(err)

	//err = edgesKernelEnd.Synchronize()
	//panicErr(err)

	fmt.Fprintf(os.Stderr, "Copy start %s from device...\n", picBaseName)
	err = grad.MemcpyFromDevice(unsafe.Pointer(&finalImg.Arr[0]), uint64(len(finalImg.Arr)))
	panicErr(err)
	fmt.Fprintf(os.Stderr, "Copy ended %s from device\n", picBaseName)

	err = end.Record(nil)
	panicErr(err)

	err = end.Synchronize()
	panicErr(err)

	elapsedTimeMs, err := cuda.EventElapsedTime(edgesKernelStart, edgesKernelEnd)
	panicErr(err)

	fmt.Printf(reportFormat, "Image_KernelCall", picBaseName, float64(took.Nanoseconds())/1000)
	fmt.Printf(reportFormat, "Image_KernelDur", picBaseName, elapsedTimeMs)

	final := image.NewGray(img.Bounds())
	final.Pix = finalImg.Arr

	err = jpeg.Encode(writer, final, nil)
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
