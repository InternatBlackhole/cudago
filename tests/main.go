package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"math"
	"os"

	"github.com/InternatBlackhole/cudago/cuda"
	"github.com/InternatBlackhole/cudago/tests/cuda_stuff"
)

func main() {
	err := cuda.Init()
	panicErr(err)

	dev, err := cuda.DeviceGet(0)
	panicErr(err)

	pctx, err := cuda.DevicePrimaryCtxRetain(dev)
	panicErr(err)
	defer pctx.Release()

	err = pctx.SetCurrent()
	panicErr(err)

	err = cuda_stuff.InitLibrary_edges()
	panicErr(err)
	defer cuda_stuff.CloseLibrary_edges()

	reader, err := os.Open("test10.jpg")
	panicErr(err)
	defer reader.Close()

	origImage, err := jpeg.Decode(reader)
	panicErr(err)

	img := rgbaToGray(origImage)

	imgSize := img.Bounds().Size()
	size := uint64(imgSize.X * imgSize.Y)

	grayImg, err := cuda.MemAlloc(size)
	panicErr(err)
	defer grayImg.MemFree()

	grad, err := cuda.MemAlloc(size)
	panicErr(err)
	defer grad.MemFree()

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

	finalImg := make([]byte, size)

	err = start.Record()
	panicErr(err)

	err = grayImg.MemcpyToDevice(img.Pix)
	panicErr(err)

	err = cuda_stuff.Borders(dimGrid, dimBlock, grayImg.Ptr, imgSize.X, imgSize.Y, grad.Ptr, int(size))
	panicErr(err)

	err = grad.MemcpyFromDevice(finalImg)
	panicErr(err)

	err = end.Record()
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
	final.Pix = finalImg

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
