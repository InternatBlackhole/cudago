package cuda_test

import (
	"image"
	"image/jpeg"
	"image/png"
	"math"
	"os"
	"runtime"
	"testing"
	"unsafe"

	"github.com/InternatBlackhole/cudago/cuda"
)

const (
	blockSize = 32
)

func TestEdgesKernel(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	var err error
	err = cuda.Init()
	if err != nil {
		t.Fatal(err)
	}

	dev, err := cuda.DeviceGet(0)
	if err != nil {
		t.Fatal(err)
	}

	pctx, err := cuda.DevicePrimaryCtxRetain(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer pctx.Release()

	_, active, err := pctx.GetState()
	if err != nil {
		t.Fatal(err)
	} else if !active {
		t.Fatal("Primary context is not active")
	}

	err = cuda.SetCurrentContext(&pctx.Context)
	if err != nil {
		t.Fatal(err)
	}

	lib, err := cuda.LoadLibraryFromPath("../tests/edgesC.ptx", nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	kerns, err := lib.GetKernels()
	if err != nil {
		t.Fatal(err)
	}
	if len(kerns) != 1 {
		t.Fatal("No kernels found in the library, should have exactly one")
	} else {
		t.Log("Kernel found in the library")
	}

	kern := kerns[0]
	name, err := kern.GetName()
	if err != nil {
		t.Fatal(err)
	}
	if name != "borders" {
		t.Fatalf("Expected kernel name to be 'Edges', got %s", name)
	} else {
		t.Log("Kernel name is correct")
	}

	reader, err := os.OpenFile("../tests/test10.jpg", os.O_RDONLY, 0644)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()

	origImg, err := jpeg.Decode(reader)
	if err != nil {
		t.Fatal(err)
	}

	img := rgbaToGray(origImg)

	imgSize := img.Bounds().Size()
	t.Log("Image size is", imgSize)

	size := uint64(imgSize.X * imgSize.Y * 1) // one color (byte) channel output

	// Allocate memory for the image, and copy the image data to the device
	grayImgData, err := cuda.DeviceMemAlloc(size)
	if err != nil {
		t.Fatal(err)
	}
	defer grayImgData.Free()

	grad, err := cuda.DeviceMemAlloc(size)
	if err != nil {
		t.Fatal(err)
	}
	defer grad.Free()

	dimBlock := cuda.Dim3{X: blockSize, Y: blockSize, Z: 1}
	dimGrid := cuda.Dim3{
		X: uint32(math.Ceil(float64(imgSize.X) / blockSize)),
		Y: uint32(math.Ceil(float64(imgSize.Y) / blockSize)),
		Z: 1,
	}
	t.Logf("CUDA config: block_size: %d, grid_size: (x: %d, y: %d, z: %d)\n", blockSize, dimGrid.X, dimGrid.Y, dimGrid.Z)

	start, err := cuda.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	defer start.Destroy()

	end, err := cuda.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	defer end.Destroy()

	finalImg := make([]byte, size)

	err = start.Record(nil)
	if err != nil {
		t.Fatal(err)
	}

	err = grayImgData.MemcpyToDevice(unsafe.Pointer(&img.Pix[0]), uint64(len(img.Pix)))
	if err != nil {
		t.Fatal(err)
	}

	err = kern.Launch(dimGrid, dimBlock, unsafe.Pointer(&grayImgData.Ptr), unsafe.Pointer(&imgSize.X),
		unsafe.Pointer(&imgSize.Y), unsafe.Pointer(&grad.Ptr), unsafe.Pointer(&size))
	if err != nil {
		t.Fatal(err)
	}

	err = grad.MemcpyFromDevice(unsafe.Pointer(&finalImg[0]), uint64(len(finalImg)))
	if err != nil {
		t.Fatal(err)
	}

	err = end.Record(nil)
	if err != nil {
		t.Fatal(err)
	}

	err = end.Synchronize()
	if err != nil {
		t.Fatal(err)
	}

	elapsedTime, err := cuda.EventElapsedTime(start, end)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Elapsed time on GPU: %f ms\n", elapsedTime)
	newFile := "../tests/test10_edges.png"
	writer, err := os.OpenFile(newFile, os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		t.Fatal(err)
	}
	defer writer.Close()

	final := image.NewGray(image.Rect(0, 0, imgSize.X, imgSize.Y))
	final.Pix = finalImg

	err = png.Encode(writer, final)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Image saved to %s\n", newFile)
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
