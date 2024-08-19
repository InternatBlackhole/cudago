package cuda

//perhaps use environ flags in compilation

//#include <cuda.h>
import "C"
import (
	"errors"
	"unsafe"
)

type Dim3 struct {
	x, y, z uint32
}

type CudaFunction C.CUfunction

func LaunchKernel(kernel CudaFunction, grid, block Dim3, args ...unsafe.Pointer) error {
	config := C.CUlaunchConfig{
		C.uint(grid.x),
		C.uint(grid.y),
		C.uint(grid.z),
		C.uint(block.x),
		C.uint(block.y),
		C.uint(block.z),
		0,         //sharedMemBytes
		nil,       //stream, 0 for default
		nil,       //CUlaunchAttribute*
		0,         //numAttrs
		[4]byte{}, //___cgo alignment
	}

	stat := C.cuLaunchKernelEx((*C.CUlaunchConfig)(unsafe.Pointer(&config)), kernel, &args[0], nil)

	return errors.New(ResultMap[cudaResult(stat)])
}
