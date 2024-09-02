package cuda

//perhaps use environ flags in compilation

//#include <cuda.h>
//#include <stdlib.h>
import "C"
import (
	"unsafe"
)

type Dim3 struct {
	X, Y, Z uint32
}

type CudaKernel struct {
	kern C.CUkernel
}

func (k *CudaKernel) Function() CudaFunction {
	return CudaFunction(unsafe.Pointer(k.kern))
}

func (k *CudaKernel) GetLibrary() (*CudaLibrary, error) {
	var mod C.CUlibrary
	stat := C.cuKernelGetLibrary(&mod, k.kern)

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return &CudaLibrary{mod}, nil
}

func (kernel *CudaKernel) LaunchKernel(grid, block Dim3, args ...unsafe.Pointer) error {
	config := C.CUlaunchConfig{
		C.uint(grid.X),
		C.uint(grid.Y),
		C.uint(grid.Z),
		C.uint(block.X),
		C.uint(block.Y),
		C.uint(block.Z),
		0,         //sharedMemBytes
		nil,       //stream, 0 for default
		nil,       //CUlaunchAttribute*
		0,         //numAttrs
		[4]byte{}, //___cgo alignment
	}

	//copy of args to C
	argc := C.malloc(C.size_t(len(args)) * C.size_t(unsafe.Sizeof(uintptr(0))))
	defer C.free(argc)

	for i, arg := range args {
		*(*unsafe.Pointer)(unsafe.Pointer(uintptr(argc) + uintptr(i)*unsafe.Sizeof(uintptr(0)))) = arg
	}

	stat := C.cuLaunchKernelEx((*C.CUlaunchConfig)(unsafe.Pointer(&config)), kernel.Function(), (*unsafe.Pointer)(&argc), nil)

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
}
