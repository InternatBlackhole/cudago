package cuda_runtime

//#include <stdlib.h>
//#include <cuda_runtime.h>
import "C"
import "unsafe"

const (
	//move it into a cgo comment and let it be set by a C compiler?
	pointerSize = 8
)

type Dim3 struct {
	X int
	Y int
	Z int
}

func (d Dim3) toC() C.dim3 {
	return C.dim3{C.uint(d.X), C.uint(d.Y), C.uint(d.Z)}
}

type CudaFunction uintptr

func CudaLaunchKernel(f CudaFunction, gridDim Dim3, blockDim Dim3, sharedMemBytes uint64, kernelArgs []unsafe.Pointer /*, stream C.cudaStream_t*/) error {
	//TODO: Add for cudaStream_t

	//copy, in C, kernelArgs to avoid collection errors
	args := C.malloc(C.size_t(len(kernelArgs) * pointerSize))
	defer C.free(args)

	for i := range kernelArgs {
		*(*unsafe.Pointer)(unsafe.Pointer(uintptr(args) + uintptr(i*pointerSize))) = kernelArgs[i]
	}

	//launch kernel
	err := C.cudaLaunchKernel(
		unsafe.Pointer(uintptr(f)),
		gridDim.toC(),
		blockDim.toC(),
		(*unsafe.Pointer)(args), //kernelArgs
		C.ulong(sharedMemBytes),
		nil, //stream
	)

	if err != C.cudaSuccess {
		return NewCudaError(uint32(err))
	}

	return nil
}

func CudaLaunchCooperativeKernel(f CudaFunction, gridDim Dim3, blockDim Dim3, sharedMemBytes uint64, kernelArgs []unsafe.Pointer /*, stream C.cudaStream_t*/) error {
	//TODO: not all devices support cooperative launch
	//copy, in C, kernelArgs to avoid collection errors
	args := C.malloc(C.size_t(len(kernelArgs) * pointerSize))
	defer C.free(args)

	for i := range kernelArgs {
		*(*unsafe.Pointer)(unsafe.Pointer(uintptr(args) + uintptr(i*pointerSize))) = kernelArgs[i]
	}

	//launch kernel
	err := C.cudaLaunchCooperativeKernel(
		unsafe.Pointer(uintptr(f)),
		gridDim.toC(),
		blockDim.toC(),
		(*unsafe.Pointer)(args), //kernelArgs
		C.ulong(sharedMemBytes),
		nil, //stream
	)

	if err != C.cudaSuccess {
		return NewCudaError(uint32(err))
	}

	return nil
}
