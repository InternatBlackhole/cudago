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

func (k *CudaKernel) Function() (*CudaFunction, error) {
	var fun C.CUfunction
	stat := C.cuKernelGetFunction(&fun, k.kern)
	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return &CudaFunction{fun}, nil
}

func (k *CudaKernel) GetLibrary() (*CudaLibrary, error) {
	var mod C.CUlibrary
	stat := C.cuKernelGetLibrary(&mod, k.kern)

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}
	return &CudaLibrary{mod}, nil
}

func (k *CudaKernel) GetName() (string, error) {
	var name *C.char = (*C.char)(C.malloc(256))
	defer C.free(unsafe.Pointer(name))
	stat := C.cuKernelGetName((**C.char)(&name), k.kern)

	if stat != C.CUDA_SUCCESS {
		return "", NewCudaError(uint32(stat))
	}
	return C.GoString(name), nil
}

func (kernel *CudaKernel) Launch(grid, block Dim3, args ...unsafe.Pointer) error {
	return kernel.LaunchEx(grid, block, 0, nil, args...)
}

// TODO: add attributes
func (kernel *CudaKernel) LaunchEx(grid, block Dim3, sharedMem uint64, stream *CudaStream /*attributes?,*/, args ...unsafe.Pointer) error {
	fun, err := kernel.Function()
	if err != nil {
		return err
	}
	return internalLaunchEx(fun.fun, grid, block, sharedMem, stream, args...)
}

func internalLaunchEx(func_ C.CUfunction, grid, block Dim3, sharedMem uint64, stream *CudaStream, args ...unsafe.Pointer) error {
	offset := func(ptr unsafe.Pointer, off int, size uintptr) unsafe.Pointer {
		return unsafe.Pointer(uintptr(ptr) + size*uintptr(off))
	}
	config := &C.CUlaunchConfig{
		C.uint(grid.X),
		C.uint(grid.Y),
		C.uint(grid.Z),
		C.uint(block.X),
		C.uint(block.Y),
		C.uint(block.Z),
		C.uint(sharedMem), //sharedMemBytes
		nil,               //stream, 0 for default
		nil,               //CUlaunchAttribute*
		0,                 //numAttrs
		[4]byte{},         //___cgo alignment
	}

	if stream != nil {
		config.hStream = stream.stream
	}

	ptrSize := unsafe.Sizeof(uintptr(0))

	//copy of args to C
	argp := C.malloc(C.size_t(len(args)) * C.size_t(ptrSize))
	argv := C.malloc(C.size_t(len(args)) * C.size_t(ptrSize))
	defer C.free(argp)
	defer C.free(argv)

	for i := range args {
		*(*unsafe.Pointer)(offset(argp, i, ptrSize)) = offset(argv, i, ptrSize) // argp[i] = &argv[i]
		*((*uint64)(offset(argv, i, ptrSize))) = *((*uint64)(args[i]))          // argv[i] = args[i]
	}

	stat := C.cuLaunchKernelEx(config, func_, (*unsafe.Pointer)(argp), nil)

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}
	return nil
}
