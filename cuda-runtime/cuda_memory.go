package cudaruntime

// #include <cuda_runtime.h>
import "C"
import "unsafe"

type CudaPointer unsafe.Pointer

func CudaMalloc(size uint64) (CudaPointer, error) {
	var ptr CudaPointer
	err := C.cudaMalloc((*unsafe.Pointer)(&ptr), C.size_t(size))
	if err != C.cudaSuccess {
		return nil, NewCudaError(uint32(err))
	}
	return ptr, nil
}

func CudaFree(ptr CudaPointer) error {
	err := C.cudaFree(unsafe.Pointer(ptr))
	if err != C.cudaSuccess {
		return NewCudaError(uint32(err))
	}
	return nil
}

type CudaMemcpyKind int32

const (
	CudaMemcpyHostToHost     CudaMemcpyKind = 0 // C.cudaMemcpyHostToHost
	CudaMemcpyHostToDevice   CudaMemcpyKind = 1 // C.cudaMemcpyHostToDevice
	CudaMemcpyDeviceToHost   CudaMemcpyKind = 2 // C.cudaMemcpyDeviceToHost
	CudaMemcpyDeviceToDevice CudaMemcpyKind = 3 // C.cudaMemcpyDeviceToDevice
	CudaMemcpyDefault        CudaMemcpyKind = 4 // C.cudaMemcpyDefault
)

func CudaMemcpy(dst uintptr, src uintptr, size uint64, kind CudaMemcpyKind) error {
	err := C.cudaMemcpy(unsafe.Pointer(dst), unsafe.Pointer(src), C.size_t(size), uint32(kind))
	if err != C.cudaSuccess {
		return NewCudaError(uint32(err))
	}
	return nil
}

// TODO: add async version of CudaMemcpy? Streams needed for that.

func CudaMemset(ptr CudaPointer, value int32, count uint64) error {
	err := C.cudaMemset(unsafe.Pointer(ptr), C.int(value), C.size_t(count))
	if err != C.cudaSuccess {
		return NewCudaError(uint32(err))
	}
	return nil
}
