package cuda

//#include <cuda.h>
import "C"
import (
	"errors"
)

func MemAlloc(size uint) (uintptr, error) {
	var ptr C.ulonglong
	stat := C.cuMemAlloc(&ptr, C.size_t(size))

	if stat != C.CUDA_SUCCESS {
		return 0, errors.New(ResultMap[cudaResult(stat)])
	}

	return uintptr(ptr), nil
}

func MemFree(ptr uintptr) error {
	stat := C.cuMemFree(C.ulonglong(ptr))

	if stat != C.CUDA_SUCCESS {
		return errors.New(ResultMap[cudaResult(stat)])
	}

	return nil
}
