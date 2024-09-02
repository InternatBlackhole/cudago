package cuda

//#include <cuda.h>
import "C"
import (
	"runtime"
)

var (
	IsCudaInitialized bool = false
)

func Init() error {
	if IsCudaInitialized {
		return nil
	}
	runtime.LockOSThread() // Lock this goroutine to a OS thread
	err := C.cuInit(0)
	if err != C.CUDA_SUCCESS {
		return NewCudaError(uint32(err))
	}
	IsCudaInitialized = true
	return nil
}

func DriverVersion() (int32, error) {
	var version int32
	err := C.cuDriverGetVersion((*C.int)(&version))
	if err != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(err))
	}
	return version, nil
}
