package cuda

//#include <cuda.h>
import "C"
import "runtime"

var (
	isCudaInitialized bool = false
)

/*
 * Initializes the CUDA driver API for the current process.
 */
func Init() error {
	if isCudaInitialized {
		return nil
	}
	runtime.LockOSThread() // Lock this goroutine to a OS thread. no need, the init function initilzes the whole process
	err := C.cuInit(0)
	if err != C.CUDA_SUCCESS {
		return NewCudaError(uint32(err))
	}
	isCudaInitialized = true
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
