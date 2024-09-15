package cuda

//#include <cuda.h>
import "C"

var (
	isCudaInitialized bool = false
)

/*
 * Initializes the CUDA driver API for the current process.
 */
func Init() Result {
	if isCudaInitialized {
		return nil
	}
	//runtime.LockOSThread() // the init function initializes the whole process
	err := C.cuInit(0)
	if err != C.CUDA_SUCCESS {
		return NewCudaError(uint32(err))
	}
	isCudaInitialized = true
	return nil
}

func DriverVersion() (int32, Result) {
	var version int32
	err := C.cuDriverGetVersion((*C.int)(&version))
	if err != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(err))
	}
	return version, nil
}
