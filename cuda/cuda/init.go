package cuda

//#include <cuda.h>
import "C"
import (
	"errors"
	"runtime"
)

func Init() error {
	runtime.LockOSThread() // Lock this goroutine to a OS thread
	err := C.cuInit(0)
	if err != C.CUDA_SUCCESS {
		return errors.New(ResultMap[cudaResult(err)])
	}
	return nil
}
