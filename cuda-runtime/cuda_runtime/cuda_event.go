package cuda_runtime

// #include <cuda_runtime.h>
import "C"
import (
	"runtime"
	"unsafe"
)

type CudaEvent struct {
	event unsafe.Pointer
}

func CudaEventCreate() (*CudaEvent, error) {
	var event C.cudaEvent_t
	err := C.cudaEventCreate(&event)

	if err != C.cudaSuccess {
		return nil, NewCudaError(uint32(err))
	}

	return &CudaEvent{unsafe.Pointer(&event)}, nil
}

func (e CudaEvent) Destroy() error {
	err := C.cudaEventDestroy(C.cudaEvent_t(e.event))

	if err != C.cudaSuccess {
		return NewCudaError(uint32(err))
	}

	return nil
}

func (e CudaEvent) Record() error {
	// TODO: Add for cudaStream_t
	err := C.cudaEventRecord(C.cudaEvent_t(e.event), nil) // 0 (nil) is the default stream

	if err != C.cudaSuccess {
		return NewCudaError(uint32(err))
	}

	return nil
}

func (e CudaEvent) Synchronize() error {
	//will block until event is done
	runtime.LockOSThread() //probably needed but not sure
	err := C.cudaEventSynchronize(C.cudaEvent_t(e.event))
	runtime.UnlockOSThread()

	if err != C.cudaSuccess {
		return NewCudaError(uint32(err))
	}

	return nil
}

func (e CudaEvent) ElapsedTime(other CudaEvent) (float32, error) {
	var ms C.float
	err := C.cudaEventElapsedTime(&ms, C.cudaEvent_t(other.event), C.cudaEvent_t(e.event))

	if err != C.cudaSuccess {
		return 0, NewCudaError(uint32(err))
	}

	return float32(ms), nil
}

func (e CudaEvent) Query() (bool, error) {
	err := C.cudaEventQuery(C.cudaEvent_t(e.event))

	if err == C.cudaSuccess {
		return true, nil
	} else if err == C.cudaErrorNotReady {
		return false, nil
	}

	return false, NewCudaError(uint32(err))
}
