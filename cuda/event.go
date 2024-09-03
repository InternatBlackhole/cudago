package cuda

//#include <cuda.h>
//#include <stdlib.h>
import "C"

type CudaEvent struct {
	event C.CUevent
}

type CudaEventFlag uint32

func NewEvent() (*CudaEvent, error) {
	return NewEventCustomFlags(CU_EVENT_DEFAULT)
}

func NewEventCustomFlags(event_flags CudaEventFlag) (*CudaEvent, error) {
	var event C.CUevent
	stat := C.cuEventCreate(&event, C.uint(event_flags))

	if stat != C.CUDA_SUCCESS {
		return nil, NewCudaError(uint32(stat))
	}

	return &CudaEvent{event}, nil
}

func (e *CudaEvent) Destroy() error {
	stat := C.cuEventDestroy(e.event)

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

// TODO: Implement the stream parameter
// TODO: Implement the flags parameter
func (e *CudaEvent) Record( /*stream *CudaStream*/ ) error {
	stat := C.cuEventRecord(e.event, nil)

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func (e *CudaEvent) Synchronize() error {
	stat := C.cuEventSynchronize(e.event)

	if stat != C.CUDA_SUCCESS {
		return NewCudaError(uint32(stat))
	}

	return nil
}

func EventElapsedTime(start, end *CudaEvent) (float32, error) {
	var time float32
	stat := C.cuEventElapsedTime((*C.float)(&time), start.event, end.event)

	if stat != C.CUDA_SUCCESS {
		return 0, NewCudaError(uint32(stat))
	}

	return time, nil
}

const (
	CU_EVENT_DEFAULT        CudaEventFlag = 0x0 // Default event flag
	CU_EVENT_BLOCKING_SYNC  CudaEventFlag = 0x1 // Event uses blocking synchronization
	CU_EVENT_DISABLE_TIMING CudaEventFlag = 0x2 // Event will not record timing data
	CU_EVENT_INTERPROCESS   CudaEventFlag = 0x4 // Event is suitable for interprocess use. CU_EVENT_DISABLE_TIMING must be set
)
